import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, Tuple, List
import os
import json
from datetime import datetime
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from BreastCancerModel import BreastCancerModel
from FineTuneStrategy import FineTuneStrategy
from peft import get_peft_model, LoraConfig, TaskType


def fine_tune(
        model: BreastCancerModel,
        strategy: FineTuneStrategy,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./checkpoints",
        verbose: bool = True,
) -> Dict:

    model = model.to(device)
    _apply_strategy(model, strategy)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.train()
    
    optimizer, scheduler = _setup_optimizer_and_scheduler(model, learning_rate, weight_decay, num_epochs)

    class_weights = _compute_class_weights(train_loader, device)
    if verbose:
        print(f"class_weights: {class_weights}\n")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    metrics = {
        "train_losses": [],
        "val_losses": [],
        "val_accuracies": [],
        "val_f1_scores": [],
        "val_precision_scores": [],
        "val_recall_scores": [],
        "val_f1_macro": [],
        "val_f1_weighted": [],
        "val_confusion_matrices": [],
        "best_val_accuracy": 0.0,
        "best_val_f1": 0.0,
        "best_epoch": 0,
        "strategy": strategy.name,
        "learning_rate": learning_rate,
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    if verbose:
        print(f"Starting fine-tuning with strategy: {strategy.name}")
        print(f"Device: {device}, Epochs: {num_epochs}, LR: {learning_rate}")
        print("-" * 80)

    for epoch in range(num_epochs):
        # Training
        train_loss = _train_epoch(model, train_loader, optimizer, loss_fn, device)
        metrics["train_losses"].append(train_loss)

        val_metrics = _validate(model, val_loader, loss_fn, device)

        metrics["val_losses"].append(val_metrics["loss"])
        metrics["val_accuracies"].append(val_metrics["accuracy"])
        metrics["val_f1_scores"].append(val_metrics["f1_micro"])
        metrics["val_precision_scores"].append(val_metrics["precision"])
        metrics["val_recall_scores"].append(val_metrics["recall"])
        metrics["val_f1_macro"].append(val_metrics["f1_macro"])
        metrics["val_f1_weighted"].append(val_metrics["f1_weighted"])
        metrics["val_confusion_matrices"].append(val_metrics["confusion_matrix"].tolist())

        if val_metrics["accuracy"] > metrics["best_val_accuracy"]:
            metrics["best_val_accuracy"] = val_metrics["accuracy"]
            metrics["best_epoch"] = epoch

        if val_metrics["f1_weighted"] > metrics["best_val_f1"]:
            metrics["best_val_f1"] = val_metrics["f1_weighted"]

        scheduler.step()

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1 (Weighted): {val_metrics['f1_weighted']:.4f}")
            print(f"  Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall: {val_metrics['recall']:.4f}")
            print("-" * 80)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_without_improvement = 0
            checkpoint_path = _save_checkpoint(
                model, optimizer, epoch, metrics, checkpoint_dir, is_best=True
            )
            metrics["best_model_path"] = checkpoint_path

            metrics["best_epoch_metrics"] = val_metrics
        else:
            epochs_without_improvement += 1

    if verbose:
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best Val Accuracy: {metrics['best_val_accuracy']:.4f} at Epoch {metrics['best_epoch'] + 1}")
        print(f"Best Val F1 (Weighted): {metrics['best_val_f1']:.4f}")
        print("=" * 80)

        if "best_epoch_metrics" in metrics:
            print("\nBest Epoch Confusion Matrix:")
            print(metrics["best_epoch_metrics"]["confusion_matrix"])

    return metrics


def _apply_strategy(model: BreastCancerModel, strategy: FineTuneStrategy) -> None:
    if strategy == FineTuneStrategy.FULL_FT:
        model.unfreeze_backbone()
    elif strategy == FineTuneStrategy.LINEAR_PROBE:
        model.freeze_backbone()
    elif strategy == FineTuneStrategy.LORA:
        model.unfreeze_backbone()
        _apply_lora(model)
    else:
        raise ValueError(f"Unknown fine tuning strategy: {strategy}")


def _apply_lora(model: BreastCancerModel) -> None:
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=model.adapter.get_lora_target_modules(),
        lora_dropout=0.1,
        bias="none",
    )

    model.base_model = get_peft_model(model.base_model, lora_config)
    model.base_model.print_trainable_parameters()


def _compute_class_weights(train_loader: DataLoader, device: str, num_classes: int = 5) -> torch.Tensor:
    class_counts = {}

    for images, labels in train_loader:
        for label in labels:
            label_idx = label.item() - 1
            class_counts[label_idx] = class_counts.get(label_idx, 0) + 1

    total_samples = sum(class_counts.values())

    weights = torch.ones(num_classes, device=device)
    for class_idx, count in class_counts.items():
        weights[class_idx] = total_samples / (num_classes * count)

    weights = weights / weights.mean()

    return weights


def _train_epoch(
        model: BreastCancerModel,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: str) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).long() - 1
        
        # Ensure inputs require gradients if needed
        images.requires_grad_(True)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def _validate(
        model: BreastCancerModel,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: str,
        num_classes: int = 5) -> Dict:
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).long() - 1

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    accuracy = (y_pred == y_true).mean()

    # Calculate precision, recall, F1 scores
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


    return {
        "loss": avg_loss,
        "accuracy": float(accuracy),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "confusion_matrix": cm,
    }


def _setup_optimizer_and_scheduler(
        model: BreastCancerModel,
        learning_rate: float,
        weight_decay: float,
        num_epochs: int) -> Tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    return optimizer, scheduler


def _save_checkpoint(
        model: BreastCancerModel,
        optimizer: optim.Optimizer,
        epoch: int,
        metrics: Dict,
        checkpoint_dir: str,
        is_best: bool = False) -> str:

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }

    filename = f"best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
    filepath = os.path.join(checkpoint_dir, filename)

    torch.save(checkpoint, filepath)

    if is_best:
        metrics_path = os.path.join(checkpoint_dir, "best_model_metrics.json")
        # Create a JSON-serializable copy of metrics
        json_metrics = _make_json_serializable(metrics)
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)

    return filepath


def _make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, str):
        return obj
    else:
        return obj


def load_checkpoint(checkpoint_path: str, model: BreastCancerModel, device: str = "cpu") -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint["metrics"],
        "timestamp": checkpoint["timestamp"],
    }


def print_metrics_summary(metrics: Dict) -> None:
    print("\n" + "=" * 80)
    print("TRAINING METRICS SUMMARY")
    print("=" * 80)

    print(f"\nStrategy: {metrics['strategy']}")
    print(f"Learning Rate: {metrics['learning_rate']}")
    print(f"Number of Epochs Trained: {len(metrics['train_losses'])}")
    print(f"Best Epoch: {metrics['best_epoch'] + 1}")

    print("\n" + "-" * 80)
    print("BEST VALIDATION METRICS")
    print("-" * 80)
    print(f"Accuracy: {metrics['best_val_accuracy']:.4f}")
    print(f"F1 Score (Weighted): {metrics['best_val_f1']:.4f}")

    if "best_epoch_metrics" in metrics:
        best = metrics["best_epoch_metrics"]
        print(f"F1 Score (Macro): {best['f1_macro']:.4f}")
        print(f"Precision (Macro): {best['precision']:.4f}")
        print(f"Recall (Macro): {best['recall']:.4f}")

        print("\n" + "-" * 80)
        print("CONFUSION MATRIX (Best Epoch)")
        print("-" * 80)
        print(best["confusion_matrix"])

    print("\n" + "=" * 80)
