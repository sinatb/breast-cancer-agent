import copy
import json
import os
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .BreastCancerModel import BreastCancerModel
from .FineTuneResult import FineTuneResult
from .FineTuneStrategy import FineTuneStrategy

LORA_SETTINGS = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.1, "bias": "none"}

def fine_tune(
        model: BreastCancerModel,
        strategy: FineTuneStrategy,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_dir: str | os.PathLike,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True,
) -> FineTuneResult:

    model = model.to(device)
    _apply_strategy(model, strategy)
    checkpoint_dir = _generate_model_dir(model, strategy, learning_rate, weight_decay, num_epochs, model_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.train()

    optimizer, scheduler = _setup_optimizer_and_scheduler(model, learning_rate, weight_decay, num_epochs)

    class_weights = _compute_class_weights(train_loader.dataset.labels, device=device)
    if verbose:
        print(f"class_weights: {class_weights}\n")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    best_result = FineTuneResult(model.model_name,
                                 strategy.name,
                                 checkpoint_dir,
                                 learning_rate,
                                 weight_decay,
                                 num_epochs)

    if verbose:
        print(f"Starting fine-tuning with strategy: {strategy.name} of model {model.model_name}.")
        print(f"Device: {device}, Epochs: {num_epochs}, LR: {learning_rate}, WD: {weight_decay}")
        print("-" * 80)

    for epoch in range(num_epochs):
        # Training
        train_loss, train_accuracy = _train_epoch(model, train_loader, optimizer, loss_fn, device)

        val_metrics = _validate(model, val_loader, loss_fn, device)
        current_result = FineTuneResult(model.model_name,
                                        strategy.name,
                                        checkpoint_dir,
                                        learning_rate,
                                        weight_decay,
                                        num_epochs,
                                        val_metrics["loss"],
                                        val_metrics["accuracy"],
                                        val_metrics["precision"],
                                        val_metrics["recall"],
                                        val_metrics["f1_micro"],
                                        val_metrics["f1_macro"],
                                        val_metrics["f1_weighted"],
                                        val_metrics["confusion_matrix"].tolist(),
                                        train_loss,
                                        train_accuracy)


        scheduler.step()

        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train Accuracy: {train_accuracy:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1 (Weighted): {val_metrics['f1_weighted']:.4f}")
            print(f"  Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall: {val_metrics['recall']:.4f}")
            print("-" * 80)

        if current_result.f1_macro > best_result.f1_macro:
            best_result = copy.deepcopy(current_result)
            _save_checkpoint(model, strategy, epoch, best_result.to_dict(), checkpoint_dir)

    if verbose:
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best Val F1 (Macro): {best_result.f1_macro:.4f}")
        print(f"Val Accuracy at that epoch: {best_result.accuracy:.4f}")
        print("=" * 80)
        print(best_result.confusion_matrix)

    return best_result


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
    targets = model.adapter.get_lora_target_modules(model.base_model)
    unmatched = _unmatched_lora_targets(model.base_model, targets)
    if unmatched:
        raise ValueError(f"LoRA targets for {model.model_name} match no module in the model: {unmatched}")

    lora_config = LoraConfig(target_modules=targets, **LORA_SETTINGS)

    head = model.adapter.get_classifier(model.base_model)
    model.base_model = get_peft_model(model.base_model, lora_config)
    for param in head.parameters():
        param.requires_grad = True
    model.base_model.print_trainable_parameters()


def _unmatched_lora_targets(model: nn.Module, targets: list[str]) -> list[str]:
    names = [n for n, _ in model.named_modules()]
    return [t for t in targets if not any(n == t or n.endswith("." + t) for n in names)]


def _compute_class_weights(labels: Sequence[int], num_classes: int = 5, device: str = "cpu") -> torch.Tensor:
    class_counts = {}
    for label in labels:
        class_counts[int(label)] = class_counts.get(int(label), 0) + 1

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
        device: str) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    correct = 0
    seen = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).long()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        correct += (outputs.detach().argmax(dim=1) == labels).sum().item()
        seen += labels.numel()

    mean_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = correct / seen if seen > 0 else 0.0
    return mean_loss, accuracy


def _validate(
        model: BreastCancerModel,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        device: str,
        num_classes: int = 5) -> dict:
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).long()

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
        num_epochs: int) -> tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    return optimizer, scheduler


def _save_checkpoint(
        model: BreastCancerModel,
        strategy: FineTuneStrategy,
        epoch: int,
        metrics: dict,
        checkpoint_dir: str | os.PathLike) -> str:
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n for n, _ in model.named_buffers()}
    state_dict = {k: v for k, v in model.state_dict().items() if k in trainable | buffers}

    checkpoint = {
        "model_name": model.model_name,
        "strategy": strategy.name,
        "lora_settings": LORA_SETTINGS if strategy == FineTuneStrategy.LORA else None,
        "epoch": epoch,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "state_dict": state_dict,
    }

    filepath = os.path.join(checkpoint_dir, "best_model.pt")
    torch.save(checkpoint, filepath)
    metrics_path = os.path.join(checkpoint_dir, "best_model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(_make_json_serializable(metrics), f, indent=2)

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


def load_checkpoint(checkpoint_path: str | os.PathLike,
                    model: BreastCancerModel,
                    device: str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["model_name"] != model.model_name:
        raise ValueError(f"checkpoint is for {checkpoint['model_name']}, not {model.model_name}")

    _apply_strategy(model, FineTuneStrategy[checkpoint["strategy"]])
    state_dict = checkpoint["state_dict"]
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    missing = sorted(trainable - set(state_dict))
    if missing:
        raise ValueError(f"checkpoint lacks trainable weights: {missing}")

    unexpected = model.load_state_dict(state_dict, strict=False).unexpected_keys
    if unexpected:
        raise ValueError(f"checkpoint has weights the model does not: {unexpected}")
    model.to(device)

    return {
        "epoch": checkpoint["epoch"],
        "metrics": checkpoint["metrics"],
        "timestamp": checkpoint["timestamp"],
    }


def _generate_model_dir(model: BreastCancerModel,
                        strategy: FineTuneStrategy,
                        learning_rate: float,
                        weight_decay: float,
                        epochs: int,
                        base_dir: str | os.PathLike) -> str:
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    name = f"{model.model_name}-{strategy.name}-lr{learning_rate:g}-wd{weight_decay:g}-ep{epochs}-{run_id}"
    return str(Path(base_dir) / name)
