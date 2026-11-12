"""fine_tune internals: class weights, label handling, input gradients and checkpoints.

Uses the tiny models from conftest and random tensors; nothing is downloaded and no real image is read.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import IMAGE_SIZE
from torch import nn
from torch.utils.data import DataLoader, Dataset

import finetuning.FineTuner as finetuner
from finetuning.FineTuner import (
    _compute_class_weights,
    _generate_model_dir,
    _train_epoch,
    _validate,
    fine_tune,
    load_checkpoint,
)
from finetuning.FineTuneStrategy import FineTuneStrategy


class CountingDataset(Dataset):
    """Random images with 0-based labels, exposed like BreastCancerDataset.labels, counting every item load."""

    def __init__(self, labels: list[int], seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randn(len(labels), 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
        self._labels = labels
        self.loads = 0

    @property
    def labels(self):
        return list(self._labels)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, index):
        self.loads += 1
        return self.images[index], self._labels[index]


LABELS = [0, 1, 2, 3, 4, 0, 1, 2]


def _loader(seed: int = 0) -> DataLoader:
    return DataLoader(CountingDataset(LABELS, seed=seed), batch_size=4, shuffle=False)


def _run(model, strategy, tmp_path, epochs: int = 1):
    train, val = _loader(0), _loader(1)
    result = fine_tune(
        model=model,
        strategy=strategy,
        train_loader=train,
        val_loader=val,
        model_dir=tmp_path,
        num_epochs=epochs,
        learning_rate=1e-2,
        weight_decay=0.0,
        device="cpu",
        verbose=False,
    )
    return result, train, val


# --------------------------------------------------------------------------- class weights (Q6)


def test_class_weights_from_labels():
    weights = _compute_class_weights([0, 0, 0, 1], num_classes=5, device="cpu")

    raw = torch.tensor([4 / (5 * 3), 4 / (5 * 1), 1.0, 1.0, 1.0])
    torch.testing.assert_close(weights, raw / raw.mean())


def test_class_weights_do_not_load_images(make_model, tmp_path):
    """Q6: counting labels used to run the full image pipeline over the training set once more."""
    _, train, _ = _run(make_model("resnet50"), FineTuneStrategy.LINEAR_PROBE, tmp_path, epochs=1)
    assert train.dataset.loads == len(LABELS), "training set was loaded more than once per epoch"


# --------------------------------------------------------------------------- labels and inputs


def test_train_epoch_reports_training_accuracy():
    """Q2: the agent is told to compare train and validation accuracy, so training accuracy must exist."""
    model = nn.Linear(5, 5, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(5))
    labels = torch.tensor([0, 1, 2, 3])
    images = nn.functional.one_hot(torch.tensor([0, 1, 2, 4]), 5).float() * 10  # the last one is predicted wrong
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    loss, accuracy = _train_epoch(model, [(images, labels)], optimizer, nn.CrossEntropyLoss(), "cpu")

    assert accuracy == 0.75
    assert loss > 0


def _val_metrics(accuracy: float, f1_macro: float) -> dict:
    return {
        "loss": 1.0,
        "accuracy": accuracy,
        "precision": 0.5,
        "recall": 0.5,
        "f1_micro": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": 0.5,
        "confusion_matrix": np.zeros((5, 5), dtype=int),
    }


def test_best_epoch_is_chosen_by_macro_f1(make_model, tmp_path, monkeypatch):
    """Epoch 1 looks like "always predict class 0": high accuracy, low macro-F1. Epoch 2 is the better model."""
    epochs = iter([_val_metrics(accuracy=0.64, f1_macro=0.16), _val_metrics(accuracy=0.50, f1_macro=0.40)])
    monkeypatch.setattr(finetuner, "_validate", lambda *args, **kwargs: next(epochs))

    result, _, _ = _run(make_model("resnet50"), FineTuneStrategy.LINEAR_PROBE, tmp_path, epochs=2)

    assert result.f1_macro == 0.40
    assert result.accuracy == 0.50


def test_result_carries_training_metrics(make_model, tmp_path):
    result, _, _ = _run(make_model("resnet50"), FineTuneStrategy.LINEAR_PROBE, tmp_path, epochs=2)

    assert 0.0 <= result.train_accuracy <= 1.0
    assert 0.0 < result.train_loss < float("inf")
    assert "train_accuracy" in result.get_summary()
    assert "confusion_matrix" in result.get_summary()


def test_validate_treats_labels_as_class_indices():
    """Labels are 0-based from the dataset on; the loop must not shift them again."""
    labels = torch.arange(5)
    images = nn.functional.one_hot(labels, 5).float() * 10  # nn.Identity then predicts each label exactly
    metrics = _validate(nn.Identity(), [(images, labels)], nn.CrossEntropyLoss(), "cpu")
    assert metrics["accuracy"] == 1.0


def test_train_epoch_does_not_track_input_gradients():
    """Q9: ``images.requires_grad_(True)`` computed gradients for input pixels for no benefit."""
    model = nn.Linear(4, 5)
    images, labels = torch.randn(3, 4), torch.tensor([0, 1, 4])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    _train_epoch(model, [(images, labels)], optimizer, nn.CrossEntropyLoss(), "cpu")

    assert not images.requires_grad
    assert images.grad is None


# --------------------------------------------------------------------------- checkpoints (Q10, Q5)


def test_checkpoint_dirs_are_unique_per_run(tmp_path):
    """Q5: repeating a config overwrote the earlier run's checkpoint folder."""
    model = SimpleNamespace(model_name="resnet50")
    args = (model, FineTuneStrategy.LORA, 1e-4, 1e-4, 5, tmp_path)
    first, second = _generate_model_dir(*args), _generate_model_dir(*args)

    assert first != second
    for path in (first, second):
        assert str(path).startswith(str(tmp_path))
        assert "'" not in str(path)


def _best_checkpoint(result) -> dict:
    return torch.load(f"{result.model_dir}/best_model.pt", map_location="cpu", weights_only=False)


@pytest.mark.parametrize("strategy", list(FineTuneStrategy))
def test_checkpoint_holds_only_trainable_params_and_buffers(strategy, make_model, tmp_path):
    """Q10: each checkpoint stored the whole model and optimizer (about 1 GB for ViT under FULL_FT)."""
    model = make_model("resnet50")
    result, _, _ = _run(model, strategy, tmp_path)
    checkpoint = _best_checkpoint(result)

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    buffers = {n for n, _ in model.named_buffers()}
    assert "optimizer_state_dict" not in checkpoint
    assert set(checkpoint["state_dict"]) == trainable | buffers
    assert checkpoint["model_name"] == "resnet50"
    assert checkpoint["strategy"] == strategy.name


@pytest.mark.parametrize("name", ["resnet50", "vit_b_16"])
@pytest.mark.parametrize("strategy", list(FineTuneStrategy))
def test_checkpoint_round_trip(name, strategy, make_model, tmp_path):
    trained = make_model(name)
    result, _, _ = _run(trained, strategy, tmp_path)

    restored = make_model(name)  # same seed, so the same frozen weights as `trained` started with
    info = load_checkpoint(f"{result.model_dir}/best_model.pt", restored, device="cpu")

    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE, generator=torch.Generator().manual_seed(7))
    trained.eval()
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(x), trained(x))
    assert info["epoch"] == 0


def test_load_checkpoint_rejects_missing_trainable_weights(make_model, tmp_path):
    result, _, _ = _run(make_model("resnet50"), FineTuneStrategy.LINEAR_PROBE, tmp_path)
    path = f"{result.model_dir}/best_model.pt"
    checkpoint = _best_checkpoint(result)
    head_key = next(k for k in checkpoint["state_dict"] if k.endswith("fc.weight"))
    del checkpoint["state_dict"][head_key]
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="fc.weight"):
        load_checkpoint(path, make_model("resnet50"), device="cpu")
