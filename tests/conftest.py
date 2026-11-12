"""Shared fixtures: synthetic labels, tiny models with real module names, random batches.

Nothing here touches the real dataset, the network, a GPU or the LLM.
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import timm
import torch
from timm.models.densenet import DenseNet
from timm.models.resnet import Bottleneck, ResNet
from timm.models.swin_transformer import SwinTransformer
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import finetuning.BreastCancerModel as bcm
from config import Settings
from finetuning.BreastCancerModel import BreastCancerModel
from finetuning.FineTuner import _apply_strategy, _setup_optimizer_and_scheduler, _train_epoch
from finetuning.FineTuneResult import FineTuneResult
from finetuning.FineTuneStrategy import FineTuneStrategy
from finetuning.ModelCatalog import get_default_catalog

# Class counts of the real train_labels.csv (labels 1..5); only the counts, no data.
CLASS_COUNTS = {1: 603, 2: 212, 3: 71, 4: 66, 5: 48}

IMAGE_SIZE = 32

# Same timm classes as the catalog models, so module names, and therefore LoRA target matching, are identical, but
# tiny and randomly initialised so no weights are downloaded.
TINY_MODELS = {
    "vit_b_16": lambda: VisionTransformer(
        img_size=IMAGE_SIZE, patch_size=16, embed_dim=32, depth=1, num_heads=2, mlp_ratio=2.0
    ),
    "swin_s": lambda: SwinTransformer(
        img_size=IMAGE_SIZE, patch_size=4, embed_dim=16, depths=(1, 1), num_heads=(1, 1), window_size=2
    ),
    # timm zero-initialises the last BN of each block by default, which blocks gradients to the block's convs at random
    # init. Pretrained weights have no such zeros, so turn it off to match the real model.
    "resnet50": lambda: ResNet(Bottleneck, [1, 1, 1, 1], zero_init_last=False),
    "densenet161": lambda: DenseNet(growth_rate=8, block_config=(1, 1)),
    # Real layout on purpose: it is already small and the depthwise convs matter for LoRA.
    "mobilenet_v3_large": lambda: timm.create_model("mobilenetv3_large_100", pretrained=False),
}

MODEL_NAMES = list(TINY_MODELS)


@pytest.fixture
def labels_df() -> pd.DataFrame:
    labels = np.concatenate([np.full(n, label) for label, n in CLASS_COUNTS.items()])
    labels = np.random.default_rng(0).permutation(labels)
    return pd.DataFrame(
        {
            "filename": [f"train_{i:04d}.png" for i in range(len(labels))],
            "label": labels,
        }
    )


@pytest.fixture
def make_model(monkeypatch):
    """Factory: ``make_model(name)`` returns a BreastCancerModel built from a tiny variant."""

    def _make(name: str) -> BreastCancerModel:
        monkeypatch.setitem(bcm.model_map, name, lambda: TINY_MODELS[name]())
        torch.manual_seed(0)
        return BreastCancerModel(get_default_catalog().get_model_data(name))

    return _make


@pytest.fixture
def random_loader() -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2])  # 0-based, as BreastCancerDataset returns them
    return DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)


def head_parameters(model: BreastCancerModel) -> set[nn.Parameter]:
    """Head parameters by identity, so PEFT's renaming does not matter."""
    return set(model.adapter.get_classifier(model.base_model).parameters())


def snapshot_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def snapshot_buffers(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: b.detach().clone() for name, b in model.named_buffers()}


def apply_strategy(model: BreastCancerModel, strategy: FineTuneStrategy) -> None:
    _apply_strategy(model, strategy)


def train_one_epoch(model: BreastCancerModel, loader: DataLoader) -> None:
    """Run one epoch through the project's real optimizer setup and training loop."""
    optimizer, _ = _setup_optimizer_and_scheduler(model, learning_rate=1e-2, weight_decay=0.0, num_epochs=1)
    _train_epoch(model, loader, optimizer, nn.CrossEntropyLoss(), "cpu")


class TrainingStubs:
    """Stands in for model building, data loading and training behind the agent's tools, and records every call."""

    def __init__(self):
        self.built_models: list[str] = []
        self.loader_batch_sizes: list[int] = []
        self.loader_kwargs: list[dict] = []
        self.fine_tune_calls: list[dict] = []
        self.memory_releases = 0
        self.fine_tune_error: Exception | None = None

    def get_model(self, model_data):
        self.built_models.append(model_data.name)
        return SimpleNamespace(name=model_data.name)

    def create_dataloaders(self, dataset_path, batch_size, **kwargs):
        self.loader_batch_sizes.append(batch_size)
        self.loader_kwargs.append(kwargs)
        return "train_loader", "val_loader", "test_loader"

    def fine_tune(self, model, strategy, learning_rate, num_epochs, weight_decay, **kwargs):
        self.fine_tune_calls.append({"model": model.name, "strategy": strategy, "epochs": num_epochs, **kwargs})
        if self.fine_tune_error is not None:
            raise self.fine_tune_error
        return FineTuneResult(
            model_name=model.name,
            fine_tune_strategy=strategy.name,
            model_dir="unused",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=num_epochs,
            loss=1.0,
            accuracy=0.5,
            precision=0.4,
            recall=0.3,
            f1_macro=0.2,
            confusion_matrix=[[7, 1], [2, 5]],
            train_loss=0.25,
            train_accuracy=0.875,
        )

    def release_memory(self):
        self.memory_releases += 1


@pytest.fixture
def training_stubs(monkeypatch) -> TrainingStubs:
    """Replace everything heavy in ``agent.tools``, so the tools run without a dataset, weights or training."""
    import agent.tools as agent_tools

    stubs = TrainingStubs()
    for name in ("get_model", "create_dataloaders", "fine_tune", "release_memory"):
        monkeypatch.setattr(agent_tools, name, getattr(stubs, name), raising=False)
    return stubs


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(dataset_path=tmp_path / "data", model_dir=tmp_path / "runs")
