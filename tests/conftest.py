"""Shared fixtures: synthetic labels, tiny models with real module names, random batches.

Nothing here touches the real dataset, the network, a GPU or the LLM.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from timm.models.swin_transformer import SwinTransformer
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import mobilenet_v3_large
from torchvision.models.densenet import DenseNet
from torchvision.models.resnet import Bottleneck, ResNet

import finetuning.BreastCancerModel as bcm
from finetuning.BreastCancerModel import BreastCancerModel
from finetuning.FineTuner import _apply_strategy, _setup_optimizer_and_scheduler, _train_epoch
from finetuning.FineTuneStrategy import FineTuneStrategy
from finetuning.ModelCatalog import get_default_catalog

# Class counts of the real train_labels.csv (labels 1..5); only the counts, no data.
CLASS_COUNTS = {1: 603, 2: 212, 3: 71, 4: 66, 5: 48}

IMAGE_SIZE = 32

# Same classes as the catalog models (timm for ViT and Swin, torchvision for the CNNs), so module names, and
# therefore LoRA target matching, are identical, but tiny and randomly initialised so no weights are downloaded.
TINY_MODELS = {
    "vit_b_16": lambda: VisionTransformer(
        img_size=IMAGE_SIZE, patch_size=16, embed_dim=32, depth=1, num_heads=2, mlp_ratio=2.0
    ),
    "swin_s": lambda: SwinTransformer(
        img_size=IMAGE_SIZE, patch_size=4, embed_dim=16, depths=(1, 1), num_heads=(1, 1), window_size=2
    ),
    "resnet50": lambda: ResNet(Bottleneck, [1, 1, 1, 1]),
    "densenet161": lambda: DenseNet(growth_rate=8, block_config=(1, 1), num_init_features=8),
    # Real layout on purpose: it is already small and the depthwise convs matter for LoRA.
    "mobilenet_v3_large": lambda: mobilenet_v3_large(weights=None),
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
    labels = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3])  # 1-based, as in train_labels.csv
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
