"""Every catalog model comes from timm with ImageNet weights, so one weight cache (HF_HOME) covers them all.

Only timm's registry is read; no weights are downloaded.
"""

from functools import partial

import pytest
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import finetuning.BreastCancerModel as bcm
from finetuning.ModelCatalog import get_default_catalog

CATALOG_NAMES = list(get_default_catalog().model_dict)


def test_every_catalog_model_has_a_builder():
    assert set(bcm.model_map) == set(CATALOG_NAMES)


@pytest.mark.parametrize("name", CATALOG_NAMES)
def test_catalog_model_is_a_pretrained_timm_tag(name):
    builder = bcm.model_map[name]

    assert isinstance(builder, partial) and builder.func is timm.create_model
    assert builder.keywords.get("pretrained") is True
    tag = builder.args[0]
    assert tag in timm.list_pretrained(), f"{name}: {tag} has no pretrained weights in timm"


@pytest.mark.parametrize("name", CATALOG_NAMES)
def test_catalog_model_uses_the_loader_normalization(name):
    # dataset/DataLoader.py normalizes with the ImageNet mean and std for every model.
    cfg = timm.models.get_pretrained_cfg(bcm.model_map[name].args[0])

    assert tuple(cfg.mean) == IMAGENET_DEFAULT_MEAN
    assert tuple(cfg.std) == IMAGENET_DEFAULT_STD
