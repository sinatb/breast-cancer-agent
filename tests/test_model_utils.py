"""seed_everything: one call makes Python, NumPy and torch draws reproducible."""

import random

import numpy as np
import torch

from utils.model_utils import seed_everything


def _draws() -> tuple[float, float, float]:
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_seed_everything_makes_every_rng_reproducible():
    seed_everything(7)
    first = _draws()
    seed_everything(7)
    assert _draws() == first


def test_different_seeds_give_different_draws():
    seed_everything(1)
    first = _draws()
    seed_everything(2)
    assert _draws() != first
