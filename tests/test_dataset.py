"""BreastCancerDataset: indexing, file lookup and the 1-based -> 0-based label boundary.

``preprocess`` is replaced by a stub, so no image is read and no watershed runs.
"""

import numpy as np
import pandas as pd
import pytest

import dataset.BreastCancerDataset as bcd
from dataset.BreastCancerDataset import BreastCancerDataset
from dataset.splits import sample_per_class


@pytest.fixture
def opened_paths(monkeypatch) -> list[str]:
    paths: list[str] = []

    def fake_preprocess(path, image_size):
        paths.append(path)
        return np.zeros(image_size, dtype=np.uint8)

    monkeypatch.setattr(bcd, "preprocess", fake_preprocess)
    return paths


def _dataset(filenames: list[str], labels: list[int], images_dir: str = "imgs") -> BreastCancerDataset:
    df = pd.DataFrame({"filename": filenames, "label": labels})
    return BreastCancerDataset(df, images_dir=images_dir, image_size=(8, 8))


def test_labels_are_zero_based(opened_paths):
    ds = _dataset(["train_0001.png", "train_0002.png"], [1, 5])
    assert [ds[i][1] for i in range(len(ds))] == [0, 4]


def test_labels_property_is_zero_based_without_loading_images(opened_paths):
    ds = _dataset(["a.png", "b.png", "c.png"], [1, 3, 5])
    np.testing.assert_array_equal(ds.labels, [0, 2, 4])
    assert opened_paths == []


@pytest.mark.parametrize("index", [2, 3])
def test_index_past_end_raises(opened_paths, index):
    """Q11: ``index > len`` let index == len through; only pandas' own iloc check raised the IndexError."""
    ds = _dataset(["train_0001.png", "train_0002.png"], [1, 2])
    with pytest.raises(IndexError):
        ds[index]


@pytest.mark.parametrize("filename", ["train_0042.png", "train_12345.png", "scan.png"])
def test_opens_the_file_named_in_the_csv(opened_paths, filename):
    """Q11: the ``[6:10]`` slice assumed ``train_`` plus exactly four digits."""
    ds = _dataset([filename], [1], images_dir="imgs")
    ds[0]
    assert opened_paths == [f"imgs/{filename}"]


def _train_order(tmp_path, labels_df, torch_seed: int) -> list[int]:
    import torch

    from dataset.DataLoader import create_dataloaders

    torch.manual_seed(torch_seed)
    train_loader, _, _ = create_dataloaders(tmp_path, image_size=(8, 8), batch_size=4, num_workers=0)
    return [int(label) for _, labels in train_loader for label in labels]


def test_train_loader_order_does_not_depend_on_global_rng(tmp_path, labels_df, opened_paths):
    """E1: the shuffle order must come from the split seed, not from whatever the global torch RNG holds."""
    sample_per_class(labels_df, 12).to_csv(tmp_path / "train_labels.csv", index=False)
    assert _train_order(tmp_path, labels_df, torch_seed=0) == _train_order(tmp_path, labels_df, torch_seed=1)


def _gradient_image():
    from PIL import Image

    return Image.fromarray(np.tile(np.arange(0, 256, 4, dtype=np.uint8), (64, 1)))


def test_eval_transform_is_deterministic():
    from dataset.DataLoader import EVAL_TRANSFORMS

    image = _gradient_image()
    assert EVAL_TRANSFORMS(image).equal(EVAL_TRANSFORMS(image))


def test_train_transform_augments():
    """Q8: training had no augmentation, so train and test transforms were identical."""
    import torch

    from dataset.DataLoader import EVAL_TRANSFORMS, TRAIN_TRANSFORMS

    torch.manual_seed(0)
    image = _gradient_image()
    first, second = TRAIN_TRANSFORMS(image), TRAIN_TRANSFORMS(image)
    assert first.shape == EVAL_TRANSFORMS(image).shape
    assert not first.equal(second)
