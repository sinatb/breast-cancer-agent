"""BreastCancerDataset: indexing, file lookup and the 1-based -> 0-based label boundary.

``preprocess`` is replaced by a stub, so no image is read and no watershed runs.
"""

import numpy as np
import pandas as pd
import pytest

import dataset.BreastCancerDataset as bcd
from dataset.BreastCancerDataset import BreastCancerDataset


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
