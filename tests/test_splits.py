"""Correctness and seeding of the train / val / test split."""

import numpy as np
import pandas as pd
import pytest

from dataset.splits import sample_per_class, split_dataframe


def test_split_sizes(labels_df):
    train_df, val_df, test_df = split_dataframe(labels_df)
    assert (len(train_df), len(val_df), len(test_df)) == (800, 100, 100)


def test_splits_are_disjoint_and_cover_everything(labels_df):
    train_df, val_df, test_df = split_dataframe(labels_df)
    train, val, test = (set(df["filename"]) for df in (train_df, val_df, test_df))

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)

    combined = pd.concat([train_df, val_df, test_df])
    assert not combined["filename"].duplicated().any()
    assert set(combined["filename"]) == set(labels_df["filename"])
    # Rows are moved intact: each filename keeps its label.
    pd.testing.assert_frame_equal(combined.sort_index(), labels_df)


def test_split_does_not_mutate_input(labels_df):
    original = labels_df.copy()
    split_dataframe(labels_df)
    pd.testing.assert_frame_equal(labels_df, original)


def test_same_seed_gives_same_split_regardless_of_global_rng(labels_df):
    np.random.seed(0)
    first = split_dataframe(labels_df, random_state=42)
    np.random.seed(12345)
    second = split_dataframe(labels_df, random_state=42)

    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_split(labels_df):
    *_, test_a = split_dataframe(labels_df, random_state=1)
    *_, test_b = split_dataframe(labels_df, random_state=2)
    assert set(test_a["filename"]) != set(test_b["filename"])


def test_default_seed_is_1381(labels_df):
    default = split_dataframe(labels_df)
    explicit = split_dataframe(labels_df, random_state=1381)
    for a, b in zip(default, explicit, strict=True):
        pd.testing.assert_frame_equal(a, b)


@pytest.mark.xfail(strict=True, reason="E2 (goals.md): split is not stratified by label")
@pytest.mark.parametrize("split_index, split_name", [(0, "train"), (1, "val"), (2, "test")])
def test_split_is_stratified(labels_df, split_index, split_name):
    split = split_dataframe(labels_df)[split_index]
    fraction = len(split) / len(labels_df)
    expected = labels_df["label"].value_counts() * fraction
    actual = split["label"].value_counts().reindex(expected.index, fill_value=0)

    off = (actual - expected).abs()
    assert (off <= 1).all(), f"{split_name} class counts {actual.to_dict()} vs expected {expected.round(1).to_dict()}"


def test_sample_per_class_takes_the_same_number_from_every_class(labels_df):
    sample = sample_per_class(labels_df, 10)

    assert sample["label"].value_counts().to_dict() == {label: 10 for label in range(1, 6)}
    assert set(sample["filename"]) <= set(labels_df["filename"])


def test_sample_per_class_is_deterministic_and_keeps_rows_intact(labels_df):
    first, second = sample_per_class(labels_df, 4), sample_per_class(labels_df, 4)

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first, labels_df.loc[first.index])


def test_sample_per_class_keeps_small_classes_whole(labels_df):
    sample = sample_per_class(labels_df, 60)

    assert (sample["label"] == 5).sum() == 48
    assert (sample["label"] == 1).sum() == 60
