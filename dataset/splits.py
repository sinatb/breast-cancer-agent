
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.1,
    validation_size: float = 0.1,
    random_state: int = 1381,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    val_size = validation_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_df["label"],
    )
    return train_df, val_df, test_df


def sample_per_class(df: pd.DataFrame, samples_per_class: int) -> pd.DataFrame:
    return df.groupby("label", sort=False).head(samples_per_class)
