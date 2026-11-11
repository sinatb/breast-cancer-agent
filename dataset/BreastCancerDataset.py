import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .preprocess import preprocess


class BreastCancerDataset(Dataset):

    def __init__(self, dataframe, images_dir, image_size=(224,224), transform=None):
        self.df = dataframe
        self.images_dir = images_dir
        self.transform = transform
        self.image_size = image_size

    @property
    def labels(self) -> np.ndarray:
        return self.df["label"].to_numpy(dtype=np.int64) - 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if index >= len(self.df):
            raise IndexError(f"index {index} out of range")
        row = self.df.iloc[index]
        img_path = os.path.join(self.images_dir, str(row["filename"]))
        img = preprocess(img_path, self.image_size)
        image = Image.fromarray(img)

        label = int(row["label"]) - 1

        if self.transform:
            image = self.transform(image)

        return image, label
