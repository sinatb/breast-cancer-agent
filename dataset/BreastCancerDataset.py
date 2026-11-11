import os

from PIL import Image
from torch.utils.data import Dataset

from .preprocess import preprocess


class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, images_dir, image_size=(224,224), transform=None):
        self.df = dataframe
        self.images_dir = images_dir
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if index > len(self.df):
            raise IndexError(f"index {index} out of range")
        image_number = str(self.df.iloc[index]["filename"])[6:10]
        img_path = os.path.join(self.images_dir, f"train_{image_number}.png")
        img = preprocess(img_path, self.image_size)
        image = Image.fromarray(img)

        label = int(self.df.iloc[index]["label"])

        if self.transform:
            image = self.transform(image)

        return image, label
