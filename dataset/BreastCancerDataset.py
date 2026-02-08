from PIL import Image
from .preprocess import preprocess
from torch.utils.data import Dataset

class BreastCancerDataset(Dataset):
    def __init__(self, dataframe, image_size=(224,224), transform=None):
        self.df = dataframe
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if index > len(self.df):
            raise IndexError(f"index {index} out of range")
        image_number = str(self.df.iloc[index]["filename"])[6:10]
        img = preprocess(image_number, self.image_size)
        image = Image.fromarray(img)

        label = int(self.df.iloc[index]["label"])

        if self.transform:
            image = self.transform(image)

        return image, label