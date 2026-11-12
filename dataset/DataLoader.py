import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .BreastCancerDataset import BreastCancerDataset
from .preprocess import IMAGE_SIZE
from .splits import sample_per_class, split_dataframe


def create_dataloaders(
                       dataset_path,
                       random_state = 1381,
                       test_size=0.1,
                       validation_size=0.1,
                       image_size=IMAGE_SIZE,
                       batch_size=64,
                       num_workers=4,
                       samples_per_class=None):

    df = pd.read_csv(os.path.join(dataset_path, "train_labels.csv"))
    if samples_per_class is not None:
        df = sample_per_class(df, samples_per_class)
    train_df, val_df, test_df = split_dataframe(
        df,
        test_size=test_size,
        validation_size=validation_size,
        random_state=random_state,
    )
    images_dir = os.path.join(dataset_path, "images")
    print(f"Created Dataframes {train_df.shape}, {val_df.shape}, {test_df.shape}")

    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = BreastCancerDataset(
        train_df,
        images_dir=images_dir,
        image_size=image_size,
        transform=train_transforms,
    )
    val_dataset = BreastCancerDataset(
        val_df,
        images_dir=images_dir,
        image_size=image_size,
        transform=test_transforms,
    )
    test_dataset = BreastCancerDataset(
        dataframe=test_df,
        images_dir=images_dir,
        image_size=image_size,
        transform=test_transforms,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)
    return train_loader, val_loader,test_loader


def show_batch(loader):
    images, labels = next(iter(loader))

    plt.figure(figsize=(16, 8))
    for i in range(min(4, len(images))):
        plt.subplot(1, 4, i + 1)
        img = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_img = std * img + mean
        np_img = np.clip(np_img, 0, 1)

        img = np.clip(np_img, 0, 1)

        plt.imshow(img)
        plt.title(f"Label: {labels[i].item()}")
        plt.axis("off")
    plt.show()
