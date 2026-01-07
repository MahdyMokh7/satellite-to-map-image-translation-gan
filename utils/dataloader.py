import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SatelliteMapDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Args:
            data_dir (str): Directory path to the processed dataset (train/val/test).
            split (str): The dataset split (train/val/test).
            transform (callable, optional): Optional transformation to be applied on a sample.
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.satellite_dir = os.path.join(self.data_dir, split, 'satellite')
        self.map_dir = os.path.join(self.data_dir, split, 'map')

        self.satellite_files = sorted(os.listdir(self.satellite_dir))
        self.map_files = sorted(os.listdir(self.map_dir))

    def __len__(self):
        return len(self.satellite_files)

    def __getitem__(self, idx):
        satellite_path = os.path.join(self.satellite_dir, self.satellite_files[idx])
        map_path = os.path.join(self.map_dir, self.map_files[idx])

        satellite_image = cv2.imread(satellite_path)
        map_image = cv2.imread(map_path)

        satellite_image = cv2.cvtColor(satellite_image, cv2.COLOR_BGR2RGB)
        map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)

        satellite_image = Image.fromarray(satellite_image)
        map_image = Image.fromarray(map_image)

        if self.transform:
            satellite_image = self.transform(satellite_image)
            map_image = self.transform(map_image)

        return satellite_image, map_image


def get_dataloader(data_dir, split='train', batch_size=8, shuffle=True, transform=None):
    """
    Args:
        data_dir (str): The directory where the processed dataset is located.
        split (str): The dataset split (train/val/test).
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        transform (callable, optional): Transformations to apply on the images.

    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = SatelliteMapDataset(data_dir, split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])