
import os
import random
from typing import List
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image

class DigitsDataset(Dataset):
    # https://github.com/med-air/FedBN/blob/master/utils/data_utils.py
    def __init__(self, data_path, channels, percent=0.1, filename=None, train=True, transform=None):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent*10)):
                        if part == 0:
                            self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                        else:
                            images, labels = np.load(os.path.join(data_path, 'partitions/train_part{}.pkl'.format(part)), allow_pickle=True)
                            self.images = np.concatenate([self.images,images], axis=0)
                            self.labels = np.concatenate([self.labels,labels], axis=0)
                else:
                    self.images, self.labels = np.load(os.path.join(data_path, 'partitions/train_part0.pkl'), allow_pickle=True)
                    data_len = int(self.images.shape[0] * percent*10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)
        else:
            self.images, self.labels = np.load(os.path.join(data_path, filename), allow_pickle=True)

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode='L')
        elif self.channels == 3:
            image = Image.fromarray(image, mode='RGB')
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class DatasetManager():
    def __init__(self,
        path: str,
        percent: float,
        batch_size: float
    ) -> None:
        self.datasets = ["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST_M"]
        channels = [1, 3, 1, 3, 3]
        transforms = [
            torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize((28, 28)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize((28, 28)),
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            torchvision.transforms.Compose([
                torchvision.transforms.Resize((28, 28)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
        ]
        self.__trainloaders = {
            name: torch.utils.data.DataLoader(
                DigitsDataset(
                    data_path=path + name,
                    channels=channel,
                    percent=percent,
                    train=True,
                    transform=transform
                ),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True
            ) for name, channel, transform in zip(self.datasets, channels, transforms)
        }
        self.__testloaders = {
            name: torch.utils.data.DataLoader(
                DigitsDataset(
                    data_path=path + name,
                    channels=channel,
                    percent=percent,
                    train=False,
                    transform=transform
                ),
                batch_size=batch_size,
                shuffle=False,
                drop_last=True
            ) for name, channel, transform in zip(self.datasets, channels, transforms)
        }

    def get_trainloaders(self,
        names: List[str]
    ) -> List[torch.utils.data.DataLoader]:
        trainloaders = []
        for name in names:
            trainloaders.append(self.__trainloaders[name])
        return trainloaders

    def get_testloaders(self,
        names: List[str]
    ) -> List[torch.utils.data.DataLoader]:
        testloaders = []
        for name in names:
            testloaders.append(self.__testloaders[name])
        return testloaders

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
