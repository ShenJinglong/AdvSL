
import os
from random import shuffle
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image

class DigitsDataset(Dataset):
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
    def __init__(self, path, percent, batch_size) -> None:
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
                shuffle=True
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
                shuffle=False
            ) for name, channel, transform in zip(self.datasets, channels, transforms)
        }

    def get_trainloaders(self, names):
        trainloaders = []
        for name in names:
            trainloaders.append(self.__trainloaders[name])
        return trainloaders

    def get_testloaders(self, names):
        testloaders = []
        for name in names:
            testloaders.append(self.__testloaders[name])
        return testloaders