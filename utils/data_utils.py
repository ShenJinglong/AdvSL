
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
        self.__percent = percent
        self.__batch_size = batch_size
        self.__path = path
        self.__datasets = {
            "MNIST": {
                "channel": 1,
                "transform": [
                    torchvision.transforms.Grayscale(num_output_channels=3),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            },
            "SVHN": {
                "channel": 3,
                "transform": [
                    torchvision.transforms.Resize((28, 28)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    
                ]
            },
            "USPS": {
                "channel": 1,
                "transform": [
                    torchvision.transforms.Resize((28, 28)),
                    torchvision.transforms.Grayscale(num_output_channels=3),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            },
            "SynthDigits": {
                "channel": 3,
                "transform": [
                    torchvision.transforms.Resize((28, 28)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            },
            "MNIST_M": {
                "channel": 3,
                "transform": [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            },
        }
        self.__aug_transforms = {
            "blur": [torchvision.transforms.GaussianBlur(7)],
            "rot": [torchvision.transforms.RandomRotation(60)],
            "noise": [torchvision.transforms.Lambda(lambda x: x + 0.2*torch.randn_like(x))],
            "bright": [torchvision.transforms.ColorJitter(brightness=1)],
            "hue": [torchvision.transforms.ColorJitter(hue=0.5)],
        }

    def get_trainloaders(self,
        names: List[str]
    ) -> List[torch.utils.data.DataLoader]:
        trainloaders = []
        for name in names:
            substrs = name.split("-")
            trainloaders.append(torch.utils.data.DataLoader(
                DigitsDataset(
                    data_path=os.path.join(self.__path, substrs[0]),
                    channels=self.__datasets[substrs[0]]["channel"],
                    percent=self.__percent,
                    train=True,
                    transform=torchvision.transforms.Compose(
                        self.__datasets[substrs[0]]["transform"] + self.__aug_transforms[substrs[1]] if len(substrs) == 2 else self.__datasets[substrs[0]]["transform"]
                    )
                ),
                batch_size = self.__batch_size,
                shuffle=True,
                drop_last = True,
                pin_memory = True,
                num_workers = 4
            ))
        return trainloaders

    def get_testloaders(self,
        names: List[str]
    ) -> List[torch.utils.data.DataLoader]:
        testloaders = []
        for name in names:
            substrs = name.split("-")
            testloaders.append(torch.utils.data.DataLoader(
                DigitsDataset(
                    data_path=os.path.join(self.__path, substrs[0]),
                    channels=self.__datasets[substrs[0]]["channel"],
                    percent=self.__percent,
                    train=False,
                    transform=torchvision.transforms.Compose(
                        self.__datasets[substrs[0]]["transform"] + self.__aug_transforms[substrs[1]] if len(substrs) == 2 else self.__datasets[substrs[0]]["transform"]
                    )
                ),
                batch_size = self.__batch_size,
                shuffle=False,
                drop_last = False,
                pin_memory = True,
                num_workers = 4
            ))
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
