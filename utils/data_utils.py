
import os
import random
from typing import List
import numpy as np
import torch
import torchvision
from PIL import Image

class DigitsDataset(torch.utils.data.Dataset):
    # https://github.com/med-air/FedBN/blob/master/utils/data_utils.py
    def __init__(self, root_path, domain, channels, percent=0.1, train=True, transform=None):
        data_path = os.path.join(root_path, domain)
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

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.int64).squeeze()

        print(self.images.shape[0])

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

class OfficeCaltech10Dataset(torch.utils.data.Dataset):
    def __init__(self, root_path, domain, channels, percent=1, train=True, transform=None) -> None:
        super().__init__()
        self.__image_names, self.__labels = [], []
        self.__transform = transform
        self.__root_path = root_path
        self.__domain = domain
        classes = ["back_pack", "bike", "calculator", "headphones", "keyboard", "laptop_computer", "monitor", "mouse", "mug", "projector"]
        for i, class_name in enumerate(classes):
            filenames = os.listdir(os.path.join(root_path, domain, class_name))
            if train:
                filenames = filenames[:int(len(filenames)*0.9)]
                if percent <= 1 and percent > 0:
                    filenames = filenames[:int(len(filenames)*percent)]
                elif percent > 1:
                    filenames = filenames[:percent]
                else:
                    raise ValueError("dataset percent error.")
                self.__image_names.extend([class_name + '/' + filename for filename in filenames])
                self.__labels.extend([i]*len(filenames))
            else:
                filenames = filenames[int(len(filenames)*0.9):]
                self.__image_names.extend([class_name + '/' + filename for filename in filenames])
                self.__labels.extend([i]*len(filenames))
        
        print(len(self.__image_names))

    def __len__(self):
        return len(self.__image_names)

    def __getitem__(self, idx):
        with Image.open(os.path.join(self.__root_path, self.__domain, self.__image_names[idx])) as img:
            if self.__transform is not None:
                img = self.__transform(img)
            return img, self.__labels[idx]

class OfficeHomeDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, domain, channels, percent=1, train=True, transform=None) -> None:
        super().__init__()
        self.__image_names, self.__labels = [], []
        self.__transform = transform
        self.__root_path = root_path
        self.__domain = domain
        classes = os.listdir(os.path.join(root_path, domain))
        for i, class_name in enumerate(classes):
            filenames = os.listdir(os.path.join(root_path, domain, class_name))
            if train:
                filenames = filenames[:int(len(filenames)*0.9)]
                if percent <= 1 and percent > 0:
                    filenames = filenames[:int(len(filenames)*percent)]
                elif percent > 1:
                    filenames = filenames[:percent]
                else:
                    raise ValueError("dataset percent error.")
                self.__image_names.extend([class_name + '/' + filename for filename in filenames])
                self.__labels.extend([i]*len(filenames))
            else:
                filenames = filenames[int(len(filenames)*0.9):]
                self.__image_names.extend([class_name + '/' + filename for filename in filenames])
                self.__labels.extend([i]*len(filenames))
        
        print(len(self.__image_names))
    
    def __len__(self):
        return len(self.__image_names)

    def __getitem__(self, idx):
        with Image.open(os.path.join(self.__root_path, self.__domain, self.__image_names[idx])) as img:
            if self.__transform is not None:
                img = self.__transform(img)
            return img, self.__labels[idx]

# class DomainNetDataset(torch.utils.data.Dataset):
#     def __init__(self, root_path, domain, channels, percent=1, train=True, transform=None) -> None:
#         super().__init__()
#         self.__image_names, self.__labels = [], []
#         self.__transform = transform
#         self.__root_path = root_path
#         self.__domain = domain
#         classes = os.listdir(os.path.join(root_path, domain, domain))
#         for i, class_name in enumerate(classes):
#             filenames = os.listdir(os.path.join(root_path, domain, domain, class_name))
#             if train:
#                 filenames = filenames[:int(len(filenames)*0.9)]
#                 if percent <= 1 and percent > 0:
#                     filenames = filenames[:int(len(filenames)*percent)]
#                 elif percent > 1:
#                     filenames = filenames[:percent]
#                 else:
#                     raise ValueError("dataset percent error.")
#                 self.__image_names.extend([class_name + '/' + filename for filename in filenames])
#                 self.__labels.extend([i]*len(filenames))
#             else:
#                 filenames = filenames[int(len(filenames)*0.9):]
#                 self.__image_names.extend([class_name + '/' + filename for filename in filenames])
#                 self.__labels.extend([i]*len(filenames))

#         print(len(self.__image_names))

#     def __len__(self):
#         return len(self.__image_names)

#     def __getitem__(self, idx):
#         with Image.open(os.path.join(self.__root_path, self.__domain, self.__domain, self.__image_names[idx])) as img:
#             if self.__transform is not None:
#                 img = self.__transform(img)
#             return img, self.__labels[idx]



class DomainNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, domain, channels, percent=1, train=True, transform=None) -> None:
        super().__init__()
        if train:
            dataset = torch.load(os.path.join(root_path, domain, f"domain-net_{domain}_train.pth"))
            self.imgs = dataset['imgs']
            self.labels = dataset['labels']
        else:
            dataset = torch.load(os.path.join(root_path, domain, f"domain-net_{domain}_test.pth"))
            self.imgs = dataset['imgs']
            self.labels = dataset['labels']

        self.transform = transform

        print(self.imgs.shape[0])

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

class DatasetManager():
    def __init__(self,
        root_path: str,
        dataset_name: str,
        percent: float,
        batch_size: float
    ) -> None:
        self.__percent = percent
        self.__batch_size = batch_size
        self.__root_path = root_path
        self.__dataset_name = dataset_name
        if dataset_name == "digits":
            self.__Dataset = DigitsDataset
        elif dataset_name == "office-caltech10":
            self.__Dataset = OfficeCaltech10Dataset
        elif dataset_name == "office-home":
            self.__Dataset = OfficeHomeDataset
        elif dataset_name == "domain-net":
            self.__Dataset = DomainNetDataset
        digits_transform = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   
        ]
        office_caltech10_transform = [
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.Lambda(lambda img: torchvision.transforms.Grayscale(num_output_channels=3)(img) if img.mode == 'L' else img),
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        office_home_transform = [
            torchvision.transforms.Resize((64, 64)),
            torchvision.transforms.Lambda(lambda img: torchvision.transforms.Grayscale(num_output_channels=3)(img) if img.mode == 'L' else img),
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        domain_net_transform = [
            
        ]

        self.__datasets = {
            ################################# digits
            "MNIST": {
                "channel": 1,
                "transform": [
                    torchvision.transforms.Grayscale(num_output_channels=3),
                    *digits_transform
                ]
            },
            "SVHN": {
                "channel": 3,
                "transform": [
                    torchvision.transforms.Resize((28, 28)),
                    *digits_transform
                ]
            },
            "USPS": {
                "channel": 1,
                "transform": [
                    torchvision.transforms.Resize((28, 28)),
                    torchvision.transforms.Grayscale(num_output_channels=3),
                    *digits_transform
                ]
            },
            "SynthDigits": {
                "channel": 3,
                "transform": [
                    torchvision.transforms.Resize((28, 28)),
                    *digits_transform
                ]
            },
            "MNIST_M": {
                "channel": 3,
                "transform": digits_transform
            },
            ######################################## office-caltech10
            "amazon": {
                "channel": 3,
                "transform": office_caltech10_transform
            },
            "caltech": {
                "channel": 3,
                "transform": office_caltech10_transform
            },
            "dslr": {
                "channel": 3,
                "transform": office_caltech10_transform
            },
            "webcam": {
                "channel": 3,
                "transform": office_caltech10_transform
            },
            ############################### office-home
            "Art": {
                "channel": 3,
                "transform": office_home_transform
            },
            "Clipart": {
                "channel": 3,
                "transform": office_home_transform
            },
            "Product": {
                "channel": 3,
                "transform": office_home_transform
            },
            "RealWorld": {
                "channel": 3,
                "transform": office_home_transform
            },
            ########################################### domain-net
            "clipart": {
                "channel": 3,
                "transform": domain_net_transform
            },
            "real": {
                "channel": 3,
                "transform": domain_net_transform
            },
            "sketch": {
                "channel": 3,
                "transform": domain_net_transform
            },
            "quickdraw": {
                "channel": 3,
                "transform": domain_net_transform
            },
            "infograph": {
                "channel": 3,
                "transform": domain_net_transform
            },
            "painting": {
                "channel": 3,
                "transform": domain_net_transform
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
                self.__Dataset(
                    root_path=os.path.join(self.__root_path, self.__dataset_name),
                    domain=substrs[0],
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
                pin_memory = False,
                num_workers = 0
            ))
        return trainloaders

    def get_testloaders(self,
        names: List[str]
    ) -> List[torch.utils.data.DataLoader]:
        testloaders = []
        for name in names:
            substrs = name.split("-")
            testloaders.append(torch.utils.data.DataLoader(
                self.__Dataset(
                    root_path=os.path.join(self.__root_path, self.__dataset_name),
                    domain=substrs[0],
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
                pin_memory = False,
                num_workers = 0
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


if __name__ == "__main__":
    dm = DatasetManager("/home/sjinglong/data/datasets/personal/advsl/", "domain-net", 40, 32)
    # dm.get_trainloaders(["MNIST", "SVHN", "USPS", "SynthDigits", "MNIST_M"]) # 0.5
    # dm.get_trainloaders(["amazon", "caltech", "dslr", "webcam"]) # 15
    # dm.get_trainloaders(["Art", "Clipart", "Product", "RealWorld"]) # 30
    dm.get_trainloaders(["clipart", "real", "sketch", "quickdraw", "infograph", "painting"]) # 40
