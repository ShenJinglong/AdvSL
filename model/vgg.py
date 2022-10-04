# coding=utf-8

import torch
import torchvision
from model.ModelBase import ModelBase

class VGG16_Mnist(ModelBase):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self._blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=3, out_channels=64, padding=5, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True)
            ), # 1
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            ), # 2
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True)
            ), # 3
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            ), # 4
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(inplace=True)
            ), # 5
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(inplace=True)
            ), # 6
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            ), # 7
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(inplace=True)
            ), # 8
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(inplace=True)
            ), # 9
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            ), # 10
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(inplace=True)
            ), # 11
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(inplace=True)
            ), # 12
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3, stride=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)
            ), # 13
            torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((7,7)),
                torch.nn.Flatten(1),
                torch.nn.Linear(in_features=512*7*7, out_features=4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout()
            ), # 14
            torch.nn.Sequential(
                torch.nn.Linear(in_features=4096, out_features=4096),
                torch.nn.ReLU(True),
                torch.nn.Dropout()
            ), # 15
            torch.nn.Sequential(
                torch.nn.Linear(in_features=4096, out_features=num_classes)
            )  # 16
        ])
        self.block_num = len(self._blocks)
        self.__initialize_weights()

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

class VGG16(torchvision.models.vgg.VGG, ModelBase):
    def __init__(self, num_classes: int = 1000, init_weights: bool = True) -> None:
        super().__init__(
            torchvision.models.vgg.make_layers(
                [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
                batch_norm=True
            ),
            num_classes,
            init_weights
        )
        modules = dict(self.named_modules())
        self._blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                modules['features.0'],
                modules['features.1'],
                modules['features.2'],
            ),
            torch.nn.Sequential(
                modules['features.3'],
                modules['features.4'],
                modules['features.5'],
                modules['features.6'],
            ),
            torch.nn.Sequential(
                modules['features.7'],
                modules['features.8'],
                modules['features.9'],
            ),
            torch.nn.Sequential(
                modules['features.10'],
                modules['features.11'],
                modules['features.12'],
                modules['features.13'],
            ),
            torch.nn.Sequential(
                modules['features.14'],
                modules['features.15'],
                modules['features.16'],
            ),
            torch.nn.Sequential(
                modules['features.17'],
                modules['features.18'],
                modules['features.19'],
            ),
            torch.nn.Sequential(
                modules['features.20'],
                modules['features.21'],
                modules['features.22'],
                modules['features.23'],
            ),
            torch.nn.Sequential(
                modules['features.24'],
                modules['features.25'],
                modules['features.26'],
            ),
            torch.nn.Sequential(
                modules['features.27'],
                modules['features.28'],
                modules['features.29'],
            ),
            torch.nn.Sequential(
                modules['features.30'],
                modules['features.31'],
                modules['features.32'],
                modules['features.33'],
            ),
            torch.nn.Sequential(
                modules['features.34'],
                modules['features.35'],
                modules['features.36'],
            ),
            torch.nn.Sequential(
                modules['features.37'],
                modules['features.38'],
                modules['features.39'],
            ),
            torch.nn.Sequential(
                modules['features.40'],
                modules['features.41'],
                modules['features.42'],
                modules['features.43'],
            ),
            torch.nn.Sequential(
                modules['avgpool'],
                torch.nn.Flatten(),
                modules['classifier.0'],
                modules['classifier.1'],
                modules['classifier.2'],
            ),
            torch.nn.Sequential(
                modules['classifier.3'],
                modules['classifier.4'],
                modules['classifier.5'],
            ),
            torch.nn.Sequential(
                modules['classifier.6'],
            )
        ])
        self.block_num = len(self._blocks)
    
    def forward(self,
        x:torch.Tensor,
        start:int = 0,
        stop:int = None
    ) -> torch.Tensor:
        return ModelBase.forward(self, x, start, stop)

if __name__ == "__main__":
    model = VGG16_Mnist()
    inputs = torch.randn((64, 3, 28, 28))
    outputs = model(inputs)