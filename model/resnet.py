
from typing import Tuple
import torch

class BasicBlock(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ) -> None:
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self,
        x: torch.Tensor
    ) -> torch.Tensor:
        y = torch.relu(self.bn1(self.conv1(x)))
        
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = torch.relu(y)
        return y

class ResNet18_Mnist(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.__blocks = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU()
            ),                              # 1
            BasicBlock(64, 64, 1),          # 2
            BasicBlock(64, 64, 1),          # 3
            BasicBlock(64, 128, 2),         # 4
            BasicBlock(128, 128, 1),        # 5
            BasicBlock(128, 256, 2),        # 6
            BasicBlock(256, 256, 1),        # 7
            BasicBlock(256, 512, 2),        # 8
            BasicBlock(512, 512, 1),        # 9
            torch.nn.Sequential(
                torch.nn.AvgPool2d(kernel_size=4, stride=1),
                torch.nn.Flatten(),
                torch.nn.Linear(512, 10)
            )                               # 10
        ])
        self.block_num = len(self.__blocks)
        self.__initialize_weights()

    def forward(self,
        x: torch.Tensor,
        start: int = 0,
        stop: int = 10
    ):
        for block in self.__blocks[start:stop]:
            x = block(x)
        return x
    
    def get_splited_module(self,
        cut_point:int
    ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        if cut_point < 0 or cut_point > self.block_num:
            raise ValueError(f"Cut point {cut_point} out of module scope [0 - {self.block_num}].")
        return (torch.nn.Sequential(*self.__blocks[:cut_point]), torch.nn.Sequential(*self.__blocks[cut_point:]))

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1.0)
                torch.nn.init.constant_(m.bias, 0.0)

if __name__ == "__main__":
    model = ResNet18_Mnist()
    shallow, deep = model.get_splited_module(2)
    data = torch.randn((1, 3, 28, 28))
    output = model(data)
    print(output)
    y = shallow(data)
    print(y.size())
    y = deep(y)
    print(y)

    print(shallow)
