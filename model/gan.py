
import torch

class Discriminator(torch.nn.Module):
    def __init__(self, in_size) -> None:
        super().__init__()
        self.__layers = torch.nn.Sequential(
            torch.nn.Linear(in_size, 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512,256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.__layers(img_flat)
        return validity