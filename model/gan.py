
import torch

class Discriminator(torch.nn.Module):
    def __init__(self,
        in_size: int,
        out_size: int
    ) -> None:
        super().__init__()
        self.__layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_size, 512),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(512,256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(256, out_size),
            torch.nn.Sigmoid(),
        )

    def forward(self,
        img: torch.Tensor
    ) -> torch.Tensor:
        validity = self.__layers(img)
        return validity

if __name__ == "__main__":
    sample = torch.ones((64, 64, 28, 28), device="cuda:0")
    model = Discriminator(64*28*28).to("cuda:0")
    out = model(sample)
    loss_fn = torch.nn.BCELoss().to("cuda:0")
    loss = loss_fn(out, torch.ones((64, 1), device="cuda:0"))
    loss.backward()
    print(next(model.parameters()).grad)

