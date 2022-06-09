
from typing import List
import torch

from model.gan import Discriminator

class ClusterUnion:
    def __init__(self,
        k: float,
        fm_size: torch.Size,
        lr: float,
        device: str
    ) -> None:
        self.__k = k
        self.__update_counter = 0
        self.__discriminator = Discriminator(fm_size.numel()).to(device)
        self.__loss_fn = torch.nn.BCELoss().to(device)
        self.__optim = torch.optim.SGD(self.__discriminator.parameters(), lr)
        self.__device = device

    def update(self,
        anchor_fm: torch.Tensor,
        float_fms: List[torch.Tensor]
    ) -> None:
        self.__update_counter += self.__k
        if self.__update_counter >= 1:
            anchor_label = torch.ones((anchor_fm.size(0), 1), device=self.__device)
            float_labels = [torch.zeros((float_fm.size(0), 1), device=self.__device) for float_fm in float_fms]
            for _ in range(int(self.__update_counter)):
                for float_fm, float_label in zip(float_fms, float_labels):
                    self.__optim.zero_grad()
                    float_loss = self.__loss_fn(self.__discriminator(float_fm.detach()), float_label)
                    anchor_loss = self.__loss_fn(self.__discriminator(anchor_fm.detach()), anchor_label)
                    loss = (float_loss + anchor_loss) / 2
                    loss.backward()
                    self.__optim.step()
            self.__update_counter = 0
        self.__optim.zero_grad()
        gene_losses = [self.__loss_fn(self.__discriminator(float_fm), torch.ones((float_fm.size(0), 1), device=self.__device)) for float_fm in float_fms]
        [gene_loss.backward(retain_graph=True) for gene_loss in gene_losses]

    def update_gr(self,
        anchor_fm: torch.Tensor,
        float_fms: List[torch.Tensor]
    ) -> None:
        self.__optim.zero_grad()
        anchor_label = torch.ones((anchor_fm.size(0), 1), device=self.__device)
        float_labels = [torch.zeros((float_fm.size(0), 1), device=self.__device) for float_fm in float_fms]
        anchor_loss = self.__loss_fn(self.__discriminator(anchor_fm), anchor_label)
        float_losses = [self.__loss_fn(self.__discriminator(float_fm), float_label) for float_fm, float_label in zip(float_fms, float_labels)]
        anchor_loss.backward(retain_graph=True)
        [float_loss.backward(retain_graph=True) for float_loss in float_losses]
        self.__optim.step()

    def update_gr_(self,
        anchor_fm: torch.Tensor,
        float_fms: List[torch.Tensor]
    ) -> None:
        self.__optim.zero_grad()
        anchor_label = torch.ones((anchor_fm.size(0), 1), device=self.__device)
        float_labels = [torch.zeros((float_fm.size(0), 1), device=self.__device) for float_fm in float_fms]
        anchor_loss = self.__loss_fn(self.__discriminator(anchor_fm.detach()), anchor_label)
        float_losses = [self.__loss_fn(self.__discriminator(float_fm), float_label) for float_fm, float_label in zip(float_fms, float_labels)]
        total_loss = (anchor_loss + sum(float_losses)) / (1 + len(float_losses))
        total_loss.backward(retain_graph=True)
        self.__optim.step()
