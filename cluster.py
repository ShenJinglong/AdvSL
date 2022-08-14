
from typing import List
import torch
import logging

from model.gan import Discriminator
from utils.model_utils import ratio_model_grad

class ClusterUnion:
    def __init__(self,
        k: float,
        fm_size: torch.Size,
        lr: float,
        device: str
    ) -> None:
        self.__k = k
        self.__update_counter = 0
        self.__discriminator = Discriminator(fm_size.numel(), 1).to(device)
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

class ClusterUnionMultiout:
    def __init__(self,
        k: float,
        fm_size: torch.Size,
        client_num: int,
        lr: float,
        device: str
    ) -> None:
        self.__k = k
        self.__update_counter = 0
        self.__discriminator = Discriminator(fm_size.numel(), client_num).to(device)
        self.__loss_fn = torch.nn.CrossEntropyLoss().to(device)
        self.__bce_loss_fn = torch.nn.BCELoss().to(device)
        self.__optim = torch.optim.SGD(self.__discriminator.parameters(), lr)
        self.__device = device

    def update_gr(self,
        fms: List[torch.Tensor]
    ) -> None:
        self.__optim.zero_grad()
        # labels = [torch.zeros((fm.size(0), len(fms))).scatter_(
        #     1, torch.ones((fm.size(0),1))*i, torch.ones((fm.size(0),1))
        # ) for i, fm in enumerate(fms)] # one-hot coding
        labels = [torch.ones((fm.size(0),), dtype=torch.int64, device=self.__device)*i for i, fm in enumerate(fms)]
        losses = [self.__loss_fn(self.__discriminator(fm), label) for fm, label in zip(fms, labels)]
        [loss.backward(retain_graph=True) for loss in losses]
        ratio_model_grad(self.__discriminator, 1./len(fms))
        self.__optim.step()

    def update(self,
        fms: List[torch.Tensor]
    ) -> None:
        self.__optim.zero_grad()
        gene_losses = [self.__bce_loss_fn(self.__discriminator(fm), torch.ones((fm.size(0), len(fms)), device=self.__device)/len(fms)) for fm in fms]
        [gene_loss.backward(retain_graph=True) for gene_loss in gene_losses]
        logging.info("generator updated ...")
        
        self.__update_counter += self.__k
        if self.__update_counter >= 1:
            labels = [torch.ones((fm.size(0),), dtype=torch.int64, device=self.__device)*i for i, fm in enumerate(fms)]
            for _ in range(int(self.__update_counter)):
                self.__optim.zero_grad()
                for fm, label in zip(fms, labels):
                    loss = self.__loss_fn(self.__discriminator(fm.detach()), label)
                    loss.backward()
                ratio_model_grad(self.__discriminator, 1./len(fms))
                self.__optim.step()
                logging.info("discriminator updated ...")
            self.__update_counter = 0

if __name__ == "__main__":
    dummy_union = ClusterUnion(1, torch.Size((8,)), 5, 0.25, "cpu")
    dummy_inputs = [torch.rand((2,8)) for _ in range(5)]
    dummy_union.update_multiout(dummy_inputs)