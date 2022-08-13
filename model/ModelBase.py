# coding=utf-8

from typing import Tuple
import torch

__all__ = ["ModelBase"]

class ModelBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
        x:torch.Tensor,
        start:int = 0,
        stop:int = None
    ) -> torch.Tensor:
        if stop == None:
            stop = self.block_num
        if start < 0 or start >= stop:
            raise ValueError(f"Start point `{start}` can not be negative or >= stop point `{stop}`.")
        if stop > self.block_num:
            raise ValueError(f"Stop point `{stop}` is out of scope `[1 - {self.block_num}]`")
        for block in self._blocks[start:stop]:
            x = block(x)
        return x

    def get_partial_params(self,
        start:int = 0,
        stop:int = None
    ) -> Tuple[torch.Tensor]:
        if stop == None:
            stop = self.block_num
        if start < 0 or start >= stop:
            raise ValueError(f"Start point `{start}` can not be negative or >= stop point `{stop}`.")
        if stop > self.block_num:
            raise ValueError(f"Stop point `{stop}` is out of scope `[1 - {self.block_num}]`")
        return (param for param in self._blocks[start:stop].parameters())

    def get_splited_module(self,
        cut_point:int
    ) -> Tuple[torch.nn.Module]:
        if cut_point < 0 or cut_point > self.block_num:
            raise ValueError(f"The given cut_point `{cut_point}` is out of scope `[0 - {self.block_num}]`.")
        return (torch.nn.Sequential(*self._blocks[:cut_point]), torch.nn.Sequential(*self._blocks[cut_point:]))
