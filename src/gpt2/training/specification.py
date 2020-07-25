import torch
import torch.nn as nn
import torch.optim as optim
from gpt2.data import Dataset
from typing import Tuple, Iterator, Dict


class TrainingSpec(object):
    def initialize(self):
        pass

    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError()

    def construct_model(self) -> nn.Module:
        raise NotImplementedError()

    def create_optimizer(self, params: Iterator[nn.Parameter]
                         ) -> Tuple[optim.Optimizer,
                                    optim.lr_scheduler._LRScheduler]:
        raise NotImplementedError()

    def train_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                        ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def eval_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                       ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()
