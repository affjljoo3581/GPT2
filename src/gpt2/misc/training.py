import torch
import torch.nn as nn
import torch.optim as optim
from .objective import Objective
from .preserving import Preservable
from .recording import Recordable, records
from ..data.serving import Dataset
from typing import Optional
import torch.cuda.nvtx as nvtx


class Trainer(Recordable, Preservable):
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler._LRScheduler,
                 train_dataset: Dataset,
                 eval_dataset: Dataset,
                 train_objective: Objective,
                 eval_objective: Objective):
        super().__init__()
        self.iters = -1
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_objective = train_objective
        self.eval_objective = eval_objective

    def save(self, checkpoint: str):
        torch.save({'metrics': self.metrics,
                    'model': self.model.cpu().state_dict()}, checkpoint)

    @records('train')
    def train(self, batch: Optional[int] = None):
        nvtx.range_push('initialize')
        self.model.train()
        self.optimizer.zero_grad()
        nvtx.range_pop()

        nvtx.range_push('dataset.fetch')
        data = self.train_dataset.fetch(batch, device='cuda')
        nvtx.range_pop()

        nvtx.range_push('optimize model')
        nvtx.range_push('objective.loss & loss.backward')
        loss = self.train_objective.loss(data['input'], data['output'])
        loss.backward()
        nvtx.range_pop()

        nvtx.range_push('optimizer.step')
        self.optimizer.step()
        self.scheduler.step()
        nvtx.range_pop()
        nvtx.range_pop()

        return {'loss': loss.item()}

    @records('eval')
    def evaluate(self, batch: Optional[int] = None):
        with torch.no_grad():
            self.model.eval()

            nvtx.range_push('dataset.fetch')
            data = self.eval_dataset.fetch(batch, device='cuda')
            nvtx.range_pop()
            nvtx.range_push('objective.loss')
            loss = self.eval_objective.loss(data['input'], data['output'])
            nvtx.range_pop()

        return {'loss': loss.item()}
