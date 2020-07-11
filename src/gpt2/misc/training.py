import torch
import torch.nn as nn
import torch.optim as optim
from .objective import Objective
from .preserving import Preservable
from .recording import Recordable, records
from ..data.serving import Dataset
from typing import Optional


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
        self.iters = 0
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
        self.model.train()
        self.optimizer.zero_grad()

        data = self.train_dataset.fetch(batch, device='cuda')

        loss = self.train_objective(data['input'], data['output'])
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return {'loss': loss.item()}

    @records('eval')
    def evaluate(self, batch: Optional[int] = None):
        with torch.no_grad():
            self.model.eval()

            data = self.eval_dataset.fetch(batch, device='cuda')
            loss = self.eval_objective(data['input'], data['output'])

        return {'loss': loss.item()}
