import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel as parallel
from .recording import Recorder
from .objective import Objective
from ..data.serving import DataLoader
from typing import Dict, Any, Optional, List

# Try to import `apex` library for mixed-precision training. Note that the
# library should be installed when ``use_amp=True``.`
try:
    from apex import amp
except ModuleNotFoundError:
    pass


class Trainer(object):
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 eval_loader: DataLoader,
                 train_objective: Objective,
                 eval_objective: Objective,
                 optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler._LRScheduler,
                 recorder: Recorder,
                 gpus: Optional[List[int]] = None,
                 use_amp: bool = False):
        # Convert model and optimizer for mixed-precision training if
        # ``use_amp=True``.
        if use_amp:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', verbosity=0)

        # Set model to the objectives after converting.
        train_objective.set_model(model)
        eval_objective.set_model(model)

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model
        self.train_objective = train_objective
        self.eval_objective = eval_objective
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.recorder = recorder
        self.use_amp = use_amp

    def train(self, batch: Optional[int] = None):
        # Prepare training models.
        self.model.train()
        self.optimizer.zero_grad()

        # Fetch training data.
        data = self.train_loader.fetch(batch, device='cuda')

        # Calculate loss.
        if self.gpus is None:
            loss = self.train_objective(data['input'], data['output'])
        else:
            loss = parallel.data_parallel(
                self.train_objective,
                data['input'],
                device_ids=self.gpus,
                output_device=0,
                dim=0,
                module_kwargs={'outputs': data['output']})

        # Calculate gradients.
        if self.use_amp:
            # Calculate gradients through scaled loss rather than the original
            # loss to prevent underflowing in mixed precision.
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # Update variables and learning rate.
        self.optimizer.step()
        self.scheduler.step()

        # Record metrics for training.
        self.recorder.add_train_metrics(loss=loss.item())

    def evaluate(self, batch: Optional[int] = None):
        with torch.no_grad():
            self.model.eval()

            data = self.eval_loader.fetch(batch, device='cuda')

            if self.gpus is None:
                loss = self.eval_objective(data['input'], data['output'])
            else:
                loss = parallel.data_parallel(
                    self.eval_objective,
                    data['input'],
                    device_ids=self.gpus,
                    output_device=0,
                    dim=0,
                    module_kwargs={'outputs': data['output']})

        # Record metrics for evaluation.
        self.recorder.add_eval_metrics(loss=loss.item())

    def state_dict(self) -> Dict[str, Any]:
        # Collect all `state_dict`s from members.
        state_dict = {}
        for name, module in self.__dict__.items():
            if getattr(module, 'state_dict', None):
                state_dict[name] = module.state_dict()

        # Save automatic mixed-precision states as well.
        if self.use_amp:
            state_dict['amp'] = amp.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # Restore all `state_dict`s to members.
        for name, module in self.__dict__.items():
            if getattr(module, 'load_state_dict', None):
                module.load_state_dict(state_dict[name])

        # Restore automatic mixed-precision states as well.
        if self.use_amp:
            amp.load_state_dict(state_dict['amp'])
