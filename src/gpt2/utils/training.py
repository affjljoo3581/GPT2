import torch
import torch.nn as nn
import torch.optim as optim
from gpt2.data.serving import DataLoader
from .recording import Recorder
from typing import Optional, Callable, Dict, Any

# Try to import `apex` library for mixed-precision training. Note that the
# library should be installed when ``use_amp=True``.`
try:
    from apex import amp
except ModuleNotFoundError:
    pass


class Trainer(object):
    """Simple integrated model trainer.

    Arguments:
        train_loader (DataLoader): Data loader for training.
        eval_loader (DataLoader): Data loader for evaluation.
        model: The model based on ``torch.nn.Module``.
        optimizer: The optimizer based on ``torch.optim.Optimizer``.
        scheduler: Learning rate scheduler.
        criterion: Objective loss function.
        recorder (Recorder): Metrics recorder.
        use_amp (bool): The boolean determining whether to use automatic mixed
            precision.
    """
    def __init__(self,
                 train_loader: DataLoader,
                 eval_loader: DataLoader,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: optim.lr_scheduler._LRScheduler,
                 criterion: Callable[..., torch.Tensor],
                 recorder: Recorder,
                 use_amp: bool = False):
        # Convert model and optimizer for mixed-precision training if
        # ``use_amp=True``.
        if use_amp:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O1', verbosity=0)

        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.recorder = recorder
        self.use_amp = use_amp

    def _backward(self, loss: torch.Tensor):
        if self.use_amp:
            # Calculate gradients through scaled loss rather than the original
            # loss to prevent underflowing in mixed precision.
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def train(self, batch: Optional[int] = None):
        """Train model and record metrics for training."""
        # Prepare training.
        self.model.train()
        self.optimizer.zero_grad()

        # Fetch training data.
        data = self.train_loader.fetch(batch)

        # Predict next words and calculate loss.
        preds, _ = self.model(data['input'].cuda())
        loss = self.criterion(preds.transpose(1, 2), data['output'].cuda())

        # Calculate gradients.
        # loss.backward()
        self._backward(loss)

        # Update variables and learning rate.
        self.optimizer.step()
        self.scheduler.step()

        # Record metrics for training.
        self.recorder.add_train_metrics(loss=loss.item())

    def evaluate(self, batch: Optional[int] = None):
        """Evaluate model and record its metrics."""
        with torch.no_grad():
            # Prepare evaluation.
            self.model.eval()

            # Fetch evaluation data.
            data = self.eval_loader.fetch(batch)

            # Predict next words and calculate loss.
            preds, _ = self.model(data['input'].cuda())
            loss = self.criterion(preds.transpose(1, 2), data['output'].cuda())

        # Record metrics for evaluation.
        self.recorder.add_eval_metrics(loss=loss.item())

    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary of this class."""
        state = {'train_loader': self.train_loader.tell(),
                 'eval_loader': self.eval_loader.tell(),
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict(),
                 'recorder': self.recorder.state_dict()}

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore this class from the given state dictionary."""
        self.train_loader.seek(state_dict['train_loader'])
        self.eval_loader.seek(state_dict['eval_loader'])
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.recorder.load_state_dict(state_dict['recorder'])
