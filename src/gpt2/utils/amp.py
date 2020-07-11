import torch
import torch.optim as optim
from ..misc.training import Trainer
from ..misc.objective import Objective

try:
    from apex import amp
except ModuleNotFoundError:
    pass


def _modify_objective(objective: Objective, optimizer: optim.Optimizer):
    def _modified_objective_loss(*args, **kwargs):
        loss = _old_objective_loss(*args, **kwargs)

        # Patch `loss.backward` to perform loss scaling.
        def _modified_tensor_backward():
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                torch.autograd.backward(scaled_loss)
        loss.backward = _modified_tensor_backward

        return loss

    # Modify `objective.loss` to return patched loss tensor.
    _old_objective_loss = objective.loss
    objective.loss = _modified_objective_loss


def apply(trainer: Trainer):
    # Initialize model and optimizer.
    trainer.model, trainer.optimizer = amp.initialize(
        trainer.model, trainer.optimizer, opt_level='O2', verbosity=0)

    # Patch objectives.
    trainer.train_objective.model = trainer.model
    trainer.eval_objective.model = trainer.model

    _modify_objective(trainer.train_objective, trainer.optimizer)

    # Add `amp` key to the trainer object to make the state of apex.amp
    # preservable.
    trainer.amp = amp
