import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from ..data.serving import Dataset
from ..misc import progress
from ..misc.training import Trainer
from typing import Optional, List


def initialize(idx: int, gpus: List[int]):
    # Store current distributing states.
    global _current_idx
    global _gpu_devices

    _current_idx = idx
    _gpu_devices = gpus

    # Initialize distributed process environment.
    torch.cuda.set_device(idx)
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:8000',
                            world_size=len(gpus),
                            rank=idx)

    # Disable progress bar if current process is not a master.
    if idx != 0:
        progress.ProgressBar = lambda start, end, *_: range(start, end)


def _modify_dataset(dataset: Dataset, idx: int, gpus: List[int]):
    def _modified_dataset_fetch(batch: Optional[int] = None,
                                device: Optional[str] = None):
        if batch is None or batch % len(gpus) != 0:
            raise ValueError('batch size must be a multiple of total gpu '
                             'count.')

        batch = batch // len(gpus)

        # Skip sequences which is for other gpus and take the corresponding
        # ones.
        dataset.skip(idx * batch)
        data = _old_dataset_fetch(batch, device)
        dataset.skip((len(gpus) - idx - 1) * batch)

        return data

    # Modify `dataset.fetch` to skip unnecessary batches.
    _old_dataset_fetch = dataset.fetch
    dataset.fetch = _modified_dataset_fetch


def apply(trainer: Trainer):
    # Use previously stored states.
    global _current_idx
    global _gpu_devices

    # Convert to the distributed components.
    trainer.model = DistributedDataParallel(
        trainer.model, device_ids=[_gpu_devices[_current_idx]])

    trainer.train_objective.model = trainer.model
    trainer.eval_objective.model = trainer.model

    _modify_dataset(trainer.train_dataset, _current_idx, _gpu_devices)
    _modify_dataset(trainer.eval_dataset, _current_idx, _gpu_devices)

    # Modify `trainer.restore` to load training states to the corresponding
    # device memory.
    trainer.restore = lambda ckpt, _ = None, _old_restore = trainer.restore: \
        _old_restore(ckpt, map_location=f'cuda:{_gpu_devices[_current_idx]}')

    # Prevent saving the training states except for the master gpu process.
    if _current_idx != 0:
        trainer.save = lambda *args, **kwargs: None
        trainer.preserve = lambda *args, **kwargs: None

    # Patch to save `trainer.model.module` rather than `trainer.model` because
    # parameters in the model are wrapped with `DistributedDataParallel`
    # module.
    if _current_idx == 0:
        def _modified_trainer_save(checkpoint: str):
            # Replace to the original model.
            container = trainer.model
            trainer.model = trainer.model.module

            # Save the model's parameters.
            _old_trainer_save(checkpoint)

            # After saving parameters, restore to the `DistributedDataParallel`
            # module.
            trainer.model = container

        _old_trainer_save = trainer.save
        trainer.save = _modified_trainer_save
