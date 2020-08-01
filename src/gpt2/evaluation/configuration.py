from typing import Iterator


class EvaluateConfig(object):
    def __init__(self,
                 batch_eval: int,
                 total_steps: int,
                 use_gpu: bool):
        self.batch_eval = batch_eval
        self.total_steps = total_steps
        self.use_gpu = use_gpu

    def iterate(self) -> Iterator:
        if self.total_steps == -1:
            while True:
                yield
        else:
            yield from range(self.total_steps)
