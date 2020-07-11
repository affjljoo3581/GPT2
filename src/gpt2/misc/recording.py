import functools
from typing import Optional


class Recordable(object):
    def __init__(self):
        self.metrics = {}
        self.batch_metrics = {}

    def record(self,
               scope: Optional[str] = None,
               **metrics: float):
        for name, value in metrics.items():
            # Add scope prefix to the metrics name.
            name = f'{scope}/{name}'

            # Add metrics to the batch.
            if name not in self.batch_metrics:
                self.batch_metrics[name] = []
            self.batch_metrics[name].append(value)

    def stamp(self, step: int = 0):
        for name, values in self.batch_metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []

            # Add averaged batch metrics.
            self.metrics[name].append((step, sum(values) / len(values)))

        # After update batch metrics, clear the batch.
        self.batch_metrics.clear()

    def format(self, fstring: str) -> str:
        return fstring.format(**{
            k.replace('/', '_'): v[-1][1] for k, v in self.metrics.items()})


def records(scope: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            print(self, args, kwargs)
            result = func(self, *args, **kwargs)

            # Record the output metrics with the given scope prefix.
            if isinstance(self, Recordable):
                self.record(scope=scope, **result)

        return wrapper
    return decorator
