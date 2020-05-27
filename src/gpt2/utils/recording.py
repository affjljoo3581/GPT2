import time
from typing import Dict, Any


class Recorder(object):
    """Record metrics for training and evaluation."""
    def __init__(self):
        self.metrics_train = [[]]
        self.metrics_eval = []
        self.stamps = [(0, time.time())]

    def add_train_metrics(self, **metrics: float):
        """Add metrics for training."""
        if not isinstance(self.metrics_train[-1], list):
            self.metrics_train.append([])
        self.metrics_train[-1].append(metrics)

    def add_eval_metrics(self, **metrics: float):
        """Add metrics for evaluation."""
        self.metrics_eval.append(metrics)

    def stamp(self, step: int):
        """Stamp current iterations with estimated time."""
        metrics = {name: [] for name in self.metrics_train[-1][0]}

        # Average last collected training metrics.
        for m in self.metrics_train[-1]:
            for k, v in m.items():
                metrics[k].append(v)
        for k, v in metrics.items():
            metrics[k] = sum(v) / len(v)

        # Change gathered metrics to the averaged metrics.
        self.metrics_train[-1] = metrics

        # Stamp current step.
        self.stamps.append((step, time.time()))

    def format(self, fstring: str) -> str:
        """Return string formatted with last recorded metrics."""
        train_params = {f'train_{k}': v
                        for k, v in self.metrics_train[-1].items()}
        eval_params = {f'eval_{k}': v
                       for k, v in self.metrics_eval[-1].items()}

        return fstring.format(**train_params, **eval_params)

    def summarize(self) -> Dict[str, Any]:
        """Summarize stamps and records."""
        steps, times = list(zip(*self.stamps))
        steps = list(steps[1:])
        times = [times[i + 1] - times[i] for i in range(len(times) - 1)]

        return {'steps': steps,
                'times': times,
                'train': self.metrics_train,
                'eval': self.metrics_eval}

    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary of this class."""
        return {'metrics_train': self.metrics_train,
                'metrics_eval': self.metrics_eval,
                'stamps': self.stamps}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore this class from the given state dictionary."""
        self.metrics_train = state_dict['metrics_train']
        self.metrics_eval = state_dict['metrics_eval']
        self.stamps = state_dict['stamps']
