from typing import Dict, Any


class Recorder(object):
    def __init__(self):
        self.metrics_train = [[]]
        self.metrics_eval = []
        self.steps = []

    def add_train_metrics(self, **metrics: float):
        if not isinstance(self.metrics_train[-1], list):
            self.metrics_train.append([])
        self.metrics_train[-1].append(metrics)

    def add_eval_metrics(self, **metrics: float):
        self.metrics_eval.append(metrics)

    def stamp(self, step: int):
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
        self.steps.append(step)

    def format(self, fstring: str) -> str:
        train_params = {f'train_{k}': v
                        for k, v in self.metrics_train[-1].items()}
        eval_params = {f'eval_{k}': v
                       for k, v in self.metrics_eval[-1].items()}

        return fstring.format(**train_params, **eval_params)

    def summarize(self) -> Dict[str, Any]:
        return {'steps': self.steps,
                'train': self.metrics_train,
                'eval': self.metrics_eval}

    def state_dict(self) -> Dict[str, Any]:
        return {'metrics_train': self.metrics_train,
                'metrics_eval': self.metrics_eval,
                'steps': self.steps}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.metrics_train = state_dict['metrics_train']
        self.metrics_eval = state_dict['metrics_eval']
        self.steps = state_dict['steps']
