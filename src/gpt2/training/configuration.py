
class TrainConfig(object):
    def __init__(self,
                 batch_train: int,
                 batch_eval: int,
                 total_steps: int,
                 eval_steps: int,
                 save_steps: int,
                 save_model_path: str,
                 save_checkpoint_path: str,
                 description: str,
                 log_format: str,
                 use_amp: bool,
                 gpus: int):
        self.batch_train = batch_train
        self.batch_eval = batch_eval
        self.total_steps = total_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.save_model_path = save_model_path
        self.save_checkpoint_path = save_checkpoint_path
        self.description = description
        self.log_format = log_format
        self.use_amp = use_amp
        self.gpus = gpus

    @property
    def distributed(self) -> bool:
        return self.gpus is not None and self.gpus > 1
