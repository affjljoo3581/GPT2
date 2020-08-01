
class EvaluateConfig(object):
    def __init__(self,
                 batch_eval: int,
                 use_gpu: bool):
        self.batch_eval = batch_eval
        self.use_gpu = use_gpu
