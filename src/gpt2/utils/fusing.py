try:
    from apex.optimizers import FusedAdam as Adam
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.optim import AdamW as Adam
    from torch.nn import LayerNorm
