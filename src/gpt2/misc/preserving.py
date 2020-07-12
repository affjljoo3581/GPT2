import torch
from typing import Optional


class Preservable(object):
    def preserve(self, checkpoint: str):
        ckpt = {}
        for k, v in self.__dict__.items():
            if getattr(v, 'state_dict', None):
                # If object has `state_dict` method, use it rather than dump
                # the value directly.
                ckpt[k] = v.state_dict()
            elif not callable(v):
                ckpt[k] = v

        torch.save(ckpt, checkpoint)

    def restore(self, checkpoint: str, map_location: Optional[str] = None):
        ckpt = torch.load(checkpoint, map_location=map_location)
        for k, v in ckpt.items():
            if getattr(getattr(self, k), 'load_state_dict', None):
                # If object has `load_state_dict` method, use it rather than
                # assign the value directly.
                getattr(self, k).load_state_dict(v)
            else:
                setattr(self, k, v)

        # Remove used checkpoint object and clear cuda cache to prevent out of
        # memory error.
        del ckpt
        torch.cuda.empty_cache()
