import torch
from typing import Optional


class Preservable(object):
    def preserve(self, checkpoint: str):
        def _get_state(obj: object):
            # If object has `state_dict` method, use it rather than dump the
            # value directly.
            return (obj.state_dict()
                    if getattr(obj, 'state_dict', None)
                    else obj)

        torch.save({k: _get_state(v) for k, v in self.__dict__.items()},
                   checkpoint)

    def restore(self, checkpoint: str, map_location: Optional[str] = None):
        ckpt = torch.load(checkpoint, map_location=map_location)
        for k, v in ckpt.items():
            if getattr(self[k], 'load_state_dict', None):
                # If object has `load_state_dict` method, use it rather than
                # assign the value directly.
                self[k].load_state_dict(v)
            else:
                self[k] = v

        # Remove used checkpoint object and clear cuda cache to prevent out of
        # memory error.
        del ckpt
        torch.cuda.empty_cache()
