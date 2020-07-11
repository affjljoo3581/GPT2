import tqdm
from .recording import Recordable
from typing import Optional, Iterator


class ProgressBar(object):
    def __init__(self,
                 start: int,
                 end: int,
                 desc: Optional[str] = None,
                 observe: Optional[Recordable] = None,
                 fstring: Optional[str] = None):
        self.tqdm_iter = tqdm.trange(end, desc=desc)

        if observe and fstring:
            def _modified_stamp(step: int = 0):
                _old_stamp(step)
                self.tqdm_iter.set_postfix_str(observe.format(fstring))

            # Patch `observe.stamp` to print last recorded metrics.
            _old_stamp = observe.stamp
            observe.stamp = _modified_stamp

    def __iter__(self) -> Iterator[int]:
        return iter(self.tqdm_iter)
