import torch
from .vocabulary import Vocabulary
from typing import Optional, Union, List, Dict, Any


class DataLoader(object):
    def __init__(self,
                 vocab: Vocabulary,
                 corpus: str,
                 seq_len: int,
                 squeeze: bool = False):
        self.vocab = vocab
        self.corpus_fp = open(corpus, 'r', encoding='utf-8')
        self.seq_len = seq_len
        self.squeeze = squeeze

    def close(self):
        self.corpus_fp.close()

    def _fetch_one(self) -> Dict[str, torch.Tensor]:
        while True:
            line = self.corpus_fp.readline()
            if not line:
                self.corpus_fp.seek(0)
                continue

            # Map each subword to its index.
            indices = [self.vocab[t] for t in line.split()]
            if len(indices) > self.seq_len - 2:
                continue

            # Add speical tokens to the sequence.
            indices = [self.vocab.bos_idx] + indices + [self.vocab.eos_idx]
            indices += ([self.vocab.pad_idx]
                        * (self.seq_len - len(indices) + 1))

            return {'input': indices[:-1], 'output': indices[1:]}

    def _squeeze(self, sequences: List[List[int]]) -> List[List[int]]:
        max_len = max(s.index(self.pad_idx) for s in sequences)
        return [s[:max_len] for s in sequences]

    def fetch(self,
              batch: Optional[int] = None,
              device: Optional[Union[str, torch.device]] = None
              ) -> Union[Dict[str, torch.Tensor],
                         List[Dict[str, torch.Tensor]]]:
        if batch is None:
            data = self._fetch_one()
        else:
            data = [self._fetch_one() for _ in range(batch)]
            data = {k: [d[k] for d in data] for k in data[0]}

            if self.squeeze:
                # Squeeze the sequences to reduce lengths.
                data = {k: self._squeeze(v) for k, v in data.items()}

        # Cast each sequence to tensor.
        return {k: torch.tensor(v, dtype=torch.long, device=device)
                for k, v in data.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.corpus_fp.seek(state_dict['offset'])

    def state_dict(self) -> Dict[str, Any]:
        return {'offset': self.corpus_fp.tell()}
