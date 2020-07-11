import torch
from .vocabulary import Vocab
from typing import Optional, Dict, List, Any


class Dataset(object):
    def skip(self, count: int):
        raise NotImplementedError()

    def fetch(self, batch: Optional[int] = None, device: Optional[str] = None
              ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()


class TokenizedCorpusDataset(Dataset):
    def __init__(self, vocab: Vocab, corpus_path: str, seq_len: int):
        self.vocab = vocab
        self.corpus_fp = open(corpus_path, 'r', encoding='utf-8')
        self.seq_len = seq_len

    def skip(self, count: int):
        if not self.corpus_fp.readline():
            self.corpus_fp.seek(0)
            self.corpus_fp.readline()

    def _fetch_one(self) -> Dict[str, List[int]]:
        while True:
            line = self.corpus_fp.readline()
            if not line:
                self.corpus_fp.seek(0)
                continue

            # Map each subword to its token index.
            indices = [self.vocab[t] for t in line.split()]
            if len(indices) > self.seq_len - 2:
                continue

            # Add special tokens to the sequence.
            indices = [self.vocab.bos_idx] + indices + [self.vocab.eos_idx]
            indices += ([self.vocab.pad_idx]
                        * (self.seq_len - len(indices) + 1))

            return {'input': indices[:-1], 'output': indices[1:]}

    def fetch(self, batch: Optional[int] = None, device: Optional[str] = None
              ) -> Dict[str, torch.Tensor]:
        if batch is None:
            data = self._fetch_one()
        else:
            data = [self._fetch_one() for _ in range(batch)]
            data = {k: [d[k] for d in data] for k in data[0]}

        # Cast sequences to tensors.
        return {k: torch.tensor(v, dtype=torch.long, device=device)
                for k, v in data.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.corpus_fp.seek(state_dict['offset'])

    def state_dict(self) -> Dict[str, Any]:
        return {'offset': self.corpus_fp.tell()}
