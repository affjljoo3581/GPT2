import torch
from .vocabulary import Vocabulary
from typing import Optional, Union, List, Dict


class DataLoader(object):
    def __init__(self, vocab: Vocabulary, corpus: str, seq_len: int):
        self.vocab = vocab
        self.corpus_fp = open(corpus, 'r', encoding='utf-8')
        self.seq_len = seq_len

    def close(self):
        self.corpus_fp.close()

    def _fetch_one(self) -> Dict[str, torch.Tensor]:
        line = self.corpus_fp.readline()
        if not line:
            self.corpus_fp.seek(0)
            return self._fetch_one()

        # Map each subword to its index.
        indices = [self.vocab[t] for t in line.split()]
        if len(indices) > self.seq_len - 2:
            continue

        # Add speical tokens to the sequence.
        indices = [self.vocab.bos_idx] + indices + [self.vocab.eos_idx]
        indices += ([self.vocab.pad_idx]
                    * (self.seq_len - len(indices) + 1))

        return {'input': indices[:-1], 'output': indices[1:]}

    def fetch(self,
              batch: Optional[int] = None
              ) -> Union[Dict[str, torch.Tensor],
                         List[Dict[str, torch.Tensor]]]:
        if batch is None:
            data = self._fetch_one()
        else:
            data = [self._fetch_one() for _ in range(batch)]
            data = {k: [d[k] for d in data] for k in data[0]}

        # Cast each sequence to tensor.
        return {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}

    def seek(self, offset: int):
        self.corpus_fp.seek(offset)

    def tell(self) -> int:
        return self.corpus_fp.tell()
