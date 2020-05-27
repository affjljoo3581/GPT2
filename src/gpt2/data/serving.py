import torch
from typing import Optional, Union, List


class DataLoader(object):
    """Simple data loader from file.

    DataLoader loads sequences by reading file and encodes them through the
    given vocabulary. Special tokens are added to each sequence.

    Arguments:
        vocab (str): Vocabulary file path.
        corpus (str): Corpus file path.
        seq_len (int): The maximum length of each sequence.
        bos_token (str): Begin-of-sentence token name.
        eos_token (str): End-of-sentence token name.
        pad_token (str): Pad token name.
    """
    def __init__(self,
                 vocab: str,
                 corpus: str,
                 seq_len: int,
                 bos_token: str = '<s>',
                 eos_token: str = '</s>',
                 pad_token: str = '<pad>'):
        self.corpus_fp = open(corpus, 'r', encoding='utf-8')
        self.seq_len = seq_len
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        # Create vocabulary dictionary which maps from subwords to indices.
        with open(vocab, 'r', encoding='utf-8') as fp:
            self.vocab = {word: i for i, word in enumerate(fp.read().split())}

    def close(self):
        """Close resources."""
        self.corpus_fp.close()

    def _fetch_one(self) -> torch.Tensor:
        while True:
            # Get sequence by reading file.
            line = self.corpus_fp.readline()

            # If current position is end of file, move to first and read again.
            if not line:
                self.corpus_fp.seek(0)
                continue

            # Map each subword to its index.
            indices = [self.vocab[t] for t in line.split()]

            # Skip if the sequence is too long.
            if len(indices) > self.seq_len - 2:
                continue

            # Add speical tokens.
            indices = ([self.vocab[self.bos_token]]
                       + indices
                       + [self.vocab[self.eos_token]])
            indices = indices + ([self.vocab[self.pad_token]]
                                 * (self.seq_len - len(indices) + 1))

            return {'input': indices[:-1], 'output': indices[1:]}

    def fetch(self,
              batch: Optional[int] = None
              ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Fetch sequences from the corpus.

        Arguments:
            batch (int): The number of sequences in batch.

        Returns:
            A tensor of shape `(seq_len)` is ``batch=None`` else a list of
            tensor of shape `(batch, seq_len)`.
        """
        if batch is None:
            data = self._fetch_one()
        else:
            data = {}
            for _ in range(batch):
                for k, v in self._fetch_one().items():
                    if k not in data:
                        data[k] = []
                    data[k].append(v)

        # Cast each sequence to tensor.
        return {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}

    def seek(self, offset: int):
        """Set current position of the corpus file at the given offset.
        """
        self.corpus_fp.seek(offset)

    def tell(self) -> int:
        """Return the current position of the corpus file."""
        return self.corpus_fp.tell()
