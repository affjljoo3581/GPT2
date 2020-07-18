import torch
import torch.nn as nn
from ..data.vocabulary import Vocab
from ..data.tokenization import Tokenizer
from ..modeling.attention import Past
from typing import Tuple, List, Optional


class Generator(object):
    def __init__(self,
                 vocab: Vocab,
                 tokenizer: Tokenizer,
                 model: nn.Module,
                 seq_len: int,
                 top_p: float = 0.92,
                 use_gpu: bool = False):
        if use_gpu:
            model.cuda().half()

        self.vocab = vocab
        self.tokenizer = tokenizer
        self.model = model
        self.seq_len = seq_len
        self.top_p = top_p
        self.use_gpu = use_gpu

    def _sample_from_top_p(self, logits: torch.Tensor
                           ) -> Tuple[List[int], List[float]]:
        # Sort the logits and use only top-p tokens.
        sorted_logits, indices = logits.sort(descending=True)
        sorted_logits.masked_fill_(
            sorted_logits.softmax(-1).cumsum(-1) > self.top_p,
            value=-1e10)

        # Use gumbel-max trick to sample from logits.
        gumbels = -torch.empty_like(sorted_logits).exponential_().log()
        return indices[(sorted_logits + gumbels).argmax()]

    @torch.no_grad()
    def _predict_logits(self,
                        words: List[int],
                        past: Optional[List[Past]] = None
                        ) -> Tuple[torch.Tensor, List[Past]]:
        x = torch.tensor(words,
                         dtype=torch.long,
                         device='cuda' if self.use_gpu else 'cpu')
        logits, past = self.model(x, past)

        # If tokens are predicted on GPU, move the calculated logits to CPU.
        if self.use_gpu:
            logits = logits.cpu().float()

        return logits[-1, :], past

    def generate(self, context: str) -> str:
        words = [self.vocab[t] for t in self.tokenizer.encode(context)]
        words = [self.vocab.bos_idx] + words

        current, past = words, None
        while len(words) < self.seq_len:
            # Predict next-word distribution and sample from it.
            logits, past = self._predict_logits(current, past)
            next_word = self._sample_from_top_p(logits)

            # Add sampled word to the sequence and change the current
            # subsequence to the sampled word.
            words.append(next_word)
            current = [next_word]

            # If end-of-sentence token is sampled, then terminate generating
            # sentence.
            if next_word == self.vocab.eos_idx:
                break

        return self.tokenizer.decode([self.vocab[w] for w in words])
