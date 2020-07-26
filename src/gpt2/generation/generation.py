import torch
import torch.nn as nn
from gpt2.data import Vocab
from gpt2.modeling import Past
from gpt2.generation import Tokenizer
from typing import Tuple, List, Optional


class Generator(object):
    def __init__(self,
                 vocab: Vocab,
                 tokenizer: Tokenizer,
                 model: nn.Module,
                 seq_len: int,
                 top_p: float = 0.85,
                 use_gpu: bool = False):
        if use_gpu:
            model.cuda().half()

        self.vocab = vocab
        self.tokenizer = tokenizer
        self.model = model
        self.seq_len = seq_len
        self.top_p = top_p
        self.top_p = top_p
        self.use_gpu = use_gpu

    def _sample_from_top_p(self, probs: torch.Tensor) -> int:
        probs, indices = probs.sort(descending=True)

        mask = probs.cumsum(-1) > self.top_p
        mask[0] = False
        probs.masked_fill_(mask, 0)

        # Sample from filtered distribution.
        return indices[probs.multinomial(1)[0]]

    @torch.no_grad()
    def _predict_probs(self,
                       words: List[int],
                       past: Optional[List[Past]] = None
                       ) -> Tuple[torch.Tensor, List[Past]]:
        x = torch.tensor(words,
                         dtype=torch.long,
                         device='cuda' if self.use_gpu else 'cpu')
        logits, past = self.model(x, past)

        # If tokens are predicted on GPU, copy the logits tensor to CPU.
        if self.use_gpu:
            logits = logits.cpu().float()

        return logits[-1, :].softmax(-1), past

    def generate(self, context: str) -> str:
        words = [self.vocab[t] for t in self.tokenizer.encode(context)]
        words = [self.vocab.bos_idx] + words

        current, past = words, None
        while len(words) < self.seq_len:
            probs, past = self._predict_probs(current, past)
            next_word = self._sample_from_top_p(probs)

            words.append(next_word)
            current = [next_word]

            # If end-of-sentence token is sampled, then terminate generating
            # sentence.
            if next_word == self.vocab.eos_idx:
                break

        return self.tokenizer.decode([self.vocab[w] for w in words])
