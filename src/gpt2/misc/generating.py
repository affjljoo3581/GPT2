import torch
import torch.nn as nn
import numpy as np
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
                 temperature: float = 0.8,
                 use_gpu: bool = False):
        if use_gpu:
            model.cuda().half()

        self.vocab = vocab
        self.tokenizer = tokenizer
        self.model = model
        self.seq_len = seq_len
        self.top_p = top_p
        self.temperature = temperature
        self.use_gpu = use_gpu

    def _sample_from_top_p(self, probs: np.ndarray) -> Tuple[int, float]:
        # Sort probabilities and indices.
        indices, probs = np.argsort(probs)[::-1], np.sort(probs)[::-1]

        # Create top-p mask.
        mask = probs.cumsum(axis=-1) < self.top_p
        mask[:, 0] = True

        # Use gumbel-max trick to sample next tokens.
        probs += np.where(mask,
                          np.random.gumbel(size=probs.shape),
                          np.full_like(probs, -np.inf))
        next_words = probs.argmax(axis=-1)

        return next_words, probs[next_words]

    def _predict_probs(self,
                       words: List[List[int]],
                       past: Optional[List[Past]] = None
                       ) -> Tuple[np.ndarray, List[Past]]:
        with torch.no_grad():
            x = torch.tensor(words,
                             dtype=torch.long,
                             device='cuda' if self.use_gpu else 'cpu')
            logits, past = self.model(x, past)

        # If tokens are predicted on GPU, move the calculated logits to CPU.
        if self.use_gpu:
            logits = logits.cpu().float()

        probs = (logits[:, -1, :] / self.temperature).softmax(axis=-1).numpy()
        return probs, past

    def generate(self, context: str, samples: int = 20) -> Tuple[str, float]:
        context = self.tokenizer.encode(context)
        sentences = [[self.vocab.bos_idx] + [self.vocab[t] for t in context]
                     for _ in range(samples)]
        log_probs = [0 for _ in range(samples)]

        current, past = sentences, None
        for _ in range(len(sentences[0]), self.seq_len):
            probs, past = self._predict_probs(current, past)
            next_words, next_probs = self._sample_from_top_p(probs)

            # Add sampled next tokens to the sequences.
            for i in range(samples):
                if sentences[i][-1] != self.vocab.eos_idx:
                    sentences[i].append(next_words[i])
                    log_probs[i] += np.log(next_probs[i])

            current = next_words.unsqueeze(1)

        words = [self.tokenizer.decode(words) for words in sentences]
        return max(list(zip(words, log_probs)), key=lambda sample: sample[1])
