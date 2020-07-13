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
                 temp: float = 0.8,
                 topk: int = 40):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.model = model
        self.seq_len = seq_len
        self.temp = temp
        self.topk = topk

    def _sample_next_word(self,
                          words: List[int],
                          past: Optional[Past] = None) -> int:
        with torch.no_grad():
            x = torch.tensor([words], dtype=torch.long)
            logits, past = self.model(x, past)

        probs = (logits[0, -1] / self.temp).softmax(-1).numpy()
        targets = probs.argsort()[-self.topk:][::-1]

        # Sample next token from calculated distribution.
        pred = np.random.choice(
            targets, p=(probs[targets] / probs[targets].sum()))

        return pred, np.log(probs[pred]), past

    def _sample(self, context: str) -> Tuple[str, float]:
        # Encode the given context sentence and add begin-of-sentence token.
        words = [self.vocab[t] for t in self.tokenizer.encode(context)]
        words = [self.vocab.bos_idx] + words

        current, total_log_prob, generated, past = words, 0, 0, None
        while len(current) < self.seq_len:
            pred, log_prob, past = self._sample_next_word(current, past)
            total_log_prob += log_prob

            current = [pred]
            generated += 1
            words.append(pred)

            # Finish generating sentence if end-of-sentence token is
            # predicted.
            if pred == self.vocab.eos_idx:
                break

        sentence = self.tokenizer.decode([self.vocab[t] for t in words])
        return sentence, total_log_prob / generated

    def generate(self, context: str, samples: int = 20) -> Tuple[str, float]:
        return max([self._sample(context)
                    for _ in range(samples)], key=lambda sample: sample[1])
