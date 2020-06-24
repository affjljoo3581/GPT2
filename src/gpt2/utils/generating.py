import torch
import torch.nn as nn
import numpy as np
from ..data.vocabulary import Vocabulary
from ..data.tokenization import Tokenizer
from typing import Tuple, List


class Generator(object):
    def __init__(self,
                 vocab: Vocabulary,
                 tokenizer: Tokenizer,
                 model: nn.Module,
                 seq_len: int,
                 temperature: float = 0.8,
                 topk: int = 40):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.model = model
        self.seq_len = seq_len
        self.temperature = temperature
        self.topk = topk

    def _sample_next_word(self, tokens: List[int], past) -> int:
        # Calculate next-token distribution by using input tokens and contexts.
        with torch.no_grad():
            x = torch.tensor([tokens], dtype=torch.long)
            logits, past = self.model(x, past)

        # Apply softmax to the logits and take top-k candidates.
        probs = (logits[0, -1] / self.temperature).softmax(-1).numpy()
        candidates = probs.argsort()[-self.topk:][::-1]

        # Sample next token from calculated distribution.
        target_probs = probs[candidates]
        next_token = np.random.choice(
            candidates, p=(target_probs / target_probs.sum()))

        # Calculate log probability of sampled token.
        log_prob = np.log(probs[next_token])

        return next_token, log_prob, past

    def _sample(self, context: str) -> Tuple[str, float]:
        # Encode the given context sentence and add begin-of-sentence token.
        sequence = [self.vocab[t] for t in self.tokenizer.encode(context)]
        sequence = [self.vocab.bos_idx] + sequence

        past, total_log_prob = None, 0

        tokens = sequence
        while len(sequence) < self.seq_len:
            # Predict next token from input sequence and contexts.
            next_token, log_prob, past = \
                self._sample_next_word(tokens, past)

            # Add log probability of predicted token.
            total_log_prob += log_prob

            # Make the predicted token as an input sequence and add the token
            # to the whole sequence.
            tokens = [next_token]
            sequence.append(next_token)

            # Finish generating sentence if end-of-sentence token is
            # predicted.
            if next_token == self.vocab.eos_idx:
                break

        # Cast token indices to subwords and merge them.
        sentence = self.tokenizer.decode([self.vocab[t] for t in sequence])

        return sentence, total_log_prob

    def generate(self, context: str, samples: int = 20) -> Tuple[str, float]:
        return max([self._sample(context)
                    for _ in range(samples)], key=lambda sample: sample[1])
