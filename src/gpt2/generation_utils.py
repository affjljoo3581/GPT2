import torch
import torch.nn as nn
import numpy as np
from .data.vocabulary import Vocabulary
from .data.tokenization import Tokenizer
from typing import Tuple


class Generator(object):
    """Simple sentence generator.

    Arguments:
        vocab (Vocabulary): The vocabulary object.
        tokenizer (Tokenizer): Subword tokenizer.
        model: The model based on ``torch.nn.Module``.
        seq_len (int): The maximum length of each sequence.
        temperature (float): The scale factor of prediction logits.
        topk (int): The number of next-word candidates.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 tokenizer: Tokenizer,
                 model: nn.Module,
                 seq_len: int,
                 temperature: (float) = 0.8,
                 topk: int = 40):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.model = model
        self.seq_len = seq_len
        self.temperature = temperature
        self.topk = topk

    def _sample(self, context: str) -> Tuple[str, float]:
        # Encode the given context sentence and add begin-of-sentence token.
        seq = [self.vocab[t] for t in self.tokenizer.encode(context)]
        seq = [self.vocab.bos_idx] + seq

        with torch.no_grad():
            past, log_prob = None, 0

            input_seq = seq
            while len(seq) < self.seq_len:
                # Calculate next-word probabilities with lastly predicted word.
                x = torch.tensor([input_seq], dtype=torch.long)
                preds, past = self.model(x, past)

                # Sample next word.
                preds = (preds[0, -1] / self.temperature).softmax(-1).numpy()
                candidates = preds.argsort()[-self.topk:][::-1]

                probs = preds[candidates] / preds[candidates].sum()
                next_word_idx = np.random.choice(candidates, p=probs)

                # Update log probability.
                log_prob += np.log(probs[next_word_idx])

                # Finish generating sentence if end-of-sentence token is
                # predicted.
                if next_word_idx == self.vocab.eos_idx:
                    break

                # Update sequence.
                input_seq = [next_word_idx]
                seq.append(next_word_idx)

        return ' '.join(seq).replace(' ##', ''), log_prob

    def generate(self, context: str, samples: int = 20) -> Tuple[str, float]:
        """Generate sentence from the given context.

        Arguments:
            context (str): Context sentence.
            samples (int): The number of samples to generate.

        Returns:
            * A generated sentence which has highest probability.
            * Log probability of the generated sentence.

        """
        sentences = [self._sample(context) for _ in range(samples)]
        return max(sentences, key=lambda s: s[1])
