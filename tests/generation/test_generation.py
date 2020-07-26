import torch
import numpy as np
from gpt2.modeling import Transformer
from gpt2.generation import Generator


def test_generator_sample_from_top_p():
    generator = Generator(vocab=None, tokenizer=None, model=None, seq_len=None,
                          top_p=0.5, use_gpu=None)

    probs = torch.tensor([0.1, 0.2, 0.15, 0.08, 0.25, 0.22])
    for _ in range(100):
        assert generator._sample_from_top_p(probs) in [1, 4, 5]


def test_generator_predict_probs_output_shape():
    model = Transformer(layers=2, pad_idx=0, words=80, seq_len=16, heads=2,
                        dims=16, rate=4, dropout=0, bidirectional=False).eval()
    generator = Generator(vocab=None, tokenizer=None, model=model,
                          seq_len=None, top_p=0.5, use_gpu=None)

    probs1, past1 = generator._predict_probs(
        [np.random.randint(80) for _ in range(8)])
    probs2, past2 = generator._predict_probs(
        [generator._sample_from_top_p(probs1)], past1)

    assert probs1.shape == (80,)
    assert probs2.shape == (80,)

    for p in past1:
        assert p[0].shape == (8, 16)
        assert p[1].shape == (8, 16)
    for p in past2:
        assert p[0].shape == (9, 16)
        assert p[1].shape == (9, 16)
