import argparse
import torch.nn as nn
from gpt2.data import Vocab, Tokenizer
from gpt2.modeling import Transformer
from gpt2.generation import GenerationSpec, GenerateConfig, Generator
from typing import List


class GPT2GenerationSpec(GenerationSpec):
    def __init__(self, vocab_path: str, seq_len: int, layers: int, heads: int,
                 dims: int, rate: int):
        self.vocab_path = vocab_path
        self.seq_len = seq_len
        self.layers = layers
        self.heads = heads
        self.dims = dims
        self.rate = rate

    def initialize(self):
        self.vocab = Vocab(vocab_path=self.vocab_path)
        self.tokenizer = Tokenizer(vocab=self.vocab)

    def construct_model(self) -> nn.Module:
        return Transformer(layers=self.layers, pad_idx=self.vocab.pad_idx,
                           words=len(self.vocab), seq_len=self.seq_len,
                           heads=self.heads, dims=self.dims, rate=self.rate,
                           dropout=0, bidirectional=False)

    def encode_context(self, context: str) -> List[int]:
        tokens = [self.vocab[t] for t in self.tokenizer.encode(context)]
        tokens = [self.vocab.bos_idx] + tokens

        return tokens

    def decode_tokens(self, tokens: List[int]) -> str:
        if self.vocab.eos_idx in tokens:
            tokens = tokens[:tokens.index(self.vocab.eos_idx)+1]
        return self.tokenizer.decode([self.vocab[t] for t in tokens])


def generate_sentence_with_gpt2_model(args: argparse.Namespace):
    spec = GPT2GenerationSpec(
        vocab_path=args.vocab_path, seq_len=args.seq_len, layers=args.layers,
        heads=args.heads, dims=args.dims, rate=args.rate)
    config = GenerateConfig(
        seq_len=args.seq_len, nucleus_prob=args.nucleus_prob,
        use_gpu=args.use_gpu)

    generator = Generator(spec, config)
    generator.initialize(from_model=args.model_path)

    while True:
        print(generator.generate(input('>>')))


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        'generate', help='generate sentences with GPT-2 model')

    parser.add_argument('--vocab_path', required=True,
                        help='vocabulary file path')
    parser.add_argument('--model_path', required=True,
                        help='trained GPT-2 model file path')

    group = parser.add_argument_group('Model configurations')
    group.add_argument('--seq_len', default=64, type=int,
                       help='maximum sequence length')
    group.add_argument('--layers', default=12, type=int,
                       help='number of transformer layers')
    group.add_argument('--heads', default=16, type=int,
                       help='number of multi-heads in attention layer')
    group.add_argument('--dims', default=1024, type=int,
                       help='dimension of representation in each layer')
    group.add_argument('--rate', default=4, type=int,
                       help='increase rate of dimensionality in bottleneck')

    group = parser.add_argument_group('Generation options')
    group.add_argument('--nucleus_prob', default=0.85, type=float,
                       help='probability threshold for nucleus sampling')
    group.add_argument('--use_gpu', action='store_true',
                       help='use gpu device in inferencing')

    parser.set_defaults(func=generate_sentence_with_gpt2_model)
