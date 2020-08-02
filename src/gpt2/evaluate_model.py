import argparse
import torch
import torch.nn as nn
from gpt2.modeling import Transformer
from gpt2.data import Dataset, Vocab, TokenizedCorpus
from gpt2.evaluation import EvaluationSpec, EvaluateConfig, Evaluator
from typing import Dict


class GPT2EvaluationSpec(EvaluationSpec):
    def __init__(self, eval_corpus: str, vocab_path: str, seq_len: int,
                 layers: int, heads: int, dims: int, rate: int):
        self.eval_corpus = eval_corpus
        self.vocab_path = vocab_path
        self.seq_len = seq_len
        self.layers = layers
        self.heads = heads
        self.dims = dims
        self.rate = rate

    def initialize(self):
        self.vocab = Vocab(vocab_path=self.vocab_path)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def prepare_dataset(self) -> Dataset:
        return TokenizedCorpus(corpus_path=self.eval_corpus,
                               vocab=self.vocab,
                               seq_len=self.seq_len,
                               repeat=False)

    def construct_model(self) -> nn.Module:
        return Transformer(layers=self.layers, pad_idx=self.vocab.pad_idx,
                           words=len(self.vocab), seq_len=self.seq_len,
                           heads=self.heads, dims=self.dims, rate=self.rate,
                           dropout=0, bidirectional=False)

    def eval_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                       ) -> Dict[str, torch.Tensor]:
        logits, _ = model(data['input'], past=None)
        loss = self.criterion(logits.transpose(1, 2), data['output'])

        mask = (data['output'] != self.vocab.pad_idx).float()
        loss = (loss * mask).sum() / mask.sum()
        perplexity = (loss.exp() * mask).sum() / mask.sum()

        return {'loss': loss, 'perplexity': perplexity}


def evaluate_gpt2_model(args: argparse.Namespace):
    spec = GPT2EvaluationSpec(
        eval_corpus=args.eval_corpus, vocab_path=args.vocab_path,
        seq_len=args.seq_len, layers=args.layers, heads=args.heads,
        dims=args.dims, rate=args.rate)
    config = EvaluateConfig(
        batch_eval=args.batch_eval, total_steps=args.total_steps,
        use_gpu=args.use_gpu)

    print(Evaluator(spec, config).evaluate(from_model=args.model_path))


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('evaluate', help='evaluate GPT-2 model')

    parser.add_argument('--model_path', required=True,
                        help='trained GPT-2 model file path')

    group = parser.add_argument_group('Corpus and vocabulary')
    group.add_argument('--eval_corpus', required=True,
                       help='evaluation corpus file path')
    group.add_argument('--vocab_path', required=True,
                       help='vocabulary file path')

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

    group = parser.add_argument_group('Evaluation options')
    group.add_argument('--batch_eval', default=64, type=int,
                       help='number of evaluation batch size')
    group.add_argument('--total_steps', default=-1, type=int,
                       help='number of total evaluation steps')
    group.add_argument('--use_gpu', action='store_true',
                       help='use gpu device in inferencing')

    parser.set_defaults(func=evaluate_gpt2_model)
