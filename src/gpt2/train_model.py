import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from gpt2.utils import fusing
from gpt2.modeling import Transformer
from gpt2.data import Dataset, Vocab, TokenizedCorpus
from gpt2.training import TrainConfig, TrainingSpec, Trainer
from typing import Tuple, Iterator, Dict


class GPT2TrainingSpec(TrainingSpec):
    def __init__(self, train_corpus: str, eval_corpus: str, vocab_path: str,
                 seq_len: int, layers: int, heads: int, dims: int, rate: int,
                 dropout: float, base_lr: float, wd_rate: float,
                 total_steps: int, use_grad_ckpt: bool):
        self.train_corpus = train_corpus
        self.eval_corpus = eval_corpus
        self.vocab_path = vocab_path
        self.seq_len = seq_len
        self.layers = layers
        self.heads = heads
        self.dims = dims
        self.rate = rate
        self.dropout = dropout
        self.base_lr = base_lr
        self.wd_rate = wd_rate
        self.total_steps = total_steps
        self.use_grad_ckpt = use_grad_ckpt

    def initialize(self):
        self.vocab = Vocab(vocab_path=self.vocab_path)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_idx,
                                             reduction='mean')

    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        train_dataset = TokenizedCorpus(corpus_path=self.train_corpus,
                                        vocab=self.vocab,
                                        seq_len=self.seq_len)
        eval_dataset = TokenizedCorpus(corpus_path=self.eval_corpus,
                                       vocab=self.vocab,
                                       seq_len=self.seq_len)
        return train_dataset, eval_dataset

    def construct_model(self) -> nn.Module:
        return Transformer(layers=self.layers, pad_idx=self.vocab.pad_idx,
                           words=len(self.vocab), seq_len=self.seq_len,
                           heads=self.heads, dims=self.dims, rate=self.rate,
                           dropout=self.dropout, bidirectional=False)

    def create_optimizer(self, params: Iterator[nn.Parameter]
                         ) -> Tuple[optim.Optimizer,
                                    optim.lr_scheduler._LRScheduler]:
        optimizer = fusing.Adam(
            params, lr=self.base_lr, weight_decay=self.wd_rate)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1 - step / self.total_steps)
        return optimizer, scheduler

    def train_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                        ) -> Dict[str, torch.Tensor]:
        logits = model(data['input'], use_grad_ckpt=self.use_grad_ckpt)
        loss = self.criterion(logits.transpose(1, 2), data['output'])
        return {'loss': loss}

    def eval_objective(self, data: Dict[str, torch.Tensor], model: nn.Module
                       ) -> Dict[str, torch.Tensor]:
        logits, _ = model(data['input'], past=None)
        loss = self.criterion(logits.transpose(1, 2), data['output'])
        return {'loss': loss}


def train_gpt2_model(args: argparse.Namespace):
    spec = GPT2TrainingSpec(
        train_corpus=args.train_corpus, eval_corpus=args.eval_corpus,
        vocab_path=args.vocab_path, seq_len=args.seq_len, layers=args.layers,
        heads=args.heads, dims=args.dims, rate=args.rate, dropout=args.dropout,
        base_lr=args.base_lr, wd_rate=args.wd_rate,
        total_steps=args.total_steps, use_grad_ckpt=args.use_grad_ckpt)
    config = TrainConfig(
        batch_train=args.batch_train, batch_eval=args.batch_eval,
        total_steps=args.total_steps, eval_steps=args.eval_steps,
        save_steps=args.save_steps, save_model_path=args.save_model_path,
        save_checkpoint_path=args.save_checkpoint_path,
        description='Train GPT-2 model',
        log_format='train/loss: {train_loss:.4f}, eval/loss: {eval_loss:.4f}',
        use_amp=args.use_amp, gpus=args.gpus)

    Trainer(spec, config).train(from_checkpoint=args.from_checkpoint,
                                from_pretrained=args.from_pretrained)


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('train', help='train GPT-2 model')

    group = parser.add_argument_group('Corpus and vocabulary')
    group.add_argument('--train_corpus', required=True,
                       help='training corpus file path')
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
    group.add_argument('--dropout', default=0.1, type=float,
                       help='probability that each element is dropped')

    group = parser.add_argument_group('Training and evaluation')
    group.add_argument('--batch_train', default=64, type=int,
                       help='number of training batch size')
    group.add_argument('--batch_eval', default=64, type=int,
                       help='number of evaluation batch size')
    group.add_argument('--base_lr', default=1e-4, type=float,
                       help='default learning rate')
    group.add_argument('--wd_rate', default=1e-2, type=float,
                       help='weight decay rate')

    group.add_argument('--total_steps', default=1000000, type=int,
                       help='number of total training steps')
    group.add_argument('--eval_steps', default=500, type=int,
                       help='period to evaluate model and record metrics')
    group.add_argument('--save_steps', default=1000, type=int,
                       help='period to save training state to checkpoint')

    group = parser.add_argument_group('Saving and restoring')
    group.add_argument('--save_model_path', default='model.pth',
                       help='save trained model weights to the file')
    group.add_argument('--save_checkpoint_path', default='checkpoint.pth',
                       help='save training state to the checkpoint file')
    group.add_argument('--from_checkpoint', default=None,
                       help='load last training state from checkpoint file')
    group.add_argument('--from_pretrained', default=None,
                       help='initialize parameters from pretrained model')

    group = parser.add_argument_group('Extensions')
    group.add_argument('--use_amp', action='store_true',
                       help='use automatic mixed-precision in training')
    group.add_argument('--use_grad_ckpt', action='store_true',
                       help='use gradient checkpointing in transformer layers')
    group.add_argument('--gpus', default=None, type=int,
                       help='number of gpu devices to use in training')

    parser.set_defaults(func=train_gpt2_model)
