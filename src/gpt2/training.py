import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from .data.serving import DataLoader
from .data.vocabulary import Vocabulary
from .modeling.gpt2 import GPT2
from .training_utils import Recorder, Trainer


def _create_linear_decay_scheduler(optimizer: optim.Optimizer,
                                   warmup_iters: int,
                                   iterations: int
                                   ) -> optim.lr_scheduler._LRScheduler:
    def lr_schedule(iters):
        # Learning rate would be increased in warmup phase, and slowly decayed
        # to zero in training phase.
        warmup_rate = iters / warmup_iters
        decaying_rate = 1 - ((iters - warmup_iters)
                             / (iterations - warmup_iters))
        return min(warmup_rate, decaying_rate)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)


def _create_tqdm_progress(iterations: int, start_iters: int = 0) -> tqdm.tqdm:
    if start_iters == 0:
        return tqdm.trange(iterations, desc='Train GPT-2')

    # Create tqdm and update progress.
    tqdm_iter = tqdm.tqdm(range(start_iters, iterations),
                          total=iterations,
                          desc='Train GPT-2')
    tqdm_iter.update(start_iters)

    return tqdm_iter


def _train_gpt2_model(args: argparse.Namespace):
    # Prepare data loaders for training and evaluation.
    vocab = Vocabulary(vocab=args.vocab,
                       unk_token=args.unk_token,
                       bos_token=args.bos_token,
                       eos_token=args.eos_token,
                       pad_token=args.pad_token)
    train_loader = DataLoader(vocab,
                              corpus=args.train_corpus,
                              seq_len=args.seq_len)
    eval_loader = DataLoader(vocab,
                             corpus=args.eval_corpus,
                             seq_len=args.seq_len)

    # Create GPT-2 model.
    model = GPT2(layers=args.layers, pad_idx=vocab.pad_idx, words=len(vocab),
                 seq_len=args.seq_len, heads=args.heads, dims=args.dims,
                 rate=args.rate, dropout=args.dropout)
    model.cuda()

    # Use cross-entropy loss, AdamW optimizer and linear learning rate decaying
    # with warmup.
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr)

    scheduler = _create_linear_decay_scheduler(optimizer,
                                               warmup_iters=args.warmup_iters,
                                               iterations=args.iterations)

    # Create metrics recorder and GPT-2 trainer.
    recorder = Recorder()
    trainer = Trainer(train_loader, eval_loader, model, optimizer, scheduler,
                      criterion, recorder, use_amp=args.use_amp)

    # Restore last training state from checkpoint.
    start_iters = 0
    if args.restore is not None:
        ckpt = torch.load(args.restore)
        start_iters = ckpt['iters'] + 1
        trainer.load_state_dict(ckpt['trainer'])

    # Start training.
    tqdm_iters = _create_tqdm_progress(iterations=args.iterations,
                                       start_iters=start_iters)
    for iters in tqdm_iters:
        trainer.train(batch=args.batch_train)

        # Show metrics of training and evaluation.
        if (iters + 1) % args.eval_iters == 0:
            trainer.evaluate(batch=args.batch_eval)
            recorder.stamp(iters + 1)

            tqdm_iters.set_postfix_str(recorder.format(
                'train/loss: {train_loss:.4f}, '
                'eval/loss: {eval_loss:.4f}'))

        # Save training state to checkpoint file.
        if (iters + 1) % args.save_iters == 0:
            torch.save({'iters': iters, 'trainer': trainer.state_dict()},
                       args.checkpoint)

    # Save GPT-2 model and summary of metrics record.
    torch.save({'model': model.cpu().state_dict(),
                'metrics': recorder.summarize()},
               args.checkpoint)

    # Close all resources.
    train_loader.close()
    eval_loader.close()


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser('train', help='train GPT-2 model.')

    parser.add_argument('--train_corpus',
                        required=True,
                        help='corpus file for training')
    parser.add_argument('--eval_corpus',
                        required=True,
                        help='corpus file for evaluation')
    parser.add_argument('--vocab',
                        required=True,
                        help='vocabulary file path')
    parser.add_argument('--restore',
                        default=None,
                        help='restore from the given checkpoint file path')
    parser.add_argument('--checkpoint',
                        default='ckpt',
                        help='checkpoint file path')
    parser.add_argument('--batch_train',
                        default=64,
                        type=int,
                        help='batch size for training')
    parser.add_argument('--batch_eval',
                        default=64,
                        type=int,
                        help='batch size for evaluation')
    parser.add_argument('--seq_len',
                        default=64,
                        type=int,
                        help='maximum length of each sequence')
    parser.add_argument('--layers',
                        default=12,
                        type=int,
                        help='number of decoder layers')
    parser.add_argument('--heads',
                        default=16,
                        type=int,
                        help='number of multi-heads in attention')
    parser.add_argument('--dims',
                        default=1024,
                        type=int,
                        help='dimension of representation in each layer')
    parser.add_argument('--rate',
                        default=4,
                        type=int,
                        help='increase rate of dimensionality in bottleneck')
    parser.add_argument('--dropout',
                        default=0.1,
                        type=float,
                        help='dropout rate')
    parser.add_argument('--base_lr',
                        default=1e-4,
                        type=float,
                        help='maximum learning rate')
    parser.add_argument('--iterations',
                        default=100000,
                        type=int,
                        help='number of training iterations')
    parser.add_argument('--warmup_iters',
                        default=10000,
                        type=int,
                        help='iterations to warm up learning rate')
    parser.add_argument('--eval_iters',
                        default=500,
                        type=int,
                        help='period to evaluate')
    parser.add_argument('--save_iters',
                        default=1000,
                        type=int,
                        help='period to save training state')
    parser.add_argument('--use_amp',
                        action='store_true',
                        help='use automatic mixed-precision in training')
    parser.add_argument('--unk_token',
                        default='<unk>',
                        help='unknown token name')
    parser.add_argument('--bos_token',
                        default='<s>',
                        help='begin-of-sentence token name')
    parser.add_argument('--eos_token',
                        default='</s>',
                        help='end-of-sentence token name')
    parser.add_argument('--pad_token',
                        default='<pad>',
                        help='pad token name')

    parser.set_defaults(func=_train_gpt2_model)
