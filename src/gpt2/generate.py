import torch
import argparse
from .data.vocabulary import Vocab
from .data.tokenization import Tokenizer
from .modeling.transformer import Transformer
from .misc.generating import Generator


def _generate_sentence(args: argparse.Namespace):
    # Prepare tokenizer and model.
    vocab = Vocab(vocab_path=args.vocab)
    tokenizer = Tokenizer(
        vocab, special_tokens=[vocab.unk_idx] + vocab.additional_tokens)

    model = Transformer(layers=args.layers, pad_idx=vocab.pad_idx,
                        words=len(vocab), seq_len=args.seq_len,
                        heads=args.heads, dims=args.dims, rate=args.rate,
                        dropout=0, bidirectional=False)
    model.eval()

    # Create integrated sentence generator.
    generator = Generator(vocab, tokenizer, model, seq_len=args.seq_len,
                          temp=args.temp, topk=args.topk, use_gpu=args.use_gpu)

    # Restore trained GPT-2 parameters from checkpoint.
    ckpt = torch.load(args.checkpoint,
                      map_location='cuda' if args.use_gpu else 'cpu')
    model.load_state_dict(ckpt['model'])

    # Start generating sentence interactively.
    while True:
        context = input('>>')
        sentence, log_prob = generator.generate(context, samples=args.samples)
        print(f'[log prob: {log_prob:.4f}] {sentence}')


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        'generate', help='generate sentence by using GPT-2 model.')

    parser.add_argument('--vocab', required=True,
                        help='vocabulary file path')
    parser.add_argument('--checkpoint', required=True,
                        help='trained model checkpoint')
    parser.add_argument('--seq_len', default=64, type=int,
                        help='maximum length of sequences')
    parser.add_argument('--layers', default=12, type=int,
                        help='number of decoder layers')
    parser.add_argument('--heads', default=16, type=int,
                        help='number of multi-heads in attention')
    parser.add_argument('--dims', default=1024, type=int,
                        help='dimension of representation in each layer')
    parser.add_argument('--rate', default=4, type=int,
                        help='increase rate of dimensionality in bottleneck')
    parser.add_argument('--temp', default=0.8, type=float,
                        help='scale factor of prediction logits')
    parser.add_argument('--samples', default=20, type=int,
                        help='number of samples to generate')
    parser.add_argument('--topk', default=40, type=int,
                        help='number of next-word candidates')
    parser.add_argument('--use_gpu', action='store_true',
                        help='use gpu for generating sentences.')

    parser.set_defaults(func=_generate_sentence)
