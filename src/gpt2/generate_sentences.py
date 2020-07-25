import torch
import argparse
from gpt2.data import Vocab
from gpt2.modeling import Transformer
from gpt2.generation import Tokenizer, Generator


def generate_sentence_with_gpt2_model(args: argparse.Namespace):
    vocab = Vocab(vocab_path=args.vocab_path)
    tokenizer = Tokenizer(
        vocab, additional_tokens=[vocab.unk_token, vocab.bos_token,
                                  vocab.eos_token, vocab.pad_token])

    model = Transformer(layers=args.layers, pad_idx=vocab.pad_idx,
                        words=vocab.words, seq_len=args.seq_len,
                        heads=args.heads, dims=args.dims, rate=args.rate,
                        dropout=0, bidirectional=False)
    model.eval()

    # Create generator to generate sentences with GPT-2 model.
    generator = Generator(vocab, tokenizer, model, seq_len=args.seq_len,
                          top_p=args.top_p, use_gpu=args.use_gpu)

    # Restore trained GPT-2 parameters.
    trained_model = torch.load(
        args.model, map_location='cuda' if args.use_gpu else 'cpu')
    model.state_dict(trained_model['model'])

    while True:
        print(generator.generate(input('>>')))


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        'generate', help='generate sentences with GPT-2 model')

    parser.add_argument('--vocab_path', required=True,
                        help='vocabulary file path')
    parser.add_argument('--model', required=True,
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

    group = parser.add_argument_group('Generating options')
    group.add_argument('--top_p', default=0.85, type=float,
                       help='probability threshold for nucleus sampling')
    group.add_argument('--use_gpu', action='store_true',
                       help='use gpu device in inferencing')

    parser.set_defaults(func=generate_sentence_with_gpt2_model)
