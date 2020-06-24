import argparse
from . import train

# Ignore warnings.
import warnings
warnings.filterwarnings(action='ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='gpt2',
        description='PyTorch implementation of OpenAI GPT-2')
    subparsers = parser.add_subparsers(dest='subcommands', required=True)

    # Add `train` keyword to the parser.
    train.add_subparser(subparsers)

    # Parse passed arguments and call corresponding function.
    args = parser.parse_args()
    args.func(args)
