import argparse
from . import training, generation, visualization


# Ignore warnings.
import warnings
warnings.filterwarnings(action='ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='gpt2',
        description='Pytorch implementation of OpenAI GPT-2')
    subparsers = parser.add_subparsers(dest='subcommands', required=True)

    # Add `train` to the parser.
    training.add_subparser(subparsers)

    # Add `generate` to the parser.
    generation.add_subparser(subparsers)

    # Add `visualize` to the parser.
    visualization.add_subparser(subparsers)

    # Parse arguments and call corresponding function.
    args = parser.parse_args()
    args.func(args)
