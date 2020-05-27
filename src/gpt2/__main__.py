import argparse
from . import train


# Ignore warnings.
import warnings
warnings.filterwarnings(action='ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of OpenAI GPT-2')
    subparsers = parser.add_subparsers(required=True)

    # Add `train` to the parser.
    train.add_subparser(subparsers)
