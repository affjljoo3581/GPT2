import argparse
from gpt2 import (train_model,
                  evaluate_model,
                  generate_sentences,
                  visualize_metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='gpt2',
        description='PyTorch implementation of OpenAI GPT-2')
    subparsers = parser.add_subparsers(dest='subcommands')  # , required=True)
    # The above code is modified for compatibility. Argparse in Python 3.6
    # version does not support `required` option in `add_subparsers`.

    train_model.add_subparser(subparsers)
    evaluate_model.add_subparser(subparsers)
    generate_sentences.add_subparser(subparsers)
    visualize_metrics.add_subparser(subparsers)

    args = parser.parse_args()
    args.func(args)
