import torch
import argparse
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def _extract_dict_from_list(list_dict: List[Dict[str, Any]]
                            ) -> Dict[str, List[Any]]:
    dict_list = {k: [] for k in list_dict[0]}

    for item in list_dict:
        for k, v in item.items():
            dict_list[k].append(v)

    return dict_list


def _visualize_metrics(args: argparse.Namespace):
    ckpt = torch.load(args.checkpoint)

    # Extract metrics from checkpoint.
    steps = ckpt['metrics']['steps']
    train_metrics = _extract_dict_from_list(ckpt['metrics']['train'])
    eval_metrics = _extract_dict_from_list(ckpt['metrics']['eval'])

    plt.figure(figsize=(15, 10))

    # Plot loss for training and evaluation.
    plt.subplot(221)
    plt.plot(steps, eval_metrics['loss'], label='evaluation')
    plt.plot(steps, train_metrics['loss'], label='training')
    plt.title('Cross-Entropy Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(loc='top right')

    # Plot log-scale loss for training and evaluation.
    plt.subplot(222)
    plt.plot(steps, eval_metrics['loss'], label='evaluation')
    plt.plot(steps, train_metrics['loss'], label='training')
    plt.xscale('log')
    plt.title('Log-Scale Cross-Entropy Loss')
    plt.xlabel('Iterations (Log Scale)')
    plt.ylabel('Loss')
    plt.legend(loc='top right')

    # Plot detail loss for last 90% iterations.
    target_range = len(steps) * 9 // 10
    min_loss = min(train_metrics['loss'][-target_range:]
                   + eval_metrics['loss'][-target_range:])
    max_loss = max(train_metrics['loss'][-target_range:]
                   + eval_metrics['loss'][-target_range:])

    plt.subplot(223)
    plt.plot(steps, eval_metrics['loss'], label='evaluation')
    plt.plot(steps, train_metrics['loss'], label='training')
    plt.title('Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(loc='top right')
    plt.ylim((min_loss - 0.1, max_loss + 0.1))

    # Plot detail loss for last 30% iterations.
    target_range = len(steps) * 3 // 10

    plt.subplot(224)
    plt.plot(steps[-target_range:],
             eval_metrics['loss'][-target_range:],
             label='evaluation')
    plt.plot(steps[-target_range:],
             train_metrics['loss'][-target_range:],
             label='training')
    plt.title('Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend(loc='top right')

    # Save figure.
    plt.tight_layout()
    plt.savefig(args.figure)


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        'visualize', help='visualize metrics for training and evaluation')

    parser.add_argument('--figure',
                        default='figure.png',
                        help='figure image file path to save plot')
    parser.add_argument('--checkpoint',
                        required=True,
                        help='checkpoint file path')

    parser.set_defaults(func=_visualize_metrics)
