import math
import torch
import argparse
import matplotlib.pyplot as plt
from typing import List


def _plot_entire_metrics_graph(train_steps: List[int],
                               train_losses: List[float],
                               eval_steps: List[int],
                               eval_losses: List[float]):
    plt.plot(train_steps, train_losses,
             label='train', color='#4dacfa', linewidth=2, zorder=10)
    plt.plot(eval_steps, eval_losses,
             label='evaluate', color='#ff6e54', linewidth=2, zorder=0)

    plt.title('Cross-Entropy Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: f'{math.floor(x / 1000)}k'))


def _plot_log_scale_metrics_graph(train_steps: List[int],
                                  train_losses: List[float],
                                  eval_steps: List[int],
                                  eval_losses: List[float]):
    plt.plot(train_steps, train_losses,
             label='train', color='#4dacfa', linewidth=2, zorder=10)
    plt.plot(eval_steps, eval_losses,
             label='evaluate', color='#ff6e54', linewidth=2, zorder=0)

    plt.title('Log-Scale Loss Graph')
    plt.xlabel('Iterations (Log Scale)')
    plt.ylabel('Loss')

    plt.xscale('log')

    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: f'{math.floor(x / 1000)}k'))


def _plot_stretched_metrics_graph(train_steps: List[int],
                                  train_losses: List[float],
                                  eval_steps: List[int],
                                  eval_losses: List[float]):
    target_range_train = len(train_steps) * 9 // 10
    target_range_eval = len(eval_steps) * 9 // 10
    min_loss = min(train_losses[-target_range_train:]
                   + eval_losses[-target_range_eval:])
    max_loss = max(train_losses[-target_range_train:]
                   + eval_losses[-target_range_eval:])

    plt.plot(train_steps, train_losses,
             label='train', color='#4dacfa', linewidth=2, zorder=10)
    plt.plot(eval_steps, eval_losses,
             label='evaluate', color='#ff6e54', linewidth=1, zorder=1)

    plt.title('Detail Loss Graph')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    plt.ylim(min_loss - 0.1, max_loss + 0.1)

    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: f'{math.floor(x / 1000)}k'))


def _plot_highlight_metrics_graph(train_steps: List[int],
                                  train_losses: List[float],
                                  eval_steps: List[int],
                                  eval_losses: List[float]):
    beta = 2 / (100 + 1)
    smoothed_eval_losses = [eval_losses[0]]
    for loss in eval_losses[1:]:
        smoothed_eval_losses.append(
            beta * loss + (1 - beta) * smoothed_eval_losses[-1])

    target_range_train = len(train_steps) * 3 // 10
    target_range_eval = len(eval_steps) * 3 // 10

    plt.plot(train_steps[-target_range_train:],
             train_losses[-target_range_train:],
             label='train', color='#4dacfa', linewidth=2, zorder=10)
    plt.plot(eval_steps[-target_range_eval:],
             smoothed_eval_losses[-target_range_eval:],
             label='evaluate', color='#ff6e54', linewidth=2, zorder=1)
    plt.plot(eval_steps[-target_range_eval:],
             eval_losses[-target_range_eval:],
             color='#ff6e543f', linewidth=3, zorder=0)

    plt.title('Detail Loss Graph')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, pos: f'{math.floor(x / 1000)}k'))


def visualize_recorded_metrics(args: argparse.Namespace):
    metrics = torch.load(args.model_path)['metrics']
    train_steps, train_losses = zip(*metrics['train/loss'])
    eval_steps, eval_losses = zip(*metrics['eval/loss'])

    plt.figure(figsize=(10, 7))

    plt.subplot(221)
    _plot_entire_metrics_graph(train_steps, train_losses,
                               eval_steps, eval_losses)

    plt.subplot(222)
    _plot_log_scale_metrics_graph(train_steps, train_losses,
                                  eval_steps, eval_losses)

    plt.subplot(223)
    _plot_stretched_metrics_graph(train_steps, train_losses,
                                  eval_steps, eval_losses)

    plt.subplot(224)
    _plot_highlight_metrics_graph(train_steps, train_losses,
                                  eval_steps, eval_losses)

    plt.tight_layout()
    if args.interactive:
        plt.show()
    else:
        plt.savefig(args.figure)


def add_subparser(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        'visualize', help='visualize metrics recorded during training')

    parser.add_argument('--figure', default='figure.png',
                        help='output figure image file path')
    parser.add_argument('--model_path', required=True,
                        help='trained GPT-2 model file path')
    parser.add_argument('--interactive', action='store_true',
                        help='show interactive plot window')

    parser.set_defaults(func=visualize_recorded_metrics)
