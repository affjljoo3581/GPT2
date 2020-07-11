from gpt2.misc.objective import LMObjective
import torch
import torch.nn as nn
from typing import Tuple


class _dummy_model(nn.Module):
    def forward(self, x: torch.Tensor, dummy: None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, x


def test_lm_objective_throws_errors():
    # Create dummy LMObjective.
    objective = LMObjective(_dummy_model(), pad_idx=0)

    # Test if the objective throws any error.
    objective(torch.zeros((10, 7, 100)), torch.randint(0, 100, (10, 7)))
