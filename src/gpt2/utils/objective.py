import torch
import torch.nn as nn


class LMObjective(nn.Module):
    def __init__(self, model: nn.Module, pad_idx: int):
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx,
                                             reduction='mean')

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor
                ) -> torch.Tensor:
        logits, _ = self.model(inputs, None)
        return self.criterion(logits, outputs)
