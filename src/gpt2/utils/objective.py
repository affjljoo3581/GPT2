import torch
import torch.nn as nn


class Objective(nn.Module):
    def set_model(self, model: nn.Module):
        self.model = model


class LMObjective(Objective):
    def __init__(self, pad_idx: int):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx,
                                             reduction='mean')

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor
                ) -> torch.Tensor:
        print(inputs)
        print(outputs)
        logits, _ = self.model(inputs, None)
        return self.criterion(logits.transpose(1, 2), outputs)
