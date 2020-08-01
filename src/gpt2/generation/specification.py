import torch
import torch.nn as nn
from typing import List


class GenerationSpec(object):
    def initialize(self):
        pass

    def construct_model(self) -> nn.Module:
        raise NotImplementedError()

    def encode_context(self, context: str) -> List[int]:
        raise NotImplementedError()

    def decode_tokens(self, tokens: List[int]) -> str:
        raise NotImplementedError()

    def decorate_sequence(self, sequence: torch.Tensor, offset: int
                          ) -> torch.Tensor:
        return sequence
