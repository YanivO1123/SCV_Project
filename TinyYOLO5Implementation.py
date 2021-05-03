import numpy as np
import torch
from typing import List, Tuple, Dict, Union, Optional, Any

class TinyYolo5(torch.nn.Module):
    """
    Our implementation of a tiny YOLO-5 network:
    """
    def __init__(self, num_layers: int, input_dim: int, hidden_dimensions: List[int], output_dim: int,
                 activation_function: torch.nn.Module = torch.nn.Relu,
                 device: torch.device = torch.device("cuda")) -> None:
        super(TinyYolo5, self).__init__()
        self.to(device)

        self.input = input_dim
        self.output = output_dim

        self.activation_function = activation_function()

    def forward(self, input: torch.Tensor):
        return self.layers(input)