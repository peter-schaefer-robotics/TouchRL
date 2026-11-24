import torch
import torch.nn as nn
from shapes import SHAPES

class Agent(nn.Module):
    """Small recurrent network: touch in, shape guess and move scores out."""
    def __init__(self, input_size=12, hidden_size=128, num_shapes=len(SHAPES), num_moves=9):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Linear(input_size, hidden_size)
        self.memory = nn.GRUCell(hidden_size, hidden_size)
        self.guess_head = nn.Linear(hidden_size, num_shapes) 
        self.move_head = nn.Linear(hidden_size, num_moves)

    def forward(self, x, h):
        z = torch.relu(self.encoder(x))
        h = self.memory(z, h)
        guess_logits = self.guess_head(h)
        move_logits = self.move_head(h)
        return guess_logits, move_logits, h