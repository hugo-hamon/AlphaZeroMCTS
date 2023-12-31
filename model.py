import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect2Model(nn.Module):

    def __init__(self, board_size: int, action_size: int, device: torch.device) -> None:

        super(Connect2Model, self).__init__()

        self.device = device
        self.size = board_size
        self.action_size = action_size

        self.fc1 = nn.Linear(in_features=self.size, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=16)

        # Heads for action and value
        self.action_head = nn.Linear(in_features=16, out_features=self.action_size)
        self.value_head = nn.Linear(in_features=16, out_features=1)

        self.to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward through the network the given board state and return the policy and value."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_logits = self.action_head(x)
        value_logit = self.value_head(x)

        return F.softmax(action_logits, dim=1), torch.tanh(value_logit)
    
    def predict(self, board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the policy and value for the given board state."""
        torch_board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        torch_board = torch_board.view(1, self.size)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(torch_board)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]
    
    def load_checkpoint(self, folder: str, filename: str) -> None:
        """Load the model from a checkpoint file."""
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"No model in path {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])


class Connect4Model(nn.Module):

    def __init__(self, board_size: int, action_size: int, device: torch.device) -> None:

        super(Connect4Model, self).__init__()

        self.device = device
        self.size = board_size
        self.action_size = action_size

        self.fc1 = nn.Linear(in_features=self.size, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=16)

        # Heads for action and value
        self.action_head = nn.Linear(in_features=16, out_features=self.action_size)
        self.value_head = nn.Linear(in_features=16, out_features=1)

        self.to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward through the network the given board state and return the policy and value."""
        # flatten the matrix
        x = x.view(-1, self.size)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        action_logits = self.action_head(x)
        value_logit = self.value_head(x)

        return F.softmax(action_logits, dim=1), torch.tanh(value_logit)
    
    def predict(self, board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the policy and value for the given board state."""
        torch_board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        torch_board = torch_board.view(1, self.size)
        self.eval()
        with torch.no_grad():
            pi, v = self.forward(torch_board)

        return pi.data.cpu().numpy()[0], v.data.cpu().numpy()[0]
    
    def load_checkpoint(self, folder: str, filename: str) -> None:
        """Load the model from a checkpoint file."""
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError(f"No model in path {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])