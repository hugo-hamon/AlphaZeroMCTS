import torch

from game import Connect2Game
from model import Connect2Model


args = {
    'numItersForTrainExamplesHistory': 20,
    'checkpoint_path': 'latest.pth',
    'num_simulations': 100,
    'batch_size': 64,
    'numIters': 500,
    'numEps': 100,
    'epochs': 2
}


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du {device}")

    game = Connect2Game()
    board_size = game.get_board_size()
    action_size = game.get_action_size()

    model = Connect2Model(board_size, action_size, device)