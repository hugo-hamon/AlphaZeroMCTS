import torch
import time

from model import Connect2Model
from game import Connect2Game
from trainer import Trainer
from mcts import MCTS


args = {
    'numItersForTrainExamplesHistory': 20,
    'checkpoint_path': 'latest.pth',
    'num_simulations': 100,
    'batch_size': 64,
    'numIters': 500,
    'numEps': 100,
    'epochs': 2
}

TRAIN = True


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du {device}")

    game = Connect2Game()
    board_size = game.get_board_size()
    action_size = game.get_action_size()

    model = Connect2Model(board_size, action_size, device)

    if TRAIN:
        trainer = Trainer(game, model, args)
        trainer.learn()
    else:
        model.load_checkpoint(".", args['checkpoint_path'])
        mcts = MCTS(game, model, args)
        
        # Play against the computer
        state = game.get_init_board()
        current_player = 1
        while True:
            game.display(state)
            if current_player == 1:
                action = int(input("Action: "))
            else:
                root = mcts.run(model, state, current_player)
                action = root.select_action(temperature=0)
            
            state, current_player = game.get_next_state(state, current_player, action)
            reward = game.get_reward_for_player(state, current_player)

            if reward is not None:
                game.display(state)
                if reward == -1:
                    print(f"Player {-current_player} won!")
                elif reward == 1:
                    print(f"Player {current_player} won!")
                else:
                    print("Draw!")
                break
            