from typing import Optional
import numpy as np


class Connect2Game:

    def __init__(self) -> None:
        self.columns = 4
        self.win = 2

    def get_init_board(self) -> np.ndarray:
        """Return the initial board (state) of the game"""
        return np.zeros((self.columns,), dtype=np.int_)

    def get_board_size(self) -> int:
        """Return the size of the board (state)"""
        return self.columns

    def get_action_size(self) -> int:
        """Return the size of the action space"""
        return self.columns

    def get_next_state(self, board: np.ndarray, player: int, action: int) -> tuple[np.ndarray, int]:
        """Return the next state of the game with the given action"""
        b = np.copy(board)
        b[action] = player

        return (b, -player)

    def has_legal_moves(self, board: np.ndarray) -> bool:
        """Return True if there are legal moves left"""
        return any(board[index] == 0 for index in range(self.columns))

    def get_valid_moves(self, board: np.ndarray) -> list[int]:
        """Return a list of valid moves"""
        valid_moves = [0] * self.get_action_size()

        for index in range(self.columns):
            if board[index] == 0:
                valid_moves[index] = 1

        return valid_moves

    def is_win(self, board: np.ndarray, player: int) -> bool:
        """Return True if the given player has won"""
        count = 0
        for index in range(self.columns):
            count = count + 1 if board[index] == player else 0
            if count == self.win:
                return True

        return False

    def get_reward_for_player(self, board: np.ndarray, player: int) -> Optional[int]:
        """
            Return the reward for the given player,
            1 if they won, -1 if they lost, 0 if the game is not over
        """

        if self.is_win(board, player):
            return 1
        if self.is_win(board, -player):
            return -1
        return None if self.has_legal_moves(board) else 0

    def get_canonical_board(self, board: np.ndarray, player: int) -> np.ndarray:
        """Return the canonical board for the given player"""
        return player * board
