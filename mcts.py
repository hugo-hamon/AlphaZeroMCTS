from __future__ import annotations
from model import Connect2Model
from game import Connect2Game
from typing import Optional
import numpy as np
import math

def ucb_score(parent: Node, child: Node) -> float:
    """The score for an action that would transition between the parent and child. """
    prior_score = child.prior * math.sqrt(parent.visit_count) / (child.visit_count + 1)
    value_score = -child.value() if child.visit_count > 0 else 0
    return value_score + prior_score


class Node:

    def __init__(self, prior: float, to_play: int) -> None:
        self.visit_count = 0
        self.to_play = to_play
        self.prior = prior
        self.value_sum = 0.
        self.children: dict[int, Node] = {}
        self.state = None

    def expanded(self) -> bool:
        """Return True if the node has children, False otherwise."""
        return len(self.children) > 0
    
    def value(self) -> float:
        """Return the value of the node"""
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count
    
    def select_action(self, temperature: float) -> int:
        """Select action according to the visit count distribution and the temperature."""
        visit_counts = np.array([child.visit_count for child in self.children.values()])
        actions = list(self.children.keys())
        if temperature == 0:
            return actions[np.argmax(visit_counts)]
        elif temperature == float("inf"):
            return np.random.choice(actions)
        else:
            # Paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
            return np.random.choice(actions, p=visit_count_distribution)
    
    def select_child(self) -> tuple[int, Optional[Node]]:
        """Select the child with the highest UCB score."""
        best_score = -np.inf
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            score = ucb_score(self, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child
    
    def expand(self, state: np.ndarray, action_probs: np.ndarray, to_play: int) -> None:
        """We expand a node and keep track of the prior policy probability given by neural network"""
        self.to_play = to_play
        self.state = state
        for a, prob in enumerate(action_probs):
            if prob != 0:
                self.children[a] = Node(prior=prob, to_play=self.to_play * -1)


class MCTS:

    def __init__(self, game: Connect2Game, model: Connect2Model, args: dict) -> None:
        self.game = game
        self.model = model
        self.args = args

    def run(self, model: Connect2Model, state: np.ndarray, to_play: int) -> Node:

        root = Node(0, to_play)

        # EXPAND root
        action_probs, value = model.predict(state)
        valid_moves = self.game.get_valid_moves(state)
        action_probs = action_probs * valid_moves  # mask invalid moves
        action_probs /= np.sum(action_probs).astype(float)
        root.expand(state, action_probs, to_play)

        for _ in range(self.args['num_simulations']):
            node = root
            search_path: list[Node] = [node]
            action = -1

            # SELECT
            while node.expanded():
                action, node = node.select_child()
                if node is None:
                    raise ValueError("Node is None")
                search_path.append(node)

            parent = search_path[-2]
            if parent.state is None:
                raise ValueError("Parent state is None")
            state = parent.state
            # Now we're at a leaf node and we would like to expand
            # Players always play from their own perspective
            next_state, _ = self.game.get_next_state(state, player=1, action=action)
            # Get the board from the perspective of the other player
            next_state = self.game.get_canonical_board(next_state, player=-1)

            # The value of the new state from the perspective of the other player
            value = self.game.get_reward_for_player(next_state, player=1)
            if value is None:
                # If the game has not ended:
                # EXPAND
                action_probs, value = model.predict(next_state)
                valid_moves = self.game.get_valid_moves(next_state)
                action_probs = action_probs * valid_moves  # mask invalid moves
                action_probs /= np.sum(action_probs).astype(float)
                node.expand(next_state, action_probs, parent.to_play * -1)
            if value is None:
                raise ValueError("Value is None")
            self.backpropagate(search_path, float(value), parent.to_play * -1)

        return root

    def backpropagate(self, search_path: list[Node], value: float, to_play: int) -> None:
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        for node in reversed(search_path):
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1