# Preface
# This Jupyter notebook provides implementations of the Minimax algorithm and Monte Carlo Tree Search (MCTS) for playing Connect Four.

# Imports
import numpy as np
import random

# Classes
class ConnectFour:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1  # 1 for player 1, -1 for player 2

    def is_terminal(self, state):
        # Check for a winning state or a full board
        return self.check_winner(state) or self.is_board_full(state)

    def utility(self, state, player):
        # Evaluate the utility of the state for the given player
        winner = self.check_winner(state)
        if winner == player:
            return 1  # Player wins
        elif winner == -player:
            return -1  # Opponent wins
        else:
            return 0  # It's a tie

    def actions(self, state):
        # Return legal actions (columns where a disc can be dropped)
        return [col for col in range(7) if state[0, col] == 0]

    def result(self, state, action):
        # Apply the action (drop a disc) to the current state
        new_state = state.copy()
        row = np.argmax(new_state[:, action] == 0)
        new_state[row, action] = self.current_player
        return new_state

    def check_winner(self, state):
        # Check for a winning state
        for player in [1, -1]:
            # Check horizontally
            for row in range(6):
                for col in range(4):
                    if np.all(state[row, col:col + 4] == player):
                        return player

            # Check vertically
            for row in range(3):
                for col in range(7):
                    if np.all(state[row:row + 4, col] == player):
                        return player

            # Check diagonally (down-right)
            for row in range(3):
                for col in range(4):
                    if np.all(np.diagonal(state[row:row + 4, col:col + 4]) == player):
                        return player

            # Check diagonally (up-right)
            for row in range(3, 6):
                for col in range(4):
                    if np.all(np.diagonal(state[row:row - 4:-1, col:col + 4]) == player):
                        return player

        return 0  # No winner

    def is_board_full(self, state):
        # Check if the board is full
        return not any(state[0] == 0)

# Minimax Algorithm
def minimax_search(game, state):
    player = game.current_player
    _, move = max_value(game, state, player)
    return move

def max_value(game, state, player):
    if game.is_terminal(state):
        return game.utility(state, player), None

    v, move = float("-inf"), None
    for action in game.actions(state):
        v2, _ = min_value(game, game.result(state, action), player)
        if v2 > v:
            v, move = v2, action

    return v, move

def min_value(game, state, player):
    if game.is_terminal(state):
        return game.utility(state, player), None

    v, move = float("inf"), None
    for action in game.actions(state):
        v2, _ = max_value(game, game.result(state, action), player)
        if v2 < v:
            v, move = v2, action

    return v, move

# Monte Carlo Tree Search (MCTS)
class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.wins = 0

def monte_carlo_tree_search(game, state, max_iterations=1000):
    root = Node(state)

    for _ in range(max_iterations):
        leaf = select(root)
        child = expand(leaf)
        result = simulate(child)
        back_propagate(result, child)

    return best_child(root).action

def select(node):
    while node.children:
        if not all(child.visits > 0 for child in node.children):
            return node
        node = best_child(node)
    return node

def expand(node):
    legal_actions = node.state[0] == 0
    untried_actions = [action for action in range(7) if legal_actions[action] and all(action != child.action for child in node.children)]
    
    if untried_actions:
        action = random.choice(untried_actions)
        new_state = game.result(node.state, action)
        child = Node(new_state, parent=node, action=action)
        node.children.append(child)
        return child
    else:
        return random.choice(node.children)

def simulate(node):
    state = node.state.copy()
    current_player = game.current_player

    while not game.is_terminal(state):
        action = random.choice(game.actions(state))
        state = game.result(state, action)
        current_player *= -1

    return game.utility(state, game.current_player)

def back_propagate(result, node):
    while node:
        node.visits += 1
        node.wins += result
        node = node.parent

def best_child(node):
    return max(node.children, key=lambda child: child.wins / child.visits)

# Testing the algorithms on Connect Four
connect_four_game = ConnectFour()

# Using Minimax
minimax_move = minimax_search(connect_four_game, connect_four_game.board.copy())
print("Minimax Move:", minimax_move)

# Using Monte Carlo Tree Search
mcts_move = monte_carlo_tree_search(connect_four_game, connect_four_game.board.copy())
print("MCTS Move:", mcts_move)
