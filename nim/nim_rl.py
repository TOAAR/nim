import random
import pickle
import numpy as np


class NimQLearner:
    def __init__(self, num_piles=3, max_stones=10, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.num_piles = num_piles
        self.max_stones = max_stones
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}  # Q-values dictionary

    def get_state(self, piles):
        """Convert piles into a tuple (state)"""
        return tuple(piles)

    def get_possible_actions(self, state):
        """Return all valid (pile, remove) actions"""
        actions = []
        for i in range(len(state)):
            for j in range(1, state[i] + 1):
                actions.append((i, j))
        return actions

    def choose_action(self, state):
        """Epsilon-greedy policy for action selection"""
        if random.random() < self.epsilon:  # Explore
            return random.choice(self.get_possible_actions(state))
        else:  # Exploit
            q_values = {action: self.q_table.get((state, action), 0) for action in self.get_possible_actions(state)}
            return max(q_values, key=q_values.get) if q_values else None

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-values using Q-learning formula"""
        max_future_q = max([self.q_table.get((next_state, a), 0) for a in self.get_possible_actions(next_state)],
                           default=0)
        old_q = self.q_table.get((state, action), 0)
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[(state, action)] = new_q

    def train(self, episodes=10000):
        """Train the AI by playing against itself"""
        for episode in range(episodes):
            piles = [random.randint(1, self.max_stones) for _ in range(self.num_piles)]
            state = self.get_state(piles)
            turn = 0  # 0 = AI1, 1 = AI2
            history = []

            while sum(piles) > 0:
                action = self.choose_action(state)
                if action is None:
                    break

                pile, remove = action
                piles[pile] -= remove
                next_state = self.get_state(piles)

                history.append((state, action, next_state))
                state = next_state
                turn = 1 - turn

            # Assign rewards (winning = +1, losing = -1)
            winner = 1 - turn
            for s, a, ns in reversed(history):
                reward = 1 if winner == 0 else -1
                self.update_q_value(s, a, reward, ns)
                winner = 1 - winner

        print("Training completed!")

    def save_q_table(self, filename="nim_q_table.pkl"):
        """Save Q-table to file"""
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename="nim_q_table.pkl"):
        """Load Q-table from file"""
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print("No Q-table found, starting fresh.")


if __name__ == "__main__":
    ai = NimQLearner()
    ai.train(10000)
    ai.save_q_table()
