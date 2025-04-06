import numpy as np
import random

class NimAI:
    def __init__(self, max_sticks=10, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        self.max_sticks = max_sticks
        self.q_table = {}  # Q-values for state-action pairs
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.randint(1, min(3, state))  # Random move (exploration)
        else:
            return max(range(1, min(4, state+1)), key=lambda x: self.get_q_value(state, x))  # Best move (exploitation)

    def update_q_value(self, state, action, reward, next_state):
        max_future_q = max([self.get_q_value(next_state, a) for a in range(1, min(4, next_state+1))], default=0)
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[(state, action)] = new_q

    def train(self, episodes=10000):
        for _ in range(episodes):
            state = self.max_sticks
            game_history = []
            while state > 0:
                action = self.choose_action(state)
                next_state = state - action
                game_history.append((state, action))
                state = next_state

            reward = -1  # Losing state
            for (state, action) in reversed(game_history):
                self.update_q_value(state, action, reward, 0 if state - action == 0 else state - action)
                reward = -reward  # Flip reward for opponent's move

    def play_against_human(self):
        """Play against a human player"""
        state = self.max_sticks
        print(f"Starting Nim with {state} sticks.")
        while state > 0:

            while True:
                try:
                    action = int(input(f"Your turn! Pick 1, 2, or 3 sticks (remaining: {state}): "))
                    if 1 <= action <= min(3, state):
                        break
                    print("Invalid move, try again.")
                except ValueError:
                    print("Please enter a number.")

            state -= action
            if state == 0:
                print("AI wins! 🤖")
                return

            action = self.choose_action(state)
            print(f"AI takes {action} sticks.")
            state -= action

            if state == 0:
                print("You win! 🎉")
                return

# Train and play
nim_ai = NimAI(max_sticks=10)
nim_ai.train(10000)
nim_ai.play_against_human()
