import pickle
import random


class NimAI:
    def __init__(self, q_table_file="nim_q_table.pkl"):
        self.q_table = self.load_q_table(q_table_file)

    def load_q_table(self, filename):
        """Load Q-table from file"""
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print("Q-table not found. Train the AI first!")
            return {}

    def get_state(self, piles):
        """Convert piles into a tuple (state)"""
        return tuple(piles)

    def get_best_action(self, state):
        """Choose the best action based on Q-values"""
        possible_actions = [(i, j) for i in range(len(state)) for j in range(1, state[i] + 1)]
        q_values = {action: self.q_table.get((state, action), 0) for action in possible_actions}
        return max(q_values, key=q_values.get) if q_values else random.choice(possible_actions)


def play_nim():
    piles = [random.randint(1, 10) for _ in range(3)]
    ai = NimAI()

    print("Welcome to Nim! Try to beat the AI!")
    print(f"Starting piles: {piles}")

    while sum(piles) > 0:
        print(f"Current piles: {piles}")
        while True:
            try:
                pile = int(input("Choose a pile (0-2): "))
                remove = int(input("How many stones to remove?: "))
                if 0 <= pile < len(piles) and 1 <= remove <= piles[pile]:
                    break
                print("Invalid move! Try again.")
            except ValueError:
                print("Enter numbers only!")

        piles[pile] -= remove
        if sum(piles) == 0:
            print("AI wins! 🤖🏆")
            break

        state = tuple(piles)
        pile, remove = ai.get_best_action(state)
        print(f"AI removes {remove} from pile {pile}")
        piles[pile] -= remove

        if sum(piles) == 0:
            print("You win! 🎉")
            break


if __name__ == "__main__":
    play_nim()