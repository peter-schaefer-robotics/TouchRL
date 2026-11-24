import numpy as np
from shapes import SHAPES
import random

class World:
    def __init__(self, num_games, grid_size=16, patch_size=3, step_size=2, max_touch_num=16):
        self.num_games = num_games
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.step_size = step_size
        self.boards = None # all boards
        self.labels = None # true labels of shapes of all boards
        self.positions = None # positions of tactile finger on the grid of all boards. finger position per game as (y, x)
        self.padded = None # padded boards to handle edge cases
        self.max_touch_num = max_touch_num
        self.touch_num = 0

    def __str__(self):
        # Return description of world in text
        if self.boards is None:
            n = 0
            rows = []
        else:
            n = len(self.boards)
            fy, fx = self.positions[0]
            rows = []
            for r, row in enumerate(self.boards[0]):
                line = ""
                for c, v in enumerate(row):
                    if r == fy and c == fx:
                        line += "X"
                    else:
                        line += "1" if v else "0"
                rows.append(line)
        return "\n".join(rows) + "\n" + f"World with {self.num_games} games on a {self.grid_size}x{self.grid_size} grid with {self.patch_size}x{self.patch_size} patches and step size {self.step_size}. Current amount of boards: {n}"

    def reset(self):
        # Build num_games new boards with random shapes, each with a corresponding lable of the shape and a random starting position of the tactile finger
        self.boards, self.labels, self.positions = self._make_boards()
        # Pad the boards to handle edge cases
        p = self.patch_size // 2
        self.padded = np.pad(self.boards, ((0, 0), (p, p), (p, p)))
        self.touch_num = 1
        return self._observe()

    def step(self, directions):
        if self.touch_num < self.max_touch_num:
            self._move(directions)
            patches = self._observe()
            self.touch_num = self.touch_num + 1
            return patches
        else: 
            raise IndexError("Maximum touches used. Please reset world.")
    
    def _move(self, moves):
        # Move numbers 0..8 chosen by the agent and their direction:
        #   0 left-up    1 up     2 right-up
        #   3 left       4 stay   5 right
        #   6 left-down  7 down   8 right-down
        # moves: one number 0..8 per game, see the table above
        for i in range(self.num_games):
            y, x = self.positions[i]
            dx, dy = moves[i] % 3, moves[i] // 3    # column and row of the 3x3 move table
            x = x + (dx - 1) * self.step_size
            y = y + (dy - 1) * self.step_size
            self.positions[i] = (np.clip(y, 0, self.grid_size - 1), np.clip(x, 0, self.grid_size - 1))      # clip limits the agent to stay within the world

    def _make_boards(self):
        # initialize all arrays with zeros
        boards = np.zeros((self.num_games, self.grid_size, self.grid_size), dtype=np.float32)
        labels = np.zeros(self.num_games, dtype=np.int64)
        positions = np.zeros((self.num_games, 2), dtype=np.int64)
        # fill arrays with randomized shapes and positions
        for i in range(self.num_games):
            label = random.randrange(len(SHAPES))  # 0, 1 or 2
            mask = list(SHAPES.values())[label](self.grid_size)
            position = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            boards[i] = mask
            labels[i] = label
            positions[i] = position
        return boards, labels, positions

    def _observe(self):
        k = self.patch_size
        patches = np.zeros((self.num_games,k,k), dtype=np.float32)
        for i in range(self.num_games):
            y, x = self.positions[i]
            patches[i] = self.padded[i,y : y + k, x : x + k] # padding shifts every cell by patch_size // 2, so this slice is centered on the finger
        return patches
