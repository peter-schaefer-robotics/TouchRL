"""TouchRL: a blind fingertip learns to find and name a hidden shape by touch.

Run with:  python main.py
"""
import platform
import matplotlib
import numpy as np
import torch
from world import World


def main():
    # ---------------------------------------------------------------------------
    # 1. World: random shape on a 16x16 grid, touch patch of patch_size, finger moves with step_size. 
    # Patch sizes must be odd. 
    # ---------------------------------------------------------------------------
    #
    num_games = 16
    w = World(num_games)
    w.reset()
    print(w)
    directions = np.random.randint(0,3, size=(num_games, 2))
    w.step(directions)
    print(w)


    # ---------------------------------------------------------------------------
    # 2. Agent: small network with memory, outputs a shape guess and a move
    # ---------------------------------------------------------------------------
    # TODO


    # ---------------------------------------------------------------------------
    # 3. Training: play games, learn the guess from the label and the moves
    #    with REINFORCE
    # ---------------------------------------------------------------------------
    # TODO


    # ---------------------------------------------------------------------------
    # 4. Results: compare with random moves, plot accuracy per touch, save a GIF
    # ---------------------------------------------------------------------------
    # TODO


def check_setup() -> None:
    """Print the python and package versions to confirm that the setup works."""
    print(f"Python    {platform.python_version()}")
    print(f"PyTorch    {torch.__version__}")
    print(f"NumPy      {np.__version__}")
    print(f"Matplotlib {matplotlib.__version__}")
    print("Setup OK")


if __name__ == "__main__":
    check_setup()
    main()
