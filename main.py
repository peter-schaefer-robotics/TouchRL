"""TouchRL: a blind fingertip learns to find and name a hidden shape by touch.

Run with:  python main.py
"""

import matplotlib
import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. World: random shape on a 16x16 grid, 3x3 touch patch, finger moves
# ---------------------------------------------------------------------------
# TODO


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
    """Print the package versions to confirm that the setup works."""
    print(f"PyTorch    {torch.__version__}")
    print(f"NumPy      {np.__version__}")
    print(f"Matplotlib {matplotlib.__version__}")
    print("Setup OK")


if __name__ == "__main__":
    check_setup()
