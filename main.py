"""TouchRL: a blind fingertip learns to find and name a hidden shape by touch.

Run with:  python main.py
"""
import platform
import matplotlib
import numpy as np
import torch
from world import World
from agent import Agent
from train import train


def main():
    # ---------------------------------------------------------------------------
    # In a world, a random shape is on a 16x16 grid, touch from a finder with 
    # patch of patch_size, and the finger moves with step_size. 
    # Patch sizes must be odd. 
    # The agent is a small network with memory, it outputs a shape guess and a move.
    # Training: guess learned supervised (cross-entropy), moves with REINFORCE.
    # ---------------------------------------------------------------------------
    num_games = 512
    world = World(num_games)
    print(world)
    agent = Agent()
    train(agent, world)



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
