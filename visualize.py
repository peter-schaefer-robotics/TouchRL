"""Plots and animations for TouchRL: training curve, accuracy per touch, and one game step by step."""
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np
import torch

from agent import Agent
from shapes import SHAPES
from train import make_input
from world import World

SHAPE_NAMES = list(SHAPES)                 # ["circle", "rectangle", "triangle"], index = label
FIG_DIR = "docs/figures"
# board colors: 0 empty (unknown), 1 shape (hidden), 2 felt empty, 3 felt shape
BOARD_CMAP = ListedColormap(["#FFFFFF", "#E3E1DA", "#F5C4B3", "#1D9E75"])
FINGER = "#D85A30"
_animations = []                           # keep a reference, otherwise the animation is deleted before the window opens


def _path(name):
    """Full path in docs/figures; creates the folder if needed."""
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)


@torch.no_grad()                           # no training here: skip gradient bookkeeping, faster
def play(agent, num_games, random_moves=False):
    """Play num_games games without learning.

    Out: world (boards, labels), positions per touch (16, num_games, 2) as (y, x),
         shape probabilities per touch (16, num_games, num_shapes)
    """
    world = World(num_games)
    patches = world.reset()
    h = torch.zeros(num_games, agent.hidden_size)
    positions, probs = [], []
    for t in range(world.max_touch_num):
        positions.append(world.positions.copy())   # copy: the world changes this array in place
        guess_logits, move_logits, h = agent(make_input(world, patches), h)
        probs.append(torch.softmax(guess_logits, dim=1).numpy())   # scores -> probabilities
        if t < world.max_touch_num - 1:
            if random_moves:               # baseline: same guessing, but touch at random
                a = torch.randint(0, 9, (num_games,))
            else:                          # learned moves, sampled like in training
                a = torch.distributions.Categorical(logits=move_logits).sample()
            patches = world.step(a.numpy())
    return world, np.stack(positions), np.stack(probs)


def accuracy_per_touch(agent, num_games=4096, random_moves=False):
    """Share of correct guesses after each touch, on fresh games. Out: array (16,)"""
    world, _, probs = play(agent, num_games, random_moves)
    return (probs.argmax(axis=2) == world.labels).mean(axis=1)


def _smooth(values, window=25):
    """Moving average, so the trend is visible through the round-to-round noise."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _plot_history(rounds, values, color, ylabel, name, ylim, ref=None, ref_label=None):
    """One training-progress plot: raw values faint, moving average strong."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(rounds, values, color=color, alpha=0.25, linewidth=0.8)
    smooth = _smooth(values)
    ax.plot(rounds[len(rounds) - len(smooth):], smooth, color=color, linewidth=2, label="moving average")
    if ref is not None:
        ax.axhline(ref, color="gray", linestyle="--", linewidth=1, label=ref_label)
    ax.set(xlabel="training round", ylabel=ylabel, ylim=ylim)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(_path(name), dpi=150)    # save; the window opens later with show()


def plot_training(history, name="training_curve.png"):
    """Accuracy after the last touch per training round. history: list from train()."""
    rounds = np.array([h[0] for h in history]); acc = np.array([h[1] for h in history])
    _plot_history(rounds, acc, "#1D9E75", "accuracy after last touch", name, (0, 1),
                  ref=1 / len(SHAPES), ref_label="chance")


def plot_touches_to_correct(history, name="touches_to_correct.png"):
    """Per training round: from which touch on the guess was right and stayed right (games solved in the end)."""
    rounds = np.array([h[0] for h in history]); touches = np.array([h[2] for h in history])
    _plot_history(rounds, touches, "#D85A30", "touch from which the guess stays right", name, (1, 16))


def plot_accuracy(agent, name="accuracy_per_touch.png"):
    """Learned touching vs random touching, same trained guess: does choosing where to touch help?"""
    learned = accuracy_per_touch(agent)
    random_ = accuracy_per_touch(agent, random_moves=True)
    touches = np.arange(1, len(learned) + 1)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(touches, learned, color="#1D9E75", marker="o", markersize=4, label="learned moves")
    ax.plot(touches, random_, color="#888780", marker="o", markersize=4, label="random moves")
    ax.axhline(1 / len(SHAPES), color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set(xlabel="number of touches", ylabel="accuracy", ylim=(0, 1), xticks=touches)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(_path(name), dpi=150)
    return learned, random_


def _felt(board, positions, t, k):
    """Board codes after touch t: which cells the finger has felt so far (see BOARD_CMAP)."""
    p, n = k // 2, board.shape[0]
    felt = np.zeros_like(board, dtype=bool)
    for y, x in positions[: t + 1]:        # union of all 3x3 patches touched so far
        felt[max(0, y - p): y + p + 1, max(0, x - p): x + p + 1] = True
    return board.astype(int) + 2 * felt   # 0/1 = not felt, 2/3 = felt


def _draw_board(ax, board, positions, t, k):
    """Board with hidden shape, felt cells, finger path and the current 3x3 patch."""
    ax.imshow(_felt(board, positions, t, k), cmap=BOARD_CMAP, vmin=0, vmax=3)
    ys, xs = positions[: t + 1, 0], positions[: t + 1, 1]
    ax.plot(xs, ys, color=FINGER, linewidth=1, alpha=0.6)            # path so far (x = column, y = row)
    y, x = positions[t]
    ax.add_patch(Rectangle((x - k // 2 - 0.5, y - k // 2 - 0.5), k, k,
                           fill=False, edgecolor=FINGER, linewidth=2))  # current touch patch
    ax.plot(x, y, "o", color=FINGER, markersize=5)                    # finger
    n = board.shape[0]
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True); ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#D3D1C7", linewidth=0.4)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(n - 0.5, -0.5)           # row 0 on top, like the arrays


def _draw_guess(ax, probs_t, label, t, num_touches):
    """Bar chart of the shape probabilities after touch t; correct shape in teal."""
    colors = ["#1D9E75" if i == label else "#B4B2A9" for i in range(len(SHAPE_NAMES))]
    ax.bar(SHAPE_NAMES, probs_t, color=colors)
    ax.set_ylim(0, 1); ax.set_ylabel("probability")
    guess = SHAPE_NAMES[int(probs_t.argmax())]
    ax.set_title(f"touch {t + 1}/{num_touches}: guess {guess} ({probs_t.max():.0%})", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def show_game(agent, gif_name="game.gif", png_name="game_frames.png", frames=(1, 4, 8, 16)):
    """Play one game and save it twice: as an animation and as a still with a few touches."""
    world, positions, probs = play(agent, 1)
    board, label, k = world.boards[0], world.labels[0], world.patch_size
    pos, pr, T = positions[:, 0], probs[:, 0], len(positions)
    legend = [Patch(color=c, label=l) for c, l in
              zip(BOARD_CMAP.colors[1:], ["shape (hidden)", "felt: empty", "felt: shape"])]

    # animation: one frame per touch
    fig, (ax_b, ax_g) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [1.2, 1]})
    def update(t):
        ax_b.clear(); ax_g.clear()
        _draw_board(ax_b, board, pos, t, k)
        ax_b.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
                    frameon=False, fontsize=8)
        _draw_guess(ax_g, pr[t], label, t, T)
    fig.suptitle(f"true shape: {SHAPE_NAMES[label]}", fontsize=11)
    fig.tight_layout()
    anim = FuncAnimation(fig, update, frames=T, interval=500)   # 500 ms per touch in the window
    anim.save(_path(gif_name), writer=PillowWriter(fps=2))
    _animations.append(anim)

    # still image: selected touches side by side, board on top, guess below
    fig, axes = plt.subplots(2, len(frames), figsize=(3 * len(frames), 6),
                             gridspec_kw={"height_ratios": [1.2, 1]})
    for col, touch in enumerate(frames):
        _draw_board(axes[0, col], board, pos, touch - 1, k)
        _draw_guess(axes[1, col], pr[touch - 1], label, touch - 1, T)
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"true shape: {SHAPE_NAMES[label]}", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(_path(png_name), dpi=150)


def show():
    """Open all plots in windows. Blocks until you close them."""
    plt.show()


if __name__ == "__main__":
    # redo the plots without training again: needs agent.pt saved by main.py
    agent = Agent()
    agent.load_state_dict(torch.load("agent.pt"))
    plot_accuracy(agent)
    show_game(agent)
    show()
