"""Plots and animations for TouchRL: training curve, accuracy per touch, and one game step by step."""
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
from matplotlib.widgets import Button
import numpy as np
from PIL import Image
import torch

from agent import Agent
from shapes import SHAPES
from train import make_input
from world import World

SHAPE_NAMES = list(SHAPES)                 # ["circle", "rectangle", "triangle"], index = label
FIG_DIR = "play_output"                    # figures of main.py and play.py (not in git)
# board colors: 0 empty (unknown), 1 shape (hidden), 2 felt empty, 3 felt shape
BOARD_CMAP = ListedColormap(["#FFFFFF", "#E3E1DA", "#F5C4B3", "#1D9E75"])
FINGER = "#D85A30"
TEAL = "#1D9E75"
SURE = 0.9                                 # "sure": at least 90 % on the right shape
LEGEND = [Patch(color=c, label=l) for c, l in
          zip(BOARD_CMAP.colors[1:], ["shape (hidden)", "felt: empty", "felt: shape"])]
_games = []                                # open game windows: kept, otherwise Python deletes their button and animation


def _path(name):
    """Full path in FIG_DIR; creates the folder if needed."""
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, name)


@torch.no_grad()                           # no training here: skip gradient bookkeeping, faster
def play(agent, num_games, random_moves=False, world_args=None, greedy=False):
    """Play num_games games without learning.

    In:  world_args: board and fingertip settings, e.g. {"grid_size": 32, "noise": 0.1} (default: World defaults)
         greedy: always take the most likely move instead of drawing one (no exploration while playing)
    Out: world (boards, labels), positions per touch (touches, num_games, 2) as (y, x),
         shape probabilities per touch (touches, num_games, num_shapes),
         felt patches per touch (touches, num_games, k, k), as the agent got them (with sensor noise, if any),
         found per touch (touches, num_games): has the finger touched the shape by this touch?
    """
    world = World(num_games, **(world_args or {}))
    patches = world.reset()
    h = torch.zeros(num_games, agent.hidden_size)
    positions, probs, felt, found = [], [], [], []
    touched = np.zeros(num_games, dtype=bool)
    for t in range(world.max_touch_num):
        positions.append(world.positions.copy())   # copy: the world changes this array in place
        felt.append(patches)
        touched = touched | world.touches_shape()
        found.append(touched)
        guess_logits, move_logits, h = agent(make_input(world, patches), h)
        probs.append(torch.softmax(guess_logits, dim=1).numpy())   # scores -> probabilities
        if t < world.max_touch_num - 1:
            if random_moves:               # for comparison: same guessing, but random moves
                a = torch.randint(0, 9, (num_games,))
            elif greedy:                   # always the most likely move: what the agent learned, without the dice
                a = move_logits.argmax(dim=1)
            else:                          # learned moves, drawn like in training
                a = torch.distributions.Categorical(logits=move_logits).sample()
            patches = world.step(a.numpy())
    return world, np.stack(positions), np.stack(probs), np.stack(felt), np.stack(found)


def accuracy_per_touch(agent, num_games=4096, random_moves=False, world_args=None):
    """Share of correct guesses after each touch, on fresh games. Out: array (touches,)"""
    world, _, probs, _, _ = play(agent, num_games, random_moves, world_args)
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


def plot_touches_to_correct(history, name="touches_to_correct.png", never=17):
    """Per training round: from which touch on the guess is right and at least 90 % sure for good.
    Average over all games; a game that never gets there counts as touches + 1 (never = 17 for 16 touches)."""
    rounds = np.array([h[0] for h in history]); touches = np.array([h[2] for h in history])
    _plot_history(rounds, touches, "#D85A30", "touch from which right and sure", name, (1, never),
                  ref=never, ref_label=f"never ({never})")


def _draw_accuracy(ax, learned, random_, found=None, found_random=None, moves="learned moves"):
    """Accuracy after each touch: learned moves vs random moves.
    found, found_random (optional, dashed): share of games in which the finger has touched the shape so far."""
    touches = np.arange(1, len(learned) + 1)
    right = "right, " if found is not None else ""
    ax.plot(touches, learned, color="#1D9E75", marker="o", markersize=4, label=f"{right}{moves}")
    ax.plot(touches, random_, color="#888780", marker="o", markersize=4, label=f"{right}random moves")
    if found is not None:
        ax.plot(touches, found, color="#1D9E75", linestyle="--", label=f"shape found, {moves}")
        ax.plot(touches, found_random, color="#888780", linestyle="--", label="shape found, random moves")
    ax.axhline(1 / len(SHAPES), color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set(xlabel="number of touches", ylabel="share of games", ylim=(0, 1), xticks=touches)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def plot_accuracy(agent, name="accuracy_per_touch.png", world_args=None):
    """Learned touching vs random touching, same trained guess: does choosing where to touch help?"""
    learned = accuracy_per_touch(agent, world_args=world_args)
    random_ = accuracy_per_touch(agent, random_moves=True, world_args=world_args)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    _draw_accuracy(ax, learned, random_)
    fig.tight_layout(); fig.savefig(_path(name), dpi=150)
    return learned, random_


def plot_results(agent, name="results.png", world_args=None, title=None, num_games=4096, window_title=None,
                 greedy=False):
    """How good is the agent on new worlds? Left: per touch, the share of right guesses (solid) and of games in
    which the finger has touched the shape at all (dashed), learned vs random moves.
    Right: confusion matrix after the last touch, i.e. for each true shape, where its guesses went.
    Out: dict with accuracy (learned and random moves), share of games with the shape found, accuracy in those"""
    world, _, probs, _, found = play(agent, num_games, world_args=world_args, greedy=greedy)
    world_r, _, probs_r, _, found_r = play(agent, num_games, random_moves=True, world_args=world_args)
    correct = probs.argmax(axis=2) == world.labels
    learned = correct.mean(axis=1)
    random_ = (probs_r.argmax(axis=2) == world_r.labels).mean(axis=1)
    right_if_found = correct[-1][found[-1]].mean()
    n = len(SHAPE_NAMES)
    confusion = np.zeros((n, n))
    np.add.at(confusion, (world.labels, probs[-1].argmax(axis=1)), 1)   # count: row = true shape, column = guess
    share = confusion / confusion.sum(axis=1, keepdims=True)           # per true shape, so every row adds up to 100 %

    fig, (ax_a, ax_c) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.7, 1]})
    if window_title:
        fig.canvas.manager.set_window_title(window_title)
    _draw_accuracy(ax_a, learned, random_, found.mean(axis=1), found_r.mean(axis=1),
                   "most likely moves" if greedy else "learned moves")
    ax_a.set_title(f"{num_games} new worlds: shape found in {found[-1].mean():.0%}, "
                   f"right in {right_if_found:.0%} of those", fontsize=10)
    ax_c.imshow(share, cmap=LinearSegmentedColormap.from_list("teal", ["#FFFFFF", "#1D9E75"]), vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax_c.text(j, i, f"{share[i, j]:.0%}", ha="center", va="center",
                      color="white" if share[i, j] > 0.5 else "black")
    ax_c.set(xticks=range(n), xticklabels=SHAPE_NAMES, yticks=range(n), yticklabels=SHAPE_NAMES,
             xlabel="guessed shape", ylabel="true shape")
    ax_c.set_title("true vs. guessed shape after the last touch", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout(); fig.savefig(_path(name), dpi=150)
    return {"accuracy": learned[-1], "accuracy_random_moves": random_[-1],
            "found": found[-1].mean(), "right_if_found": right_if_found}


def _felt(board, positions, felt, t, k):
    """Board codes after touch t (see BOARD_CMAP): cells not felt yet show the hidden board (0/1),
    felt cells show what the finger reported there last (2 = empty, 3 = shape). With sensor noise,
    some reports are wrong, e.g. a shape cell felt outside the shape."""
    p, n = k // 2, board.shape[0]
    codes = board.astype(int)
    for (y, x), patch in zip(positions[: t + 1], felt[: t + 1]):   # every patch touched so far, later ones on top
        for i in range(k):
            for j in range(k):
                r, c = y - p + i, x - p + j                         # board cell under patch cell (i, j)
                if 0 <= r < n and 0 <= c < n:                       # cells outside the board are padding
                    codes[r, c] = 2 + int(patch[i, j])
    return codes


def _draw_board(ax, board, positions, felt, t, k):
    """Board with hidden shape, felt cells, finger path and the current patch."""
    ax.imshow(_felt(board, positions, felt, t, k), cmap=BOARD_CMAP, vmin=0, vmax=3)
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


def _draw_guess(ax, probs_t, label, t, num_touches, sure_touch=None):
    """Bar chart of the shape probabilities after touch t; correct shape in teal.
    Dashed: the 90 % line. The touch where the agent first gets sure (sure_touch) has a teal title."""
    colors = [TEAL if i == label else "#B4B2A9" for i in range(len(SHAPE_NAMES))]
    ax.bar(SHAPE_NAMES, probs_t, color=colors)
    ax.axhline(SURE, color="#888780", linestyle="--", linewidth=0.8)
    ax.text(1.0, SURE, f" {SURE:.0%}", transform=ax.get_yaxis_transform(), va="center", fontsize=8,
            color="#888780")                                        # label at the right end of the line
    ax.set_ylim(0, 1); ax.set_ylabel("probability")
    guess = SHAPE_NAMES[int(probs_t.argmax())]
    first_sure = sure_touch == t + 1
    ax.set_title(f"touch {t + 1}/{num_touches}: guess {guess} ({probs_t.max():.0%})", fontsize=10,
                 color=TEAL if first_sure else "black", fontweight="bold" if first_sure else "normal")
    ax.spines[["top", "right"]].set_visible(False)


def _play_one(agent, world_args=None, greedy=False):
    """One new game, with everything needed to draw it."""
    world, positions, probs, felt, _ = play(agent, 1, world_args=world_args, greedy=greedy)
    label = world.labels[0]
    sure = np.flatnonzero(probs[:, 0, label] >= SURE)            # touches with at least 90 % on the right shape
    return {"board": world.boards[0], "label": label, "k": world.patch_size,
            "pos": positions[:, 0], "probs": probs[:, 0], "felt": felt[:, 0], "T": len(positions),
            "sure": int(sure[0]) + 1 if len(sure) else None}         # first such touch (1, 2, ...), None = never


def _sure_badge(fig, game, t=None):
    """Top right of the figure: "sure at touch 5" (teal) once the agent is sure, "not sure after 16
    touches" (gray) at the end of a game where it never gets there. t = None: the whole game is over."""
    old = getattr(fig, "_sure_badge", None)
    if old is not None and old.figure is not None:                # only one badge: remove the one of the last frame
        old.remove()
    fig._sure_badge = None
    last = game["T"] - 1 if t is None else t
    if game["sure"] is not None and last + 1 >= game["sure"]:
        text, color, back = f"sure at touch {game['sure']}", "white", TEAL
    elif last == game["T"] - 1:
        text, color, back = f"not sure after {game['T']} touches", "#2C2C2A", "#E3E1DA"
    else:
        return
    fig._sure_badge = fig.text(0.98, 0.965, text, ha="right", va="top", fontsize=11, fontweight="bold",
                               color=color, bbox=dict(boxstyle="round,pad=0.3", facecolor=back, edgecolor="none"))


def _draw_touch(fig, ax_board, ax_guess, game, t, heading):
    """One frame of the animation: board and guess after touch t."""
    ax_board.clear(); ax_guess.clear()
    _draw_board(ax_board, game["board"], game["pos"], game["felt"], t, game["k"])
    ax_board.legend(handles=LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
                    frameon=False, fontsize=8)
    _draw_guess(ax_guess, game["probs"][t], game["label"], t, game["T"], game["sure"])
    fig.suptitle(f"{heading}true shape: {SHAPE_NAMES[game['label']]}", fontsize=11)
    _sure_badge(fig, game, t)


def _save_gif(path, game, heading):
    """The game as an animated GIF: 2 touches per second, the last touch stays 2 seconds.
    Drawn on a figure without a window, so the window's button does not end up in the file."""
    fig = Figure(figsize=(8, 4))
    canvas = FigureCanvasAgg(fig)                                 # draws into memory instead of a window
    ax_board, ax_guess = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.2, 1]})
    images = []
    for t in range(game["T"]):
        _draw_touch(fig, ax_board, ax_guess, game, t, heading)
        if t == 0:
            fig.tight_layout()
        canvas.draw()
        images.append(Image.fromarray(np.array(canvas.buffer_rgba())))   # np.array copies: the buffer is reused
    durations = [500] * (game["T"] - 1) + [2000]                  # milliseconds per touch
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0)


def _pick_frames(game):
    """Touches for the snapshots: first, quarter, half and last touch; the touch where the agent gets
    sure replaces the middle one closest to it, so it is always shown."""
    T, sure = game["T"], game["sure"]
    frames = [1, max(1, T // 4), max(1, T // 2), T]
    if sure is not None and sure not in frames:
        closest = min((1, 2), key=lambda i: abs(frames[i] - sure))
        frames[closest] = sure
    return tuple(sorted(set(frames)))


def _draw_frames(fig, game, frames, heading):
    """Still image: selected touches side by side, board on top, guess below."""
    fig.clear()
    fig._sure_badge = None                                        # cleared together with the figure
    axes = fig.subplots(2, len(frames), gridspec_kw={"height_ratios": [1.2, 1]})
    for col, touch in enumerate(frames):
        _draw_board(axes[0, col], game["board"], game["pos"], game["felt"], touch - 1, game["k"])
        _draw_guess(axes[1, col], game["probs"][touch - 1], game["label"], touch - 1, game["T"], game["sure"])
    fig.legend(handles=LEGEND, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"{heading}true shape: {SHAPE_NAMES[game['label']]}", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _sure_badge(fig, game)


class GameWindow:
    """One agent, two windows: the game as an animation and snapshots of a few touches.
    The button "New game" (or the key n) lets the same agent play a new random world.
    Every game is saved as a GIF and a PNG; a new game replaces the files of the one before."""

    def __init__(self, agent, gif_name, png_name, frames=None, world_args=None, title=None, greedy=False):
        self.agent, self.world_args, self.frames, self.greedy = agent, world_args, frames, greedy
        self.gif_name, self.png_name = gif_name, png_name
        self.heading = f"{title}: " if title else ""
        self.game = _play_one(agent, world_args, greedy)

        self.fig = plt.figure(figsize=(8, 4.4))                    # the game
        self.fig_frames = plt.figure(figsize=(3 * len(frames or _pick_frames(self.game)), 6))   # the snapshots
        for fig, kind in ((self.fig, "game"), (self.fig_frames, "snapshots")):
            fig.canvas.manager.set_window_title(f"{title or 'TouchRL'}: {kind}")
        self.ax_board, self.ax_guess = self.fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.2, 1]})
        _draw_touch(self.fig, self.ax_board, self.ax_guess, self.game, 0, self.heading)
        self.fig.tight_layout(rect=(0, 0.08, 1, 1))               # room for the button below
        self.button = Button(self.fig.add_axes([0.76, 0.015, 0.21, 0.07]), "New game (n)")
        self.button.on_clicked(self.new_game)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.anim = FuncAnimation(self.fig, self._draw, frames=self.game["T"], interval=500,
                                  repeat_delay=2000)              # 500 ms per touch, 2 s pause at the end
        self._save()
        _games.append(self)

    def _draw(self, t):
        _draw_touch(self.fig, self.ax_board, self.ax_guess, self.game, t, self.heading)

    def _on_key(self, event):
        if event.key == "n":
            self.new_game()

    def new_game(self, event=None):
        """The same agent plays a new random world."""
        self.game = _play_one(self.agent, self.world_args, self.greedy)
        self.anim.frame_seq = self.anim.new_frame_seq()           # the animation starts again at touch 1
        self._save()
        print(f"new game: true shape {SHAPE_NAMES[self.game['label']]}, saved as {self.gif_name} and {self.png_name}")

    def _save(self):
        """Snapshots window and both files for the current game."""
        _draw_frames(self.fig_frames, self.game, self.frames or _pick_frames(self.game), self.heading)
        self.fig_frames.savefig(_path(self.png_name), dpi=150)
        self.fig_frames.canvas.draw_idle()
        _save_gif(_path(self.gif_name), self.game, self.heading)


def show_game(agent, gif_name="game.gif", png_name="game_frames.png", frames=None, world_args=None, title=None,
              greedy=False):
    """Play one game and save it twice: as an animation (GIF) and as a still with a few touches (PNG).
    In the window, "New game" (or the key n) lets the same agent play a new world.
    frames: touches for the still (default: first, quarter, half, last, and always the touch where the agent
    gets sure); title: e.g. the agent's name
    greedy: always take the most likely move instead of drawing one"""
    return GameWindow(agent, gif_name, png_name, frames, world_args, title, greedy)


def show():
    """Open all plots in windows, the newest game on top. Blocks until you close them."""
    if _games:                                                    # windows open in the order of their last use,
        plt.figure(_games[-1].fig_frames)                         # each one on top of the one before:
        plt.figure(_games[-1].fig)                                # the game comes last, so it lies on top
    plt.show()


if __name__ == "__main__":
    # redo the plots without training again: needs agent.pt saved by main.py
    if not os.path.exists("agent.pt"):
        raise SystemExit("No agent.pt yet: train one with  uv run main.py  (or watch a named agent with  uv run play.py)")
    agent = Agent()
    agent.load_state_dict(torch.load("agent.pt"))
    plot_accuracy(agent)
    show_game(agent)
    show()
