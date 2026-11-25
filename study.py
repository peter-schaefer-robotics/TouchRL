"""Sensitivity study for TouchRL.

Trains every setting defined in sensitivity.toml, evaluates each trained agent on fresh games and
compares the settings of each group in a table (results/<group>.csv) and a figure (results/<group>.png).

    python study.py                      all groups
    python study.py fingertip memory     only these groups

Each finished run is saved in results/runs/ and skipped next time. So an interrupted study can simply
be started again, and a setting shared by several groups (the baseline) is trained only once per seed.
"""
import csv
import hashlib
import itertools
import json
import multiprocessing
import random
import sys
import time
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from agent import Agent
from shapes import SHAPES
from train import make_input, train
from world import World

CONFIG = Path(__file__).with_name("sensitivity.toml")
RESULTS = Path(__file__).with_name("results")
RUNS = RESULTS / "runs"
WORLD_KEYS = ("grid_size", "patch_size", "step_size", "max_touch_num", "noise")
TRAIN_KEYS = ("num_rounds", "gamma", "entropy_coef", "lr", "reward")
METRICS = ("acc_last", "touches_to_sure", "solved_share")


def load_config(path=CONFIG):
    """Read the study settings."""
    with open(path, "rb") as f:                                   # tomllib wants the file in binary mode
        return tomllib.load(f)


def varied_params(config, group):
    """The parameters a group varies, in file order (first = x-axis, second = one line per value)."""
    return {k: v for k, v in config["groups"][group].items() if k != "description"}


def plan(config, groups=None):
    """Settings per group: every combination of the group's values, all other parameters from the baseline."""
    available = list(config["groups"])
    names = groups or available
    unknown = [g for g in names if g not in available]
    if unknown:
        raise ValueError(f"unknown group(s) {unknown}, available: {available}")
    settings = {}
    for name in names:
        varied = varied_params(config, name)
        typos = [k for k in varied if k not in config["baseline"]]
        if typos:
            raise ValueError(f"group '{name}': {typos} not in [baseline]")
        settings[name] = [{**config["baseline"], **dict(zip(varied, values))}
                          for values in itertools.product(*varied.values())]
    return settings


def run_key(params, seed):
    """Short file name for one run; identical settings get the same name, in any group."""
    text = json.dumps({**params, "seed": seed}, sort_keys=True)
    return hashlib.md5(text.encode()).hexdigest()[:10]


@torch.no_grad()                                                  # evaluation only, no learning
def evaluate(agent, world, sure=0.9):
    """Play fresh games with the trained agent (moves sampled as in training) and measure:
    acc_per_touch: share of right guesses after each touch
    acc_last: share of right guesses after the last touch
    touches_to_sure: touch from which the guess is right and at least 90 % sure for good (never = touches + 1)
    solved_share: share of games that get there at all
    """
    patches = world.reset()
    labels = torch.from_numpy(world.labels)
    h = torch.zeros(world.num_games, agent.hidden_size)
    probs = []
    for t in range(world.max_touch_num):
        guess_logits, move_logits, h = agent(make_input(world, patches), h)
        probs.append(torch.softmax(guess_logits, dim=1))
        if t < world.max_touch_num - 1:
            patches = world.step(torch.distributions.Categorical(logits=move_logits).sample().numpy())
    probs = torch.stack(probs)                                    # (touches, games, shapes)
    T = probs.shape[0]
    correct = probs.argmax(dim=2) == labels                       # (touches, games)
    right_and_sure = correct & (probs.max(dim=2).values >= sure)
    n_end = right_and_sure.int().flip(0).cumprod(0).flip(0).sum(0)   # per game: last touches in a row that are right and sure
    touches = torch.where(n_end > 0, T - n_end + 1, T + 1).float()
    return {"acc_per_touch": correct.float().mean(dim=1).tolist(),
            "acc_last": correct[-1].float().mean().item(),
            "touches_to_sure": touches.mean().item(),
            "solved_share": (n_end > 0).float().mean().item()}


def run_one(params, seed, eval_games, threads):
    """Train and evaluate one setting with one seed. Runs in its own process."""
    torch.set_num_threads(threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)   # same seed = same run, reproducible
    world_args = {k: params[k] for k in WORLD_KEYS}
    agent = Agent(input_size=params["patch_size"] ** 2 + 3,        # patch values + position (2) + touch count
                  hidden_size=params["hidden_size"], memory=params["memory"])
    start = time.time()
    history = train(agent, World(params["num_games"], **world_args),
                    **{k: params[k] for k in TRAIN_KEYS}, verbose=False)
    result = evaluate(agent, World(eval_games, **world_args))
    result.update(params=params, seed=seed, seconds=time.time() - start,
                  history=history[::10])                           # every 10th round is enough for the curves
    return result


def run_study(groups=None, config_path=CONFIG):
    """Run the whole study (groups=None) or only the named groups, then write a table and a figure per group."""
    config = load_config(config_path)
    study, baseline = config["study"], config["baseline"]
    settings = plan(config, groups)
    RUNS.mkdir(parents=True, exist_ok=True)

    todo = {}                                                     # runs without a result file yet, each only once
    for params_list in settings.values():
        for params in params_list:
            for seed in study["seeds"]:
                key = run_key(params, seed)
                if not (RUNS / f"{key}.json").exists():
                    todo[key] = (params, seed)
    n_settings = sum(len(p) for p in settings.values())
    print(f"{len(settings)} group(s), {n_settings} settings, {len(study['seeds'])} seeds each: {len(todo)} runs to train")

    if todo:
        workers = study["workers"]
        threads = 1 if workers > 1 else torch.get_num_threads()  # parallel runs: 1 thread each, so they don't slow each other down
        start = time.time()
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(run_one, params, seed, study["eval_games"], threads): key
                       for key, (params, seed) in todo.items()}
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                (RUNS / f"{futures[future]}.json").write_text(json.dumps(result))   # saved right away: safe to interrupt
                changed = {k: v for k, v in result["params"].items() if v != baseline[k]} or "baseline"
                print(f"[{i}/{len(todo)}] {changed}, seed {result['seed']}: accuracy {result['acc_last']:.2f}, "
                      f"right and sure from touch {result['touches_to_sure']:.1f}  ({(time.time() - start) / 60:.0f} min)")

    for name, params_list in settings.items():
        summarize(name, config, params_list)
    print(f"tables and figures in {RESULTS}/")


def summarize(name, config, params_list):
    """Mean and spread over the seeds for each setting of a group -> results/<group>.csv and .png"""
    varied = list(varied_params(config, name))
    rows = [(params, [json.loads((RUNS / f"{run_key(params, s)}.json").read_text())
                      for s in config["study"]["seeds"]])
            for params in params_list]
    with open(RESULTS / f"{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(varied + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")] + ["minutes"])
        for params, results in rows:
            values = []
            for m in METRICS:
                x = [r[m] for r in results]
                values += [round(float(np.mean(x)), 4), round(float(np.std(x)), 4)]
            minutes = round(float(np.mean([r["seconds"] for r in results])) / 60, 1)
            writer.writerow([params[k] for k in varied] + values + [minutes])
    plot_group(name, config["groups"][name].get("description", ""), varied, rows)


def _smooth(values, window=10):
    """Moving average over `window` points; returns the values and how many points were cut at the start."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values, 0
    return np.convolve(values, np.ones(window) / window, mode="valid"), window - 1


def plot_group(name, description, varied, rows):
    """Four panels: final accuracy and touches to sure per setting (mean and spread over seeds),
    training curves and accuracy per touch (mean over seeds)."""
    import matplotlib.pyplot as plt                               # imported here, so the training processes stay light
    x_key = varied[0]
    line_key = varied[1] if len(varied) > 1 else None
    x_values = list(dict.fromkeys(p[x_key] for p, _ in rows))    # unique values, in config order
    line_values = list(dict.fromkeys(p[line_key] for p, _ in rows)) if line_key else [None]
    colors = plt.cm.tab10.colors
    chance = 1 / len(SHAPES)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, ylabel in ((axes[0, 0], "acc_last", "accuracy after last touch"),
                               (axes[0, 1], "touches_to_sure", "touch from which right and sure")):
        for j, value in enumerate(line_values):
            chosen = [(p, res) for p, res in rows if line_key is None or p[line_key] == value]
            xs = [x_values.index(p[x_key]) for p, _ in chosen]
            means = [np.mean([r[metric] for r in res]) for _, res in chosen]
            spread = [np.std([r[metric] for r in res]) for _, res in chosen]
            ax.errorbar(xs, means, yerr=spread, marker="o", capsize=3, color=colors[j % 10],
                        label=f"{line_key} = {value}" if line_key else None)
        ax.set_xticks(range(len(x_values)), [str(v) for v in x_values])
        ax.set(xlabel=x_key, ylabel=ylabel)
        if line_key:
            ax.legend(frameon=False, fontsize=8)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axhline(chance, color="gray", linestyle="--", linewidth=1)

    for k, (params, results) in enumerate(rows):                  # one line per setting
        label = ", ".join(f"{key}={params[key]}" for key in varied)
        rounds = [h[0] for h in results[0]["history"]]
        train_acc, cut = _smooth(np.mean([[h[1] for h in r["history"]] for r in results], axis=0))
        axes[1, 0].plot(rounds[cut:], train_acc, color=colors[k % 10], label=label)
        acc = np.mean([r["acc_per_touch"] for r in results], axis=0)
        axes[1, 1].plot(range(1, len(acc) + 1), acc, color=colors[k % 10], marker=".", label=label)
    axes[1, 0].set(xlabel="training round", ylabel="training accuracy (moving average)", ylim=(0, 1))
    axes[1, 1].set(xlabel="number of touches", ylabel="accuracy on fresh games", ylim=(0, 1))
    axes[1, 1].legend(frameon=False, fontsize=7)
    for ax in axes[1]:
        ax.axhline(chance, color="gray", linestyle="--", linewidth=1)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{name}: {description}  (error bars: spread over seeds)")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_study(sys.argv[1:] or None)
