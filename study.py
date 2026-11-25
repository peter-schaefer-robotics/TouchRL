"""Sensitivity study for TouchRL.

Trains every setting defined in sensitivity.toml, evaluates each trained agent on fresh games and
compares the settings of each group in a table (results/<group>.csv) and a figure (results/<group>.png).
At the end, results/coverage.png puts all noise-free board and fingertip settings on one axis:
how much of the board the finger can feel at most (touches x patch area / board area).

    uv run study.py                      all groups
    uv run study.py fingertip memory     only these groups

Each finished run is saved in results/runs/ (its trained agent in results/agents/) and skipped next
time. So an interrupted study (Ctrl+C) can simply be started again, and a setting shared by several
groups is trained only once per seed. Each group's table and figure are written as soon as all its
runs are done.
"""
import csv
import hashlib
import itertools
import json
import math
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
AGENTS = RESULTS / "agents"
WORLD_KEYS = ("grid_size", "patch_size", "step_size", "max_touch_num", "noise")
TRAIN_KEYS = ("num_rounds", "gamma", "entropy_coef", "lr", "reward")
GROUP_KEYS = ("description", "seeds", "variants")                # special keys in a group, not parameters
OPTIONAL_KEYS = ("train_noise",)                                  # parameters that are not in the baseline (see sensitivity.toml)
NAN = float("nan")


def load_config(path=CONFIG):
    """Read the study settings."""
    with open(path, "rb") as f:                                   # tomllib wants the file in binary mode
        return tomllib.load(f)


def plan(config, groups=None):
    """Per group: what varies (x-axis first, then lines), its seeds, and all settings.

    Inside a group, a list varies a parameter, a single value fixes it for this group only, and
    [groups.<name>.variants] adds named bundles of values. Each setting is (label, params):
    label = the varied values (for tables and plots), params = everything the run needs.
    """
    baseline = config["baseline"]
    available = list(config["groups"])
    names = groups or available
    unknown = [g for g in names if g not in available]
    if unknown:
        raise ValueError(f"unknown group(s) {unknown}, available: {available}")
    plans = {}
    for name in names:
        group = config["groups"][name]
        fixed = {k: v for k, v in group.items() if k not in GROUP_KEYS and not isinstance(v, list)}
        varied = {k: v for k, v in group.items() if k not in GROUP_KEYS and isinstance(v, list)}
        variants = {variant: {k: v for k, v in bundle.items() if k != "description"}   # a variant may describe itself
                    for variant, bundle in group.get("variants", {}).items()}
        for values in (fixed, varied, *variants.values()):
            typos = [k for k in values if k not in baseline and k not in OPTIONAL_KEYS]
            if typos:
                raise ValueError(f"group '{name}': {typos} not in [baseline]")
        axes = list(varied) + (["variant"] if variants else [])
        if not 1 <= len(axes) <= 2:
            raise ValueError(f"group '{name}': vary one or two things (x-axis and lines), not {axes}")
        settings = []
        for values in itertools.product(*varied.values()):
            combo = dict(zip(varied, values))
            for variant, overrides in (variants.items() or [(None, {})]):
                label = {**combo, **({"variant": variant} if variant else {})}
                params = {**baseline, **fixed, **overrides, **combo}
                if params.get("train_noise") == params["noise"]:  # same noise in training and test: a normal run
                    del params["train_noise"]                     # (same file name, so existing runs are reused)
                settings.append((label, params))
        plans[name] = {"axes": axes, "settings": settings, "description": group.get("description", ""),
                       "seeds": group.get("seeds", config["study"]["seeds"])}
    return plans


def run_key(params, seed):
    """Short file name for one run; identical settings get the same name, in any group."""
    text = json.dumps({**params, "seed": seed}, sort_keys=True)
    return hashlib.md5(text.encode()).hexdigest()[:10]


@torch.no_grad()                                                  # evaluation only, no learning
def evaluate(agent, world, sure=0.9):
    """Play fresh games with the trained agent (moves sampled as in training) and measure:

    acc_per_touch      share of right guesses after each touch
    acc_last           share of right guesses after the last touch
    macro_f1           F1 score per shape after the last touch, averaged over the shapes
    confusion          after the last touch: rows = true shape, columns = guessed shape (counts)
    sure_by_touch      per touch k: share of games that are right and at least 90 % sure for good by touch k
    touches_if_solved  average touch from which that holds, only over the games that get there
    solved_share       share of games that get there at all
    touches_to_sure    like touches_if_solved, but games that never get there count as touches + 1
    found_by_touch     per touch k: share of games in which the finger has touched the shape by touch k (search)
    found_share        share of games in which the finger touched the shape at all
    acc_if_found       share of right guesses after the last touch, only over those games (reading)
    """
    patches = world.reset()
    labels = torch.from_numpy(world.labels)
    h = torch.zeros(world.num_games, agent.hidden_size)
    probs, found, found_by_touch = [], np.zeros(world.num_games, dtype=bool), []
    for t in range(world.max_touch_num):
        found |= world.touches_shape()                            # has the finger touched the shape so far?
        found_by_touch.append(found.mean())
        guess_logits, move_logits, h = agent(make_input(world, patches), h)
        probs.append(torch.softmax(guess_logits, dim=1))
        if t < world.max_touch_num - 1:
            patches = world.step(torch.distributions.Categorical(logits=move_logits).sample().numpy())
    probs = torch.stack(probs)                                    # (touches, games, shapes)
    T, K = probs.shape[0], probs.shape[2]
    guesses = probs.argmax(dim=2)
    correct = guesses == labels                                   # (touches, games)
    right_and_sure = correct & (probs.max(dim=2).values >= sure)
    n_end = right_and_sure.int().flip(0).cumprod(0).flip(0).sum(0)   # per game: last touches in a row that are right and sure
    solved = n_end > 0
    first = torch.where(solved, T - n_end + 1, T + 1)            # first touch from which right and sure for good

    confusion = torch.zeros(K, K, dtype=torch.long)
    confusion.index_put_((labels, guesses[-1]), torch.ones_like(labels), accumulate=True)
    hits = confusion.diag().float()
    precision = hits / confusion.sum(dim=0).clamp(min=1)         # of all guesses "circle", how many were circles
    recall = hits / confusion.sum(dim=1).clamp(min=1)            # of all circles, how many were guessed as circle
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
    return {"acc_per_touch": correct.float().mean(dim=1).tolist(),
            "acc_last": correct[-1].float().mean().item(),
            "macro_f1": f1.mean().item(),
            "confusion": confusion.tolist(),
            "sure_by_touch": [(first <= k).float().mean().item() for k in range(1, T + 1)],
            "touches_if_solved": first[solved].float().mean().item() if solved.any() else NAN,
            "solved_share": solved.float().mean().item(),
            "touches_to_sure": first.float().mean().item(),
            "found_by_touch": [float(v) for v in found_by_touch],
            "found_share": float(found.mean()),
            "acc_if_found": correct[-1][torch.from_numpy(found)].float().mean().item() if found.any() else NAN}


def run_one(params, seed, eval_games, threads, agent_path=None):
    """Train and evaluate one setting with one seed. Runs in its own process.
    A run that breaks down (for example with a too high learning rate) is recorded with its error
    instead of stopping the whole study."""
    torch.set_num_threads(threads)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)   # same seed = same run, reproducible
    world_args = {k: params[k] for k in WORLD_KEYS}               # the world the agent is tested in
    train_args = {**world_args, "noise": params.get("train_noise", params["noise"])}   # usually the same
    start = time.time()
    history = []
    try:
        agent = Agent(input_size=params["patch_size"] ** 2 + 3,    # patch values + position (2) + touch count
                      hidden_size=params["hidden_size"], memory=params["memory"])
        history = train(agent, World(params["num_games"], **train_args),
                        **{k: params[k] for k in TRAIN_KEYS}, verbose=False)
        result = evaluate(agent, World(eval_games, **world_args))
        if agent_path:
            torch.save(agent.state_dict(), agent_path)            # keep the trained agent for GIFs and later analysis
    except Exception as error:
        result = {"error": f"{type(error).__name__}: {error}", "acc_last": NAN, "touches_to_sure": NAN}
    result.update(params=params, seed=seed, seconds=time.time() - start,
                  history=history[::10])                           # every 10th round is enough for the curves
    return result


def run_study(groups=None, config_path=CONFIG):
    """Run the whole study (groups=None) or only the named groups. Writes each group's table and figure
    as soon as its runs are done, and the coverage figure at the end."""
    config = load_config(config_path)
    study, baseline = config["study"], config["baseline"]
    plans = plan(config, groups)
    RUNS.mkdir(parents=True, exist_ok=True)
    save_agents = study.get("save_agents", True)
    if save_agents:
        AGENTS.mkdir(parents=True, exist_ok=True)

    todo, group_keys = {}, {}                                     # runs without a result file yet, each only once
    for name, p in plans.items():
        group_keys[name] = set()
        for _, params in p["settings"]:
            for seed in p["seeds"]:
                key = run_key(params, seed)
                group_keys[name].add(key)
                if not (RUNS / f"{key}.json").exists():
                    todo[key] = (params, seed)
    n_settings = sum(len(p["settings"]) for p in plans.values())
    print(f"{len(plans)} group(s), {n_settings} settings: {len(todo)} runs to train")

    finished_groups = set()

    def summarize_ready_groups():
        """Write table and figure of every group whose runs are all done, as soon as that is the case."""
        for name, keys in group_keys.items():
            if name not in finished_groups and all((RUNS / f"{k}.json").exists() for k in keys):
                summarize(name, plans[name])
                finished_groups.add(name)
                print(f"  group '{name}' complete: {RESULTS / name}.png and .csv")

    summarize_ready_groups()                                      # groups already finished in an earlier start
    if todo:
        workers = study["workers"]
        threads = 1 if workers > 1 else torch.get_num_threads()  # parallel runs: 1 thread each, so they don't slow each other down
        start = time.time()
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
        futures = {pool.submit(run_one, params, seed, study["eval_games"], threads,
                               str(AGENTS / f"{key}.pt") if save_agents else None): key
                   for key, (params, seed) in todo.items()}       # submitted group by group, so groups finish one after another
        try:
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                (RUNS / f"{futures[future]}.json").write_text(json.dumps(result))   # saved right away: safe to interrupt
                changed = {k: v for k, v in result["params"].items() if v != baseline.get(k)} or "baseline"
                elapsed = time.time() - start
                left = elapsed / i * (len(todo) - i)                # rough: average pace so far
                outcome = (f"FAILED ({result['error'][:60]})" if "error" in result else
                           f"accuracy {result['acc_last']:.2f}, F1 {result['macro_f1']:.2f}, "
                           f"right and sure from touch {result['touches_to_sure']:.1f}")
                print(f"[{i}/{len(todo)}] {changed}, seed {result['seed']}: {outcome}  "
                      f"({elapsed / 60:.0f} min, about {left / 60:.0f} min left)")
                summarize_ready_groups()
        except KeyboardInterrupt:                                 # Ctrl+C: stop the running trainings right away
            if hasattr(pool, "terminate_workers"):                # Python 3.14+
                pool.terminate_workers()
            else:
                for process in list((pool._processes or {}).values()):
                    process.terminate()
                pool.shutdown(wait=False, cancel_futures=True)
            print("\nstopped: finished runs are saved, start the same command again to continue")
            return
        pool.shutdown()

    plot_coverage(config)
    print(f"tables and figures in {RESULTS}/")


def _value(run, metric, common):
    """One number per run; NaN if the run failed or was made before this metric existed."""
    if metric == "acc_at_common":                                 # accuracy after a touch that every budget in the group reaches
        return run["acc_per_touch"][common - 1] if "acc_per_touch" in run else NAN
    if metric == "sure_by_common":
        return run["sure_by_touch"][common - 1] if "sure_by_touch" in run else NAN
    value = run.get(metric, NAN)
    return NAN if value is None else value


def _mean_std(values):
    """Mean and spread over the seeds, ignoring missing values."""
    values = [v for v in values if not math.isnan(v)]
    return (float(np.mean(values)), float(np.std(values))) if values else (NAN, NAN)


METRICS = ("acc_last", "acc_at_common", "found_share", "acc_if_found", "macro_f1", "touches_to_sure",
           "touches_if_solved", "solved_share", "sure_by_common")


def summarize(name, p):
    """Mean and spread over the seeds for each setting of a group -> results/<group>.csv and .png"""
    rows = [(label, [json.loads((RUNS / f"{run_key(params, s)}.json").read_text()) for s in p["seeds"]])
            for label, params in p["settings"]]
    common = min(params["max_touch_num"] for _, params in p["settings"])
    names = [m.replace("common", f"touch_{common}") for m in METRICS]
    with open(RESULTS / f"{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(p["axes"] + [f"{m}_{s}" for m in names for s in ("mean", "std")] + ["failed", "minutes"])
        for label, results in rows:
            values = []
            for m in METRICS:
                mean, std = _mean_std([_value(r, m, common) for r in results])
                values += [round(mean, 4), round(std, 4)]
            failed = sum("error" in r for r in results)
            minutes = round(float(np.mean([r["seconds"] for r in results])) / 60, 1)
            writer.writerow([label[k] for k in p["axes"]] + values + [failed, minutes])
    plot_group(name, p, rows, common)


def _smooth(values, window=10):
    """Moving average over the last `window` points (fewer at the start), same length as the input."""
    values = np.asarray(values, dtype=float)
    sums = np.concatenate([[0.0], np.cumsum(values)])
    starts = np.maximum(np.arange(len(values)) - window + 1, 0)
    return (sums[1:] - sums[starts]) / (np.arange(len(values)) + 1 - starts)


def plot_group(name, p, rows, common):
    """Four panels: final accuracy and a speed measure per setting (mean and spread over seeds),
    training curves and accuracy per touch (mean over seeds)."""
    import matplotlib.pyplot as plt                               # imported here, so the training processes stay light
    x_key = p["axes"][0]
    line_key = p["axes"][1] if len(p["axes"]) > 1 else None
    x_values = list(dict.fromkeys(label[x_key] for label, _ in rows))   # unique values, in config order
    line_values = list(dict.fromkeys(label[line_key] for label, _ in rows)) if line_key else [None]
    budgets_differ = len({params["max_touch_num"] for _, params in p["settings"]}) > 1
    right = (("acc_at_common", f"accuracy after touch {common}\n(reached by every touch budget)") if budgets_differ
             else ("touches_to_sure", "touch from which right and sure\n(never = touches + 1)"))
    colors = plt.cm.tab10.colors
    chance = 1 / len(SHAPES)
    named = any(isinstance(v, str) for v in x_values)            # names on the x-axis (e.g. agents): no lines between them
    failed = dict.fromkeys(x_values, 0)                         # broken runs per x value, shown under the tick
    for label, results in rows:
        failed[label[x_key]] += sum("error" in r for r in results)
    ticks = [f"{v}\n({failed[v]} failed)" if failed[v] else str(v) for v in x_values]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, metric, ylabel in ((axes[0, 0], "acc_last", "accuracy after last touch"), (axes[0, 1], *right)):
        for j, value in enumerate(line_values):
            chosen = [(label, res) for label, res in rows if line_key is None or label[line_key] == value]
            xs = [x_values.index(label[x_key]) for label, _ in chosen]
            stats = [_mean_std([_value(r, metric, common) for r in res]) for _, res in chosen]
            ax.errorbar(xs, [m for m, _ in stats], yerr=[s for _, s in stats], marker="o", capsize=3,
                        linestyle="none" if named else "-",
                        color=colors[j % 10], label=f"{line_key} = {value}" if line_key else None)
            if metric == "acc_last":                              # next to it: did the finger find the shape at all?
                found = [_mean_std([_value(r, "found_share", common) for r in res])[0] for _, res in chosen]
                if not all(math.isnan(v) for v in found):         # runs from before this measure have none
                    ax.plot(xs, found, marker="x", linestyle="none" if named else "--", color=colors[j % 10])
                    ylabel = "accuracy after last touch (o, solid)\nshape found at all (x, dashed)"
        crowded = sum(len(t) for t in ticks) > 40                 # long names on the x-axis: tilt them
        ax.set_xticks(range(len(x_values)), ticks, rotation=30 if crowded else 0, ha="right" if crowded else "center")
        ax.set(xlabel=x_key, ylabel=ylabel)
    if line_key:
        axes[0, 0].legend(frameon=False, fontsize=8)              # same lines in both top panels: one legend is enough
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axhline(chance, color="gray", linestyle="--", linewidth=1)
    if budgets_differ:
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].axhline(chance, color="gray", linestyle="--", linewidth=1)

    for k, (label, results) in enumerate(rows):                   # one line per setting, failed runs left out
        ok = [r for r in results if "error" not in r]
        if not ok:
            continue
        text = ", ".join(f"{key}={label[key]}" for key in p["axes"])
        rounds = [h[0] for h in ok[0]["history"]]
        train_acc = _smooth(np.mean([[h[1] for h in r["history"]] for r in ok], axis=0))
        axes[1, 0].plot(rounds, train_acc, color=colors[k % 10], label=text)
        acc = np.mean([r["acc_per_touch"] for r in ok], axis=0)
        axes[1, 1].plot(range(1, len(acc) + 1), acc, color=colors[k % 10], marker=".", label=text)
    axes[1, 0].set(xlabel="training round", ylabel="training accuracy (moving average)", ylim=(0, 1))
    axes[1, 1].set(xlabel="number of touches", ylabel="accuracy on fresh games", ylim=(0, 1))
    axes[1, 1].legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))   # outside: never covers a curve
    for ax in axes[1]:
        ax.axhline(chance, color="gray", linestyle="--", linewidth=1)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"{name}: {p['description']}  (error bars: spread over seeds)")
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}.png", dpi=150)
    plt.close(fig)


def plot_coverage(config):
    """Accuracy against the coverage budget = touches x patch area / board area, i.e. how much of the board
    the finger can feel at most if it never touches a cell twice. Uses every noise-free run that differs
    from the baseline only in board size, patch size, step size or touch budget (mean over seeds)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    baseline = config["baseline"]
    other = [k for k in baseline if k not in WORLD_KEYS]         # agent and training settings must be the baseline
    points = {}
    for path in RUNS.glob("*.json"):
        run = json.loads(path.read_text())
        params = run["params"]
        if ("error" in run or params.get("noise", 0) != 0 or "train_noise" in params
                or any(params.get(k) != baseline[k] for k in other)):
            continue
        geometry = (params["grid_size"], params["patch_size"], params["step_size"], params["max_touch_num"])
        points.setdefault(geometry, []).append(run["acc_last"])
    if not points:
        return
    colors = {1: "#D85A30", 3: "#1D9E75", 5: "#534AB7"}
    markers = {12: "o", 16: "s", 24: "^", 32: "D"}
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for (grid, patch, step, touches), accs in sorted(points.items()):
        ax.scatter(touches * patch ** 2 / grid ** 2, np.mean(accs), s=45,
                   color=colors.get(patch, "gray"), marker=markers.get(grid, "o"))
    handles = ([Line2D([], [], color=c, marker="o", linestyle="", label=f"patch {p}") for p, c in colors.items()] +
               [Line2D([], [], color="gray", marker=m, linestyle="", label=f"board {g}x{g}") for g, m in markers.items()])
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.axhline(1 / len(SHAPES), color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set(xlabel="coverage budget = touches x patch area / board area (log scale)",
           ylabel="accuracy after last touch", ylim=(0.3, 1))
    ax.set_title("Accuracy vs. coverage budget (noise-free runs with baseline training)\n"
                 "points above each other: same coverage, different step size", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "coverage.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_study(sys.argv[1:] or None)
