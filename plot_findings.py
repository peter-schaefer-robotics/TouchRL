"""Figures for the findings of the TouchRL sensitivity study: one figure per finding, each with a caption.

Requires a complete study run (results/runs/ and results/agents/, created by `uv run study.py`) and
writes results/findings/<number>_<name>.png.

    uv run plot_findings.py          all figures
    uv run plot_findings.py 3 5      figures 3 and 5 only, shown in a window

Terms: accuracy = share of episodes classified correctly after the last touch; detection rate = share of
episodes in which the fingertip contacted the shape at least once; accuracy given detection = accuracy over
those episodes. Points show the mean over training seeds, error bars one standard deviation. p-values come
from a two-sided Welch's t-test over seeds, not corrected for multiple comparisons. Figures 3 and 5
evaluate the saved agents on new episodes.
"""
import json
import math
import random
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent
from study import WORLD_KEYS, load_config, run_key
from visualize import FINGER, TEAL, play

OUT = Path("results/findings")
BASELINE = load_config()["baseline"]
RUNS = [json.loads(p.read_text()) for p in Path("results/runs").glob("*.json")]
GRAY, LIGHT_TEAL, LIGHT_ORANGE, PURPLE = "#888780", "#9FE1CB", "#F5C4B3", "#534AB7"
SHAPES = ["circle", "rectangle", "triangle"]


def runs(seeds=(0, 1, 2), **changes):
    """The finished runs with these settings (all others at baseline) and these seeds."""
    params = {**BASELINE, **changes}
    if params.get("train_noise") == params["noise"]:          # stored without train_noise (see study.py)
        del params["train_noise"]
    found = [r for r in RUNS if r["params"] == params and r["seed"] in seeds]
    assert found, f"no runs for {changes}: run the study first"
    return found


def values(rs, metric="acc_last"):
    """One value per seed, in percent."""
    return 100 * np.array([r[metric] for r in rs])


def mean_sd(rs, metric="acc_last"):
    v = values(rs, metric)
    return v.mean(), v.std(ddof=1)


def welch_p(a, b):
    """Two-sided p-value of Welch's t-test, from the t density integrated numerically (no scipy needed)."""
    va, vb = np.var(a, ddof=1) / len(a), np.var(b, ddof=1) / len(b)
    if va + vb == 0:
        return float(np.mean(a) == np.mean(b))
    t = abs(np.mean(a) - np.mean(b)) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    x = np.linspace(0, t, 100_001)
    density = (math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)
               * (1 + x ** 2 / df) ** (-(df + 1) / 2))
    return max(0.0, 1 - 2 * np.trapezoid(density, x))


def p_text(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def compare(rs_a, rs_b, metric="acc_last"):
    """Welch's t-test between two sets of runs, as text."""
    return p_text(welch_p(values(rs_a, metric), values(rs_b, metric)))


def figure(title, caption, ncols=1, width=10.5):
    """Empty figure with the finding as title and the caption below the plots."""
    for old, new in ((" %", "\u00a0%"), (" = ", "\u00a0=\u00a0"), ("p < ", "p\u00a0<\u00a0"), ("p ≥ ", "p\u00a0≥\u00a0"),
                     ("p ≤ ", "p\u00a0≤\u00a0")):
        caption = caption.replace(old, new)                       # no line break inside "p = 0.02" or "94.2 %"
    text = textwrap.fill(caption, 125)
    caption_height = 0.21 * (text.count("\n") + 1) + 0.3                 # inches
    height = 4.4 + caption_height
    fig, axes = plt.subplots(1, ncols, figsize=(width, height))
    fig.suptitle(title, x=0.03, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.03, 0.12 / height, text, fontsize=10, va="bottom", linespacing=1.5,
             bbox=dict(facecolor="#F1EFE8", edgecolor="none", boxstyle="round,pad=0.7"))
    fig.subplots_adjust(top=1 - 0.62 / height, bottom=(caption_height + 0.8) / height, left=0.08, right=0.97, wspace=0.3)
    for ax in np.atleast_1d(axes):
        ax.spines[["top", "right"]].set_visible(False)
    return fig, np.atleast_1d(axes)


def line(ax, x, rs_list, metric="acc_last", **style):
    """Means with error bars, one point per set of runs."""
    stats = [mean_sd(rs, metric) for rs in rs_list]
    ax.errorbar(x, [m for m, _ in stats], yerr=[s for _, s in stats], marker="o", capsize=3, **style)


def play_saved(rs, greedy=False, games=2048):
    """Evaluate the saved agents of these runs on new episodes (same seed, same episodes).
    Out, per episode: true class, final prediction, shape contacted at least once, shape area in cells."""
    out = []
    for r in rs:
        p = r["params"]
        agent = Agent(input_size=p["patch_size"] ** 2 + 3, hidden_size=p["hidden_size"], memory=p["memory"])
        agent.load_state_dict(torch.load(Path("results/agents") / f"{run_key(p, r['seed'])}.pt"))
        random.seed(r["seed"]); np.random.seed(r["seed"]); torch.manual_seed(r["seed"])
        world, _, probs, _, found = play(agent, games, world_args={k: p[k] for k in WORLD_KEYS}, greedy=greedy)
        out.append((world.labels, probs[-1].argmax(axis=1), found[-1], world.boards.sum(axis=(1, 2))))
    return [np.concatenate(column) for column in zip(*out)]


def factor_ranges():
    acc = lambda **ch: mean_sd(runs(**ch))[0]
    entropy = lambda c: mean_sd(runs(seeds=(3, 4, 5), entropy_coef=c))[0]
    factors = [  # label, category color, values in the stable range, extreme values
        ("fingertip size (1x1 to 5x5)", TEAL, [acc(patch_size=s) for s in (1, 3, 5)], []),
        ("sensor noise (0 to 20 %)", TEAL, [acc(noise=n) for n in (0.0, 0.05, 0.1, 0.2)], []),
        ("board size (12 to 24)", GRAY, [acc(grid_size=g) for g in (12, 16, 24)], []),
        ("touch budget (8 to 24)", GRAY, [acc(max_touch_num=t) for t in (8, 16, 24)], []),
        ("memory type (none, RNN, GRU)", FINGER, [acc(memory=m) for m in ("none", "rnn", "gru")], []),
        ("entropy coefficient (0 to 0.1; extreme: 0.3 to 1.0)", FINGER, [entropy(c) for c in (0.0, 0.01, 0.1)],
         [entropy(c) for c in (0.3, 0.5, 1.0)]),
        ("learning rate (0.001 to 0.01; extreme: 0.0003, 0.03)", FINGER,
         [acc(lr=lr) for lr in (0.001, 0.003, 0.005, 0.01)], [acc(lr=lr) for lr in (0.0003, 0.03)]),
        ("hidden units (128, 256; extreme: 32)", FINGER, [acc(hidden_size=h) for h in (128, 256)], [acc(hidden_size=32)]),
        ("reward type and discount factor (8 combinations)", FINGER,
         [acc(reward=rw, gamma=g) for rw in ("level", "gain") for g in (0.0, 0.5, 0.95, 0.99)], []),
        ("batch size (128 to 1,024 episodes)", FINGER, [acc(num_games=n) for n in (128, 512, 1024)], []),
    ]
    span = lambda v: max(v) - min(v)
    sensor = [span(f[2]) for f in factors if f[1] == TEAL]
    learning = [span(f[2]) for f in factors if f[1] == FINGER]
    extreme = [span(f[2] + f[3]) for f in factors if f[3]]
    fig, (ax,) = figure("1. Sensor properties dominate accuracy within the stable hyperparameter range",
        "Range of the mean final accuracy over the tested values of each factor, with all other factors at baseline "
        f"(n = 3 seeds per value). Within the stable hyperparameter range (dark bars), the sensor properties change the "
        f"accuracy by {min(sensor):.0f} to {max(sensor):.0f} percentage points (pp), the learning-related factors by only "
        f"{min(learning):.0f} to {max(learning):.0f} pp. Extreme hyperparameter values (light bars) reduce the accuracy by "
        f"{min(extreme):.0f} to {max(extreme):.0f} pp. The 1x1 fingertip had not converged after 1,500 training rounds.")
    for y, (label, color, stable, extremes) in enumerate(reversed(factors)):
        everything = stable + extremes
        if extremes:
            ax.barh(y, span(everything), left=min(everything), color=color, alpha=0.25)
            ax.text(min(everything) - 0.5, y, f"{span(everything):.0f} pp incl. extremes", va="center", ha="right",
                    fontsize=8, color=GRAY)
        ax.barh(y, span(stable), left=min(stable), color=color)
        ax.text(max(everything) + 0.5, y, f"{span(stable):.0f} pp", va="center", fontsize=9)
    ax.set_yticks(range(len(factors)), [f[0] for f in reversed(factors)], fontsize=9)
    ax.axvline(acc(), color="black", linestyle=":", linewidth=1)
    ax.text(acc(), len(factors) - 0.4, " baseline", fontsize=8)
    ax.set(xlabel="final accuracy (%)", xlim=(60, 102))
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (TEAL, GRAY, FINGER)],
              labels=["sensor", "task", "learning"], frameon=False, loc="lower left", fontsize=9)
    fig.subplots_adjust(left=0.36)
    return fig


def error_decomposition():
    noise = (0.0, 0.05, 0.1, 0.2)
    bars, shares = [], []
    for n in noise:
        rs = runs(noise=n)
        error = 100 - values(rs)
        misclassified = values(rs, "found_share") * (100 - values(rs, "acc_if_found")) / 100
        undetected = error - misclassified
        bars.append((undetected.mean(), misclassified.mean()))
        shares.append(100 * (undetected / error).mean())
    fig, (ax,) = figure("2. With increasing sensor noise, errors shift from detection to classification failures",
        "Error rate after 16 touches, decomposed into episodes in which the fingertip never contacted the shape (dark) "
        "and episodes misclassified despite contact (light); n = 3 seeds per noise level. Without noise, detection "
        f"failures account for {shares[0]:.0f} % of all errors. From 10 % noise onward, misclassification after contact "
        f"dominates ({100 - shares[2]:.0f} % of all errors at 10 % noise, {100 - shares[3]:.0f} % at 20 %).")
    for x, ((undetected, misclassified), share) in enumerate(zip(bars, shares)):
        ax.bar(x, undetected, color=TEAL, label="shape never contacted" if x == 0 else None)
        ax.bar(x, misclassified, bottom=undetected, color=LIGHT_TEAL, label="misclassified despite contact" if x == 0 else None)
        ax.text(x, undetected + misclassified + 0.6, f"{100 - share:.0f} % of errors:\nmisclassified", ha="center", fontsize=9)
    ax.set_xticks(range(4), ["0 %", "5 %", "10 %", "20 %"])
    ax.set(xlabel="sensor noise (probability of a flipped cell reading)", ylabel="error rate (%)")
    ax.legend(frameon=False, loc="upper left")
    return fig


def undetected_class_bias():
    labels, guesses, found, areas = play_saved(runs(seeds=range(10)), games=4096)
    missed, wrong = ~found, guesses != labels
    fig, axes = figure("3. Undetected shapes are predominantly classified as triangles, the smallest class",
        "Ten baseline agents (one per seed), each evaluated on 4,096 new episodes. Left: share of episodes in which the "
        "shape was never contacted, by true class. Right: predicted class in these episodes. Triangles have the smallest "
        f"mean area ({areas[labels == 2].mean():.0f} cells, versus {areas[labels == 0].mean():.0f} for circles and "
        f"{areas[labels == 1].mean():.0f} for rectangles) and remain undetected most often, which makes 'triangle' the most "
        f"probable class given no contact ({100 * (labels[missed] == 2).mean():.0f} % of undetected shapes). Accordingly, the "
        f"agent predicts 'triangle' in {100 * (guesses[missed] == 2).mean():.0f} % of these episodes, with an accuracy of "
        f"{100 * (guesses[missed] == labels[missed]).mean():.0f} % (chance level: 33 %). Incorrect 'triangle' predictions "
        f"account for {100 * (wrong & (guesses == 2)).sum() / wrong.sum():.0f} % of all errors.", ncols=2)
    colors = [GRAY, GRAY, FINGER]
    left = [100 * missed[labels == i].mean() for i in range(3)]
    right = [100 * (guesses[missed] == i).mean() for i in range(3)]
    for ax, heights in ((axes[0], left), (axes[1], right)):
        ax.bar(SHAPES, heights, color=colors)
        for x, h in enumerate(heights):
            ax.text(x, h, f"{h:.0f} %", ha="center", va="bottom", fontsize=9)
    axes[0].set(title="shape never contacted", xlabel="true class", ylabel="share of episodes of this class (%)")
    axes[1].set(title="prediction without contact", xlabel="predicted class", ylabel="share of these episodes (%)",
                ylim=(0, 100))
    return fig


def step_size():
    steps = (1, 2, 3)
    seeds = lambda s: range(10) if s == 2 else (0, 1, 2)
    large = {s: runs(seeds=seeds(s), patch_size=3, step_size=s) for s in steps}
    small = {s: runs(seeds=seeds(s), patch_size=1, step_size=s) for s in steps}
    m = lambda rs, metric: mean_sd(rs, metric)[0]
    fig, axes = figure("4. Larger step sizes increase the detection rate",
        "Detection rate, accuracy and accuracy given detection as a function of the step size (cells per move); n = 3 "
        "seeds (step size 2: n = 10). 3x3 fingertip (left): from step size 1 to 3, the detection rate rises from "
        f"{m(large[1], 'found_share'):.1f} % to {m(large[3], 'found_share'):.1f} % and the accuracy from "
        f"{m(large[1], 'acc_last'):.1f} % to {m(large[3], 'acc_last'):.1f} % (step size 2 vs. 3: "
        f"{compare(large[2], large[3], 'found_share')} and {compare(large[2], large[3])}), as larger steps reduce the overlap "
        "between consecutive contacts. 1x1 fingertip (right): the detection rate also rises, but the accuracy given "
        f"detection falls from {m(small[1], 'acc_if_found'):.1f} % to {m(small[3], 'acc_if_found'):.1f} % "
        f"({compare(small[1], small[3], 'acc_if_found')}), presumably because sparsely sampled cells carry less information "
        "about the contour. These agents had not converged after 1,500 training rounds.", ncols=2)
    for ax, size, runs_by_step, second in ((axes[0], 3, large, ("acc_last", "accuracy")),
                                          (axes[1], 1, small, ("acc_if_found", "accuracy given detection"))):
        rs = [runs_by_step[s] for s in steps]
        line(ax, steps, rs, "found_share", color=TEAL, label="detection rate")
        line(ax, steps, rs, second[0], color=FINGER, label=second[1])
        ax.set(title=f"{size}x{size} fingertip", xlabel="step size (cells per move)", ylabel="%", xticks=steps)
        ax.legend(frameon=False)
    return fig


def entropy_bonus():
    coefficients = (0.0, 0.01, 0.1, 0.3, 0.5, 1.0)
    rs = [runs(seeds=(3, 4, 5), entropy_coef=c) for c in coefficients]
    greedy = [play_saved(r, greedy=True) for r in rs]
    greedy_acc = [100 * (g[1] == g[0]).mean() for g in greedy]
    greedy_det = [100 * g[2].mean() for g in greedy]
    acc, det = [mean_sd(r)[0] for r in rs], [mean_sd(r, "found_share")[0] for r in rs]
    fig, axes = figure("5. Entropy regularization lowers detection, mainly through action sampling at test time",
        "Accuracy (left) and detection rate (right) as a function of the entropy regularization coefficient, with actions "
        "sampled from the policy as during training (solid) and with greedy action selection (dashed). With sampled "
        f"actions, the accuracy decreases from {acc[0]:.1f} % to {acc[-1]:.1f} % and the detection rate from {det[0]:.1f} % "
        f"to {det[-1]:.1f} %. With greedy actions, the accuracy decreases by only {greedy_acc[0] - greedy_acc[-1]:.1f} "
        "percentage points; most of the loss therefore results from the stochasticity of the policy at test time. "
        f"Coefficients 0 and 0.01 do not differ significantly ({compare(rs[0], rs[1])}). n = 3 seeds; greedy evaluation "
        "on 2,048 episodes per agent.", ncols=2)
    x = range(len(coefficients))
    for ax, metric, greedy_values, title in ((axes[0], "acc_last", greedy_acc, "accuracy"),
                                             (axes[1], "found_share", greedy_det, "detection rate")):
        line(ax, x, rs, metric, color=FINGER, label="sampled actions (as in training)")
        ax.plot(x, greedy_values, marker="o", linestyle="--", color=TEAL, label="greedy actions")
        ax.set(title=title, xlabel="entropy regularization coefficient", ylabel="%", ylim=(60, 100))
        ax.set_xticks(x, [str(c) for c in coefficients])
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    return fig


def sim_to_real():
    noise = (0.0, 0.05, 0.1, 0.2)
    plain = [runs(noise=n, train_noise=0.0) for n in noise]
    robust = [runs(noise=n, train_noise=0.1) for n in noise]
    matched = [runs(noise=n) for n in noise]
    answers = np.sum([r["confusion"] for r in plain[2]], axis=0)      # rows: true class, columns: predicted class
    acc = lambda rs: mean_sd(rs)[0]
    fig, (ax,) = figure("6. Noise-free training fails under sensor noise; noise injection closes the gap",
        "Accuracy as a function of the sensor noise at test time, for agents trained without noise (simulation), with "
        "10 % noise and with the test noise (matched). The agent trained without noise degrades to "
        f"{acc(plain[1]):.1f} %, {acc(plain[2]):.1f} % and {acc(plain[3]):.1f} % at 5, 10 and 20 % noise, although its "
        f"detection rate remains at least {min(mean_sd(r, 'found_share')[0] for r in plain):.0f} %; at 10 % noise it "
        f"predicts 'circle' in {100 * answers[:, 0].sum() / answers.sum():.0f} % of the episodes. Training with 10 % noise "
        f"(domain randomization) performs on par with matched training ({compare(robust[1], matched[1])} at 5 %, "
        f"{compare(robust[3], matched[3])} at 20 %) at a cost of {acc(matched[0]) - acc(robust[0]):.1f} percentage points "
        f"without noise ({compare(robust[0], matched[0])}). n = 3 seeds.")
    x = [100 * n for n in noise]
    line(ax, x, plain, color=FINGER, label="trained without noise")
    line(ax, x, robust, color=TEAL, label="trained with 10 % noise")
    line(ax, x, matched, color=GRAY, linestyle="--", label="trained with matched noise")
    ax.axhline(100 / 3, color=GRAY, linewidth=0.8, linestyle=":")
    ax.text(0, 100 / 3 + 1, "chance level", fontsize=8, color=GRAY)
    ax.set(xlabel="sensor noise at test time (%)", ylabel="accuracy (%)", xticks=x, ylim=(25, 100))
    ax.legend(frameon=False, loc="upper right")
    return fig


def memory_vs_fingertip():
    kinds = [("none", "no memory (feed-forward)", GRAY), ("rnn", "RNN", LIGHT_ORANGE), ("gru", "GRU", FINGER)]
    rs = {(size, memory): runs(seeds=range(10), patch_size=size, memory=memory) for size in (1, 3) for memory in ("none", "rnn", "gru")}
    gain = lambda size, other: mean_sd(rs[size, "gru"])[0] - mean_sd(rs[size, other])[0]
    fig, (ax,) = figure("7. The benefit of recurrent memory increases as the information per touch decreases",
        "Accuracy without memory (feed-forward), with an Elman RNN and with a GRU, for a 1x1 and a 3x3 fingertip (n = 10 "
        "seeds). With the 3x3 fingertip, a single contact often reveals an edge segment, and the GRU exceeds the "
        f"memoryless agent by {gain(3, 'none'):.1f} percentage points ({compare(rs[3, 'gru'], rs[3, 'none'])}). With the 1x1 "
        f"fingertip, each touch reveals a single cell; the GRU exceeds the memoryless agent by {gain(1, 'none'):.1f} "
        f"percentage points and the RNN by {gain(1, 'rnn'):.1f} pp ({compare(rs[1, 'gru'], rs[1, 'rnn'])}). The 1x1 agents with "
        "memory had not converged after 1,500 training rounds.")
    for i, (memory, label, color) in enumerate(kinds):
        for x, size in enumerate((1, 3)):
            m, s = mean_sd(rs[size, memory])
            ax.bar(x + (i - 1) * 0.27, m, 0.25, yerr=s, capsize=3, color=color, label=label if x == 0 else None)
            ax.text(x + (i - 1) * 0.27, m + s + 1, f"{m:.1f}", ha="center", fontsize=9)
    ax.axhline(100 / 3, color=GRAY, linewidth=0.8, linestyle=":")
    ax.set_xticks([0, 1], ["1x1 fingertip (one cell per touch)", "3x3 fingertip (nine cells per touch)"])
    ax.set(ylabel="accuracy (%)", ylim=(0, 105))
    ax.legend(frameon=False, loc="upper left")
    return fig


def memory_capacity():
    sizes = (32, 128, 256)
    rs = {(h, memory): runs(hidden_size=h, memory=memory) for h in sizes for memory in ("none", "rnn", "gru")}
    gain = lambda h: mean_sd(rs[h, "gru"])[0] - mean_sd(rs[h, "none"])[0]
    fig, (ax,) = figure("8. Recurrent memory improves accuracy only with sufficient hidden-state capacity",
        "Accuracy as a function of the number of hidden units, without memory, with an RNN and with a GRU (n = 3 seeds). "
        f"With 32 units, recurrent memory yields no measurable gain (GRU vs. none: {compare(rs[32, 'gru'], rs[32, 'none'])}). "
        f"With 256 units, the GRU exceeds the memoryless agent by {gain(256):.1f} percentage points "
        f"({compare(rs[256, 'gru'], rs[256, 'none'])}); at 128 units, the difference of {gain(128):.1f} pp "
        f"({compare(rs[128, 'gru'], rs[128, 'none'])} with 3 seeds) is confirmed with 10 seeds in figure 7. Accuracy "
        "increases with the number of hidden units for all memory types. The 32-unit agents had not fully converged "
        "after 1,500 training rounds.")
    for memory, label, color in (("none", "no memory (feed-forward)", GRAY), ("rnn", "RNN", LIGHT_ORANGE), ("gru", "GRU", FINGER)):
        line(ax, range(3), [rs[h, memory] for h in sizes], color=color, label=label)
    ax.set_xticks(range(3), [str(s) for s in sizes])
    ax.set(xlabel="hidden units", ylabel="accuracy (%)")
    ax.legend(frameon=False, loc="lower right")
    return fig


def learning_rate():
    rates = (0.0003, 0.001, 0.003, 0.005, 0.01, 0.03)
    rs = dict(zip(rates, (runs(lr=lr) for lr in rates)))
    acc = lambda lr: mean_sd(rs[lr])[0]
    near = min(welch_p(values(rs[0.003]), values(rs[lr])) for lr in (0.005, 0.01))
    fig, (ax,) = figure("9. Learning rates between 0.001 and 0.01 train stably; 0.003 yields the highest accuracy",
        "Final accuracy as a function of the learning rate at a batch size of 512 episodes (mean and standard deviation, "
        f"dots: individual seeds, n = 3). A learning rate of 0.003 yields {acc(0.003):.1f} %, significantly above the "
        f"baseline value of 0.001 ({acc(0.001):.1f} %, {compare(rs[0.003], rs[0.001])}); 0.005 and 0.01 do not differ "
        f"significantly from 0.003 (p ≥ {near:.2f}). At 0.0003, training had not converged after 1,500 rounds "
        f"({acc(0.0003):.1f} %). At 0.03, training becomes unstable: the final accuracy of the three seeds ranges from "
        f"{values(rs[0.03]).min():.1f} % to {values(rs[0.03]).max():.1f} %.")
    line(ax, rates, list(rs.values()), color=FINGER, label="mean and standard deviation")
    for lr, r in rs.items():
        ax.scatter([lr] * len(r), values(r), color=FINGER, alpha=0.35, s=18)
    ax.set_xscale("log")
    ax.set_xticks(rates, [str(r) for r in rates])
    ax.set(xlabel="learning rate (log scale)", ylabel="accuracy (%)")
    ax.legend(frameon=False, loc="lower center")
    return fig


def batch_size():
    sizes = (128, 512, 1024)
    rates = (0.0003, 0.001, 0.003, 0.005, 0.01, 0.03)
    rs = {(lr, n): runs(lr=lr, num_games=n) for lr in rates for n in sizes}
    minutes = lambda r: np.mean([x["seconds"] for x in r]) / 60
    stable = (0.001, 0.003, 0.005, 0.01)
    p_min = min(welch_p(values(rs[lr, 128]), values(rs[lr, 512])) for lr in stable)
    speedup = [minutes(rs[lr, 512]) / minutes(rs[lr, 128]) for lr in stable]
    fig, axes = figure("10. Batch size has no measurable effect on accuracy within the stable learning-rate range",
        "Final accuracy (left) and wall-clock time per training run (right) as a function of the batch size (episodes per "
        "update), for a fixed number of 1,500 updates (n = 3 seeds; times measured with four runs in parallel). For "
        "learning rates from 0.001 to 0.01, batch sizes of 128 and 512 episodes do not differ significantly in accuracy "
        f"(p ≥ {p_min:.2f}), while a batch size of 128 reduces the training time by a factor of {min(speedup):.1f} to "
        f"{max(speedup):.1f}. Larger batches improve the accuracy only for learning rates outside the stable range "
        "(0.0003 and 0.03).", ncols=2)
    colors = (LIGHT_ORANGE, TEAL, FINGER, LIGHT_TEAL, GRAY, PURPLE)
    for lr, color in zip(rates, colors):
        line(axes[0], range(3), [rs[lr, n] for n in sizes], color=color, label=f"learning rate {lr}")
        axes[1].plot(range(3), [minutes(rs[lr, n]) for n in sizes], marker="o", color=color)
    axes[0].set(title="accuracy", ylabel="%")
    axes[1].set(title="training time", ylabel="minutes per run")
    for ax in axes:
        ax.set_xticks(range(3), [str(n) for n in sizes])
        ax.set_xlabel("batch size (episodes per update)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    return fig


def reward_type():
    gammas = (0.0, 0.5, 0.95, 0.99)
    a = [runs(gamma=g) for g in gammas]
    b = [runs(gamma=g, reward="gain") for g in gammas]
    diff = [mean_sd(rb)[0] - mean_sd(ra)[0] for ra, rb in zip(a, b)]
    fig, (ax,) = figure("11. An information-gain reward improves accuracy at low discount factors",
        "Final accuracy for reward A (negative cross-entropy of the shape prediction after each move) and reward B (its "
        "reduction by each move, a proxy for information gain) as a function of the discount factor γ (n = 3 seeds). "
        f"Reward B exceeds reward A by {diff[0]:.1f} and {diff[1]:.1f} percentage points at γ = 0 and 0.5 "
        f"({compare(b[0], a[0])} and {compare(b[1], a[1])}); at γ = 0.95 and 0.99, the differences are not significant "
        f"({compare(b[2], a[2])} and {compare(b[3], a[3])}). Note the truncated y-axis.")
    line(ax, range(4), a, color=GRAY, label="A: negative cross-entropy after the move")
    line(ax, range(4), b, color=FINGER, label="B: reduction of the cross-entropy by the move")
    ax.set_xticks(range(4), [str(g) for g in gammas])
    ax.set(xlabel="discount factor γ", ylabel="accuracy (%)", ylim=(92, 97))
    ax.legend(frameon=False, loc="lower right")
    return fig


def tuned_under_noise():
    tuned = dict(hidden_size=256, lr=0.003, reward="gain", gamma=0.5, entropy_coef=0.003)
    boards = (16, 24, 32)
    base = {g: runs(seeds=range(5), noise=0.1, grid_size=g) for g in boards}
    best = {g: runs(seeds=range(5), noise=0.1, grid_size=g, **tuned) for g in boards}
    gain = lambda metric: [mean_sd(best[g], metric)[0] - mean_sd(base[g], metric)[0] for g in boards]
    p_max = max(welch_p(values(best[g]), values(base[g])) for g in boards)
    fig, axes = figure("12. Under sensor noise, the tuned configuration improves classification rather than detection",
        "Accuracy, detection rate and accuracy given detection at 10 % sensor noise for the baseline and a tuned "
        "configuration (256 hidden units, learning rate 0.003, reward B with γ = 0.5, entropy coefficient 0.003) on "
        f"three board sizes (n = 5 seeds). The tuned configuration increases the accuracy by {min(gain('acc_last')):.1f} to "
        f"{max(gain('acc_last')):.1f} percentage points ({'all p < 0.01' if p_max < 0.01 else f'p ≤ {p_max:.2f}'}), "
        f"mainly through a higher accuracy given detection (+{min(gain('acc_if_found')):.1f} to "
        f"+{max(gain('acc_if_found')):.1f} pp); the detection rate changes by at most "
        f"{max(abs(d) for d in gain('found_share')):.1f} pp. The contributions of the individual parameter changes were "
        "not isolated.", ncols=3, width=12)
    for ax, (metric, title) in zip(axes, (("acc_last", "accuracy"), ("found_share", "detection rate"),
                                          ("acc_if_found", "accuracy given detection"))):
        for i, (setup, color, label) in enumerate(((base, GRAY, "baseline"), (best, FINGER, "tuned"))):
            for x, board in enumerate(boards):
                m, s = mean_sd(setup[board], metric)
                ax.bar(x + (i - 0.5) * 0.38, m, 0.36, yerr=s, capsize=3, color=color, label=label if x == 0 else None)
        ax.set_xticks(range(3), [f"{b}x{b}" for b in boards])
        ax.set(title=title, xlabel="board size", ylim=(60, 100))
    axes[0].set_ylabel("%")
    axes[0].legend(frameon=False, loc="upper right")
    return fig


def touch_budget():
    touches = (8, 16, 24)
    noise = (0.0, 0.05, 0.1, 0.2)
    rs = {(n, t): runs(noise=n, max_touch_num=t) for n in noise for t in touches}
    acc = lambda n, t: mean_sd(rs[n, t])[0]
    fig, (ax,) = figure("13. Additional touches compensate for low but not for high sensor noise",
        "Final accuracy as a function of the touch budget for four sensor noise levels (n = 3 seeds). At 5 % noise, 16 "
        f"touches reach the accuracy of 8 noise-free touches ({acc(0.05, 16):.1f} % vs. {acc(0.0, 8):.1f} %, "
        f"{compare(rs[0.05, 16], rs[0.0, 8])}). At 20 % noise, 24 touches reach only {acc(0.2, 24):.1f} %; these agents had "
        "not converged after 1,500 training rounds. Touch budgets above 24 were not tested.")
    for n, color in zip(noise, (TEAL, LIGHT_TEAL, LIGHT_ORANGE, FINGER)):
        line(ax, touches, [rs[n, t] for t in touches], color=color, label=f"{100 * n:.0f} % noise")
    ax.axhline(acc(0.0, 8), color=GRAY, linewidth=0.8, linestyle=":")
    ax.text(8.2, acc(0.0, 8) - 0.6, "noise-free, 8 touches", va="top", fontsize=8, color=GRAY)
    ax.set(xlabel="touch budget (touches per episode)", ylabel="accuracy (%)", xticks=touches)
    ax.legend(frameon=False, loc="lower right")
    return fig


FINDINGS = [("factor_ranges", factor_ranges), ("error_decomposition", error_decomposition),
            ("undetected_class_bias", undetected_class_bias), ("step_size", step_size),
            ("entropy_bonus", entropy_bonus), ("sim_to_real", sim_to_real),
            ("memory_vs_fingertip", memory_vs_fingertip), ("memory_capacity", memory_capacity),
            ("learning_rate", learning_rate), ("batch_size", batch_size), ("reward_type", reward_type),
            ("tuned_under_noise", tuned_under_noise), ("touch_budget", touch_budget)]

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = [int(a) for a in sys.argv[1:]] or range(1, len(FINDINGS) + 1)
    for number in chosen:
        name, make = FINDINGS[number - 1]
        fig = make()
        path = OUT / f"{number:02d}_{name}.png"
        fig.savefig(path, dpi=150)
        print(path)
        if not sys.argv[1:]:
            plt.close(fig)                                        # all figures: files only, no windows
    if sys.argv[1:]:
        plt.show()
