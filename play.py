"""Watch a named, trained agent play a new random world.

    uv run play.py                  list the named agents and how good they are
    uv run play.py best             the agent "best" plays one new random world
    uv run play.py best --seed 7    the same world every time (any number)
    uv run play.py best --greedy    always the most likely move, no dice while playing
    uv run play.py --export         copy the trained named agents from the study into agents/

The named agents are the variants of the group "zoo" in sensitivity.toml and are trained like
every other study group (uv run study.py zoo). --export stores them in agents/ in half precision
(16 instead of 32 bits per number): about 0.2 MB per agent, so all of them fit into git, while the
hundreds of study agents in results/agents/ stay out of it.

The game opens in a window, next to the agent's results on new worlds. Everything is also saved
in play_output/ (the game as a GIF, four snapshots and the results).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

import visualize
from agent import Agent
from study import AGENTS, RUNS, WORLD_KEYS, load_config, plan, run_key

ZOO = Path(__file__).with_name("agents")
OUTPUT = Path(__file__).with_name("play_output")


def export():
    """Copy each trained zoo agent from results/agents/ into agents/<name>.pt (weights in half precision)
    plus agents/<name>.json (a card: description, settings and study results), which load() needs."""
    config = load_config()
    zoo = plan(config, ["zoo"])["zoo"]
    variants = config["groups"]["zoo"]["variants"]
    seed = zoo["seeds"][0]
    ZOO.mkdir(exist_ok=True)
    for order, (label, params) in enumerate(zoo["settings"]):
        name, key = label["variant"], run_key(params, seed)
        if not (AGENTS / f"{key}.pt").exists():
            print(f"  {name}: not trained yet, run  uv run study.py zoo")
            continue
        weights = torch.load(AGENTS / f"{key}.pt")
        torch.save({k: v.half() for k, v in weights.items()}, ZOO / f"{name}.pt")   # half the size, same behavior
        run = json.loads((RUNS / f"{key}.json").read_text())
        card = {"name": name, "order": order, "description": variants[name].get("description", ""),
                "accuracy": run["acc_last"], "touches_to_sure": run["touches_to_sure"],
                "seed": seed, "params": params}
        (ZOO / f"{name}.json").write_text(json.dumps(card, indent=2))
        print(f"  {name:12s} {(ZOO / f'{name}.pt').stat().st_size / 1e6:.2f} MB")


def cards():
    """The cards of all exported agents, in the order of sensitivity.toml."""
    found = [json.loads(path.read_text()) for path in ZOO.glob("*.json")]
    return {card["name"]: card for card in sorted(found, key=lambda card: card["order"])}


def load(name):
    """Rebuild a named agent from its card and its weights."""
    card = json.loads((ZOO / f"{name}.json").read_text())
    p = card["params"]
    agent = Agent(input_size=p["patch_size"] ** 2 + 3,            # patch values + position (2) + touch count
                  hidden_size=p["hidden_size"], memory=p["memory"])
    weights = torch.load(ZOO / f"{name}.pt")
    agent.load_state_dict({k: v.float() for k, v in weights.items()})   # stored in 16 bits, computed in 32
    agent.eval()
    return agent, card


def show_list():
    """Print all named agents with their accuracy."""
    available = cards()
    if not available:
        print("No agents in agents/ yet. Train them with  uv run study.py zoo  then run  uv run play.py --export")
        return
    print("Named agents (share of right guesses after the last touch, on new worlds):\n")
    for name, card in available.items():
        print(f"  {name:12s} {card['accuracy']:6.1%}   {card['description']}")
    print("\nWatch one with:  uv run play.py <name>")


def play(name, seed=None, greedy=False):
    """The agent plays one new random world (animation and snapshots), then its results on many new worlds.
    greedy: the agent always takes its most likely move instead of drawing one at random (in training, moves are always drawn)."""
    agent, card = load(name)
    world_args = {k: card["params"][k] for k in WORLD_KEYS}      # the board and fingertip it is tested on
    if seed is not None:                                          # same seed = same world
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    visualize.FIG_DIR = str(OUTPUT)                               # figures of this game go to play_output/
    label = f"{name} (most likely moves)" if greedy else name
    print(f"{label}: {card['description']}")
    visualize.show_game(agent, f"{name}_game.gif", f"{name}_frames.png", world_args=world_args, title=label,
                        greedy=greedy)
    r = visualize.plot_results(agent, f"{name}_results.png", world_args=world_args, greedy=greedy,
                               title=f"{label}: {card['description']}", window_title=f"{label}: results")
    print(f"right after the last touch: {r['accuracy']:.1%} of new worlds (with random moves: "
          f"{r['accuracy_random_moves']:.1%})")
    print(f"found the shape at all in {r['found']:.1%} of them, and was right in {r['right_if_found']:.1%} of those")
    print(f"saved in {OUTPUT.name}/: {name}_game.gif, {name}_frames.png, {name}_results.png")
    print('In the game window, "New game" or the key n lets the agent play a new world.')
    visualize.show()


def main():
    parser = argparse.ArgumentParser(description="Watch a named, trained agent play a new random world.")
    parser.add_argument("name", nargs="?", help="name of the agent (without a name: list them all)")
    parser.add_argument("--seed", type=int, help="play the same world every time")
    parser.add_argument("--greedy", action="store_true", help="always the most likely move, no dice while playing")
    parser.add_argument("--export", action="store_true", help="copy the trained named agents into agents/")
    args = parser.parse_args()
    if args.export:
        export()
    elif args.name is None:
        show_list()
    elif not (ZOO / f"{args.name}.json").exists():
        print(f"There is no agent '{args.name}'.\n")
        show_list()
    else:
        play(args.name, args.seed, args.greedy)


if __name__ == "__main__":
    main()
