import torch
import torch.nn.functional as F


def make_input(world, patches):
    """Building the agent input for one touch.

    In:  world (for positions, grid_size, touch_num, max_touch_num), patches (num_games, k, k)
    Out: tensor (num_games, patch_size**2 + 3) = patch values + position (y, x) + touch count, all scaled to 0..1
    """
    patch = torch.from_numpy(patches).reshape(world.num_games, -1)                          
    pos = torch.from_numpy(world.positions).float() / (world.grid_size - 1)             # scale positions on world grid to 0..1 
    count = torch.full((world.num_games, 1), world.touch_num / world.max_touch_num)     # scale used up touches per game / episode between 0..1
    return torch.cat([patch, pos, count], dim=1)                                        


def train(agent, world, num_rounds=1500, gamma=0.95, entropy_coef=0.01, lr=1e-3, reward="level", verbose=True):
    """Training the agent: guess -> supervised, moves -> REINFORCE

    In:  agent, world, training settings
         reward: "level" (A: how sure after the move) or "gain" (B: how much more sure through the move)
         verbose: print progress every 100 rounds
    Out: history, list of (round, accuracy after the last touch, touch from which the guess is right and sure for good;
         a game that never gets there counts as touches + 1)
    """
    history = []
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)                     # Adam optimizer updates all agent weights, step size lr
    
    for round_num in range(num_rounds):
        # --- 1. Play a batch of games ---
        patches = world.reset()                                                 # new boards and the first touch
        labels = torch.from_numpy(world.labels)                                 # true labels (shapes)
        h = torch.zeros(world.num_games, agent.hidden_size)                     # initialize memory cell to zero       
        guesses, log_probs, entropies = [], [], []
        
        for t in range(world.max_touch_num):
            x = make_input(world, patches)
            guess_logits, move_logits, h = agent.forward(x, h)
            guesses.append(guess_logits)
            if t < world.max_touch_num - 1:
                dist = torch.distributions.Categorical(logits=move_logits)
                a = dist.sample()
                log_probs.append(dist.log_prob(a))
                entropies.append(dist.entropy())
                patches = world.step(a.numpy())                                 # next step, new touch

        guesses = torch.stack(guesses)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        T, B = world.max_touch_num, world.num_games

        # cross entropy of classification (supervised head)
        ce = F.cross_entropy(guesses.reshape(T * B, -1), labels.repeat(T), reduction="none").reshape(T, B)

        # rewards for moves / actions (RL head)
        if reward == "level":                                                  # A: move t leads to touch t+1: good move = good guess after it
            rewards = -ce[1:].detach()
        elif reward == "gain":                                                 # B: information gain, how much the move improved the guess
            rewards = (ce[:-1] - ce[1:]).detach()
        else:
            raise ValueError(f"unknown reward: {reward}")

        # returns
        returns = torch.zeros_like(rewards)
        running = torch.zeros(B)
        for k in reversed(range(T- 1 )):
            running = rewards[k] + gamma * running                              # gamma < 1: later rewards count less
            returns[k] = running

        # advantage, relative to the baseline
        adv = returns - returns.mean(dim=1, keepdim=True)
        adv = adv / (adv.std() + 1e-8)  

        loss = (ce.mean() - (adv * log_probs).mean() - entropy_coef * entropies.mean()) # guess loss, move quality and probability, and a small bonus to keep exploring
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        # track progress, every round (cheap, no learning)
        with torch.no_grad():
            probs = torch.softmax(guesses, dim=2)                   # (T, B, shapes): scores -> probabilities
            correct = guesses.argmax(dim=2) == labels               # (T, B): guess after this touch is right
            sure = correct & (probs.max(dim=2).values >= 0.9)       # right AND at least 90 % sure: rules out lucky guesses
            stays = sure.int().flip(0).cumprod(0).flip(0)           # 1 if right and sure from this touch until the end
            n_end = stays.sum(0)                                    # per game: number of last touches that are right and sure
            touches = torch.where(n_end > 0, T - n_end + 1, T + 1)  # first such touch; T + 1 = never (counts as a failure)
            touches = touches.float().mean()                        # average over ALL games, failures included
            acc = correct[-1].float().mean()                        # share of correct guesses after the last touch
        history.append((round_num, acc.item(), touches.item()))    # touches: from which touch the guess is right and sure for good
        if verbose and round_num % 100 == 0:
            print(f"round {round_num}  loss {loss.item():.3f}  accuracy after {T} touches {acc:.2f}  right and sure from touch {touches:.1f}")

    return history