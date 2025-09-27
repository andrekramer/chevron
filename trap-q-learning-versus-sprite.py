# trap_q_learning_versus_sprite.py
# Sprite (one-shot oppositional flip) vs ε-greedy Q-learning
# in a nonstationary gridworld where the "best corridor" can flip into a trap.

import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Environment definition
# -----------------------
GRID_N = 5
START = (0, 0)
GOAL = (4, 4)
ACTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up
N_ACTIONS = 4

def in_bounds(x, y):
    return 0 <= x < GRID_N and 0 <= y < GRID_N

class TrapCorridorEnv:
    """
    5x5 grid. Start (0,0) -> Goal (4,4).
    A high-prior 'corridor' (top row then rightmost column) is occasionally booby-trapped at (0,3).
    - Trap reward = -10 and ends the episode.
    - With prob p_trap_on, stepping onto (0,3) activates the trap.
    - If active, it persists across episodes with probability 'stay_p'.
    """
    def __init__(self, p_trap_on=0.15, stay_p=0.8, max_steps=50, seed=42):
        self.p_trap_on = p_trap_on
        self.stay_p = stay_p
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.trap_active = False  # will update on resets

    def reset_env_state(self):
        # Trap deactivates between episodes with probability (1 - stay_p)
        if self.trap_active and self.rng.random() > self.stay_p:
            self.trap_active = False

    def reset(self):
        self.reset_env_state()
        self.t = 0
        self.pos = list(START)
        return tuple(self.pos)

    def step(self, action):
        dx, dy = ACTIONS[action]
        nx, ny = self.pos[0] + dx, self.pos[1] + dy
        if not in_bounds(nx, ny):
            reward = -0.2  # wall bump
            done = False
        else:
            self.pos = [nx, ny]
            reward = -0.01  # step cost
            done = False

        # Trap logic at (0,3)
        if tuple(self.pos) == (0, 3):
            if not self.trap_active and self.rng.random() < self.p_trap_on:
                self.trap_active = True
            if self.trap_active:
                return tuple(self.pos), -10.0, True, {}

        if tuple(self.pos) == GOAL:
            return tuple(self.pos), 1.0, True, {}

        self.t += 1
        if self.t >= self.max_steps:
            done = True
        return tuple(self.pos), reward, done, {}

# -----------------------
# Agents
# -----------------------
class QLearningAgent:
    def __init__(self, alpha=0.3, gamma=0.95, epsilon=0.1, seed=0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(N_ACTIONS))
        self.rng = np.random.default_rng(seed)

    def select_action(self, s):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        q = self.Q[s]
        return int(np.argmax(q))

    def update(self, s, a, r, s2, done):
        qsa = self.Q[s][a]
        target = r if done else r + self.gamma * np.max(self.Q[s2])
        self.Q[s][a] += self.alpha * (target - qsa)

class SpriteAgent(QLearningAgent):
    """
    Same Q-learner, but after first catastrophic surprise at (0,3),
    apply an oppositional 'flip' to Q-values near the corridor to
    systematically promote alternatives (one-shot pivot).
    """
    def __init__(self, alpha=0.3, gamma=0.95, epsilon=0.1, flip_scale=2.5, seed=0):
        super().__init__(alpha, gamma, epsilon, seed)
        self.flip_done = False
        self.flip_scale = flip_scale

    def update_on_event(self, s, r):
        if not self.flip_done and s == (0, 3) and r <= -10.0:
            self.oppositional_flip()
            self.flip_done = True

    def oppositional_flip(self):
        # Apply a reciprocal-like transform; penalize 'corridor directions'
        for x in range(GRID_N):
            for y in range(GRID_N):
                s = (x, y)
                q = self.Q[s]
                if np.all(q == 0.0):
                    continue
                corridor_dirs = []
                if y == 0 and x < 4:
                    corridor_dirs.append(0)  # right along top row
                if x == 4 and y < 4:
                    corridor_dirs.append(1)  # down along rightmost col
                if not corridor_dirs:
                    continue
                eps = 1e-3
                shifted = q - np.min(q) + eps
                recip = 1.0 / shifted
                recip /= (np.sum(recip) + 1e-9)
                new_q = (q - np.max(q)) + self.flip_scale * recip
                for d in corridor_dirs:
                    new_q[d] -= self.flip_scale
                self.Q[s] = new_q

# -----------------------
# Training loops
# -----------------------
def run(agent_cls, episodes=300, seed=0):
    env = TrapCorridorEnv(p_trap_on=0.15, stay_p=0.8, max_steps=50, seed=seed)
    if agent_cls is SpriteAgent:
        agent = SpriteAgent(alpha=0.3, gamma=0.95, epsilon=0.1, flip_scale=2.5, seed=seed)
    else:
        agent = QLearningAgent(alpha=0.3, gamma=0.95, epsilon=0.1, seed=seed)

    rewards = []
    trap_hits = []

    for ep in range(episodes):
        s = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < 200:
            a = agent.select_action(s)
            s2, r, done, _ = env.step(a)
            if isinstance(agent, SpriteAgent):
                agent.update_on_event(s2, r)
            agent.update(s, a, r, s2, done)
            ep_reward += r
            if s2 == (0, 3) and r <= -10.0:
                trap_hits.append(1)
            else:
                trap_hits.append(0) if len(trap_hits) < ep + 1 else None
            s = s2
            steps += 1
        rewards.append(ep_reward)

        # ensure one trap-hit entry per episode
        if len(trap_hits) < ep + 1:
            trap_hits.append(0)

    return np.array(rewards), np.array(trap_hits[:episodes])

def multi_run(agent_cls, runs=30, episodes=300, base_seed=1000):
    stats = []
    for k in range(runs):
        rewards, trap_hits = run(agent_cls, episodes=episodes, seed=base_seed + k)
        # metric 1: episodes until rolling-20 trap rate < 5%
        window = 20
        rolling = np.convolve(trap_hits, np.ones(window) / window, mode="valid")
        adapt_ep = int(np.where(rolling < 0.05)[0][0]) if (rolling < 0.05).any() else episodes
        # metric 2: cumulative trap hits in first 100 episodes
        cum_100 = int(np.sum(trap_hits[:100]))
        # metric 3: mean reward first 100 episodes
        mean_r_100 = float(np.mean(rewards[:100]))
        stats.append({
            "adapt_ep(<5% traps rolling20)": adapt_ep,
            "trap_hits@100eps": cum_100,
            "mean_reward@100eps": mean_r_100
        })
    return pd.DataFrame(stats)

def avg_cum_traps(agent_cls, runs=30, episodes=300, base_seed=2000):
    cum = np.zeros(episodes)
    for k in range(runs):
        _, trap_hits = run(agent_cls, episodes=episodes, seed=base_seed + k)
        cum += np.cumsum(trap_hits)
    return cum / runs

def main():
    episodes = 300
    runs = 30

    df_q = multi_run(QLearningAgent, runs=runs, episodes=episodes)
    df_sp = multi_run(SpriteAgent, runs=runs, episodes=episodes)

    summary = pd.DataFrame({
        "metric": ["adapt_ep(<5% traps rolling20)", "trap_hits@100eps", "mean_reward@100eps"],
        "Q-learning (ε-greedy)": [
            df_q["adapt_ep(<5% traps rolling20)"].mean(),
            df_q["trap_hits@100eps"].mean(),
            df_q["mean_reward@100eps"].mean()
        ],
        "Sprite (one-shot flip)": [
            df_sp["adapt_ep(<5% traps rolling20)"].mean(),
            df_sp["trap_hits@100eps"].mean(),
            df_sp["mean_reward@100eps"].mean()
        ]
    }).round(2)

    print(summary.to_string(index=False))

    cum_q = avg_cum_traps(QLearningAgent, runs=runs, episodes=episodes)
    cum_s = avg_cum_traps(SpriteAgent, runs=runs, episodes=episodes)

    plt.figure(figsize=(7,4.5))
    plt.plot(cum_q, label="Q-learning (ε-greedy)")
    plt.plot(cum_s, label="Sprite (one-shot flip)")
    plt.xlabel("Episode")
    plt.ylabel("Avg cumulative trap hits")
    plt.title("Sprite one-shot flip reduces trap exposure")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

