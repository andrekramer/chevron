import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd
from math import sin, sqrt

# --- Environment ---
@dataclass
class GridWorld:
    size: int = 12
    goal: tuple = (10, 9)
    noise: float = 0.8  # higher = noisier observations
    
    def observe(self, pos):
        """Return a noisy observation vector pointing toward the goal."""
        dx = self.goal[0] - pos[0]
        dy = self.goal[1] - pos[1]
        v = np.array([dx, dy], dtype=float)
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v = v / nrm
        # add Gaussian noise
        v += np.random.normal(scale=self.noise, size=2)
        return v

# --- Sprite Agent with L-E-R update ---
class SpriteAgent:
    def __init__(self, env: GridWorld, alpha=0.25, theta=np.pi/6):
        self.env = env
        self.pos = np.array([1, 1], dtype=int)
        self.alpha = alpha   # coupling strength of the R rotation
        self.theta = theta   # "phase" of counterfactual tilt
        # logits over actions: up, down, left, right
        self.logits = np.zeros(4, dtype=float)  # L-space (Nigredo)
        self.actions = np.array([[1,0],[-1,0],[0,-1],[0,1]])  # right,left,down,up in (x,y)
        self.labels = ["right","left","down","up"]
    
    def step(self):
        # --- L: incorporate evidence into logits ---
        obs = self.env.observe(self.pos)
        # directions corresponding to actions (unit vectors)
        dirs = self.actions.astype(float)
        # normalize each dir
        for i in range(4):
            n = np.linalg.norm(dirs[i])
            if n > 0:
                dirs[i] /= n
        
        like = dirs @ obs  # alignment with observation
        self.logits += like  # add log-likelihood increments
        
        # --- E: softmax to produce a distribution over actions ---
        z = self.logits - np.max(self.logits)
        probs = np.exp(z) / np.sum(np.exp(z))
        
        # --- R: counterfactual tilt via 1-step lookahead (expected distance drop) ---
        goal = np.array(self.env.goal)
        current_dist = np.linalg.norm(goal - self.pos)
        next_positions = self.pos + self.actions
        # keep inside bounds
        next_positions[:,0] = np.clip(next_positions[:,0], 0, self.env.size-1)
        next_positions[:,1] = np.clip(next_positions[:,1], 0, self.env.size-1)
        dists = np.linalg.norm(goal - next_positions, axis=1)
        advantage = (current_dist - dists)  # positive if moving closer
        
        # normalize advantage to [0,1] for stability
        if advantage.max() > advantage.min():
            adv_norm = (advantage - advantage.min()) / (advantage.max() - advantage.min())
        else:
            adv_norm = np.zeros_like(advantage)
        
        phase_push = self.alpha * np.sqrt(probs * (1 - probs)) * sin(self.theta) * (2*adv_norm - 1)
        probs_r = probs + phase_push
        # ensure valid probabilities
        probs_r = np.clip(probs_r, 1e-8, None)
        probs_r = probs_r / probs_r.sum()
        
        # sample action
        a_idx = np.random.choice(4, p=probs_r)
        # update position
        self.pos = next_positions[a_idx]
        
        return {
            "pos": tuple(self.pos.tolist()),
            "probs_L": probs,     # after E (before R)
            "probs_LER": probs_r, # after R
            "logits": self.logits.copy(),
            "action": self.labels[a_idx],
            "advantage": advantage
        }

# --- Run simulation ---
np.random.seed(13)
env = GridWorld(size=12, goal=(10,9), noise=0.7)
agent = SpriteAgent(env, alpha=0.35, theta=np.pi/4)

T = 40
trail = [tuple(agent.pos.tolist())]
records = []
for t in range(T):
    rec = agent.step()
    records.append(rec)
    trail.append(rec["pos"])
    if rec["pos"] == env.goal:
        break

# --- Prepare visuals ---
grid = np.zeros((env.size, env.size))
grid[env.goal] = 1.0  # mark goal
path = np.array(trail)

# Plot 1: grid with path
fig1, ax1 = plt.subplots(figsize=(6,6))
ax1.imshow(grid.T, origin="lower")
ax1.plot(path[:,0], path[:,1], marker="o")
ax1.scatter([1],[1], marker="s", s=80)  # start
ax1.scatter([env.goal[0]],[env.goal[1]], marker="*", s=120)  # goal
ax1.set_title("Sprite navigating with L–E–R updates")
ax1.set_xlim(-0.5, env.size-0.5)
ax1.set_ylim(-0.5, env.size-0.5)
plt.show()

# Plot 2: action probabilities over time before and after R
probs_L = np.array([r["probs_L"] for r in records])
probs_LER = np.array([r["probs_LER"] for r in records])

fig2, ax2 = plt.subplots(figsize=(10,4))
ax2.plot(probs_L, linewidth=1.5)
ax2.set_title("Action probabilities after E (before R)")
ax2.set_xlabel("step")
ax2.set_ylabel("probability")
plt.show()

fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.plot(probs_LER, linewidth=1.5)
ax3.set_title("Action probabilities after R (phase-tilted)")
ax3.set_xlabel("step")
ax3.set_ylabel("probability")
plt.show()

# Display a concise table of steps
df = pd.DataFrame({
    "step": np.arange(len(records)),
    "pos": [r["pos"] for r in records],
    "action": [r["action"] for r in records],
    "p_right_E": probs_L[:,0],
    "p_left_E": probs_L[:,1],
    "p_down_E": probs_L[:,2],
    "p_up_E": probs_L[:,3],
    "p_right_R": probs_LER[:,0],
    "p_left_R": probs_LER[:,1],
    "p_down_R": probs_LER[:,2],
    "p_up_R": probs_LER[:,3],
})
import caas_jupyter_tools as cj
cj.display_dataframe_to_user("Sprite L–E–R Simulation Log", df)

