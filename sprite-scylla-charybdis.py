import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from math import sin

# --- Dual-hazard GridWorld ---
@dataclass
class HazardWorld:
    size: int = 14
    goal: tuple = (11, 9)          # (x,y)
    noise: float = 0.7             # observation noise
    scylla_center: tuple = (1, 12) # hard hazard near top-left
    scylla_radius: int = 2
    chary_center: tuple = (12, 1)  # soft hazard near bottom-right
    chary_radius: int = 3
    chary_pull: float = 0.6        # strength of whirlpool pull
    
    def observe(self, pos):
        """Noisy observation pointing toward the goal."""
        goal_vec = np.array(self.goal) - pos
        nrm = np.linalg.norm(goal_vec)
        if nrm > 0:
            goal_vec = goal_vec / nrm
        goal_vec += np.random.normal(scale=self.noise, size=2)
        return goal_vec
    
    def in_scylla(self, pos):
        return np.linalg.norm(np.array(pos) - np.array(self.scylla_center)) <= self.scylla_radius
    
    def charybdis_vector(self, pos):
        """Return a soft 'pull' toward Charybdis center (like a whirlpool)."""
        vec = np.array(self.chary_center) - np.array(pos)
        dist = np.linalg.norm(vec) + 1e-9
        inward = vec / dist
        tangent = np.array([ -inward[1], inward[0] ])  # 90° rotate to create swirl
        strength = self.chary_pull * (self.chary_radius / max(dist, self.chary_radius))
        return strength * (0.7*inward + 0.3*tangent)

# --- Sprite Agent ---
class SpriteAgent:
    def __init__(self, env: HazardWorld, alpha=0.35, theta=np.pi/4, use_R=True):
        self.env = env
        self.pos = np.array([1, 1], dtype=int)
        self.alpha = alpha
        self.theta = theta
        self.use_R = use_R
        # actions: right, left, down, up
        self.actions = np.array([[1,0],[-1,0],[0,-1],[0,1]])
        self.labels = ["right","left","down","up"]
        self.logits = np.zeros(4, dtype=float)
    
    def step(self):
        obs = self.env.observe(self.pos)
        dirs = self.actions.astype(float)
        for i in range(4):
            n = np.linalg.norm(dirs[i]); 
            if n>0: dirs[i]/=n
        
        # L: add evidence alignment into logits
        like = dirs @ obs
        self.logits += like
        
        # E: softmax
        z = self.logits - np.max(self.logits)
        probs = np.exp(z) / np.sum(np.exp(z))
        
        # R: counterfactual phase + hazard-aware bias
        probs_r = probs.copy()
        goal = np.array(self.env.goal)
        nxt = self.pos + self.actions
        nxt[:,0] = np.clip(nxt[:,0], 0, self.env.size-1)
        nxt[:,1] = np.clip(nxt[:,1], 0, self.env.size-1)
        current_dist = np.linalg.norm(goal - self.pos)
        dists = np.linalg.norm(goal - nxt, axis=1)
        advantage = (current_dist - dists)
        
        # penalize Scylla
        scylla_mask = np.array([self.env.in_scylla(tuple(p)) for p in nxt])
        advantage[scylla_mask] -= 5.0
        
        # add Charybdis soft pull penalty (avoid being sucked in)
        ch_vecs = np.array([self.env.charybdis_vector(tuple(p)) for p in nxt])
        ch_mag = np.linalg.norm(ch_vecs, axis=1)
        advantage -= 0.8*ch_mag
        
        # normalize advantage to [0,1]
        if advantage.max() > advantage.min():
            adv_norm = (advantage - advantage.min()) / (advantage.max()-advantage.min())
        else:
            adv_norm = np.zeros_like(advantage)
        
        if self.use_R:
            phase_push = self.alpha * np.sqrt(probs*(1-probs)) * sin(self.theta) * (2*adv_norm - 1)
            probs_r = np.clip(probs + phase_push, 1e-8, None)
            probs_r /= probs_r.sum()
        
        # sample action
        a_idx = np.random.choice(4, p=probs_r)
        self.pos = nxt[a_idx]
        
        return {
            "pos": tuple(self.pos.tolist()),
            "action": self.labels[a_idx],
            "probs_E": probs,
            "probs_R": probs_r,
            "failed": self.env.in_scylla(tuple(self.pos)),
            "reached": tuple(self.pos) == self.env.goal,
        }

def run_episode(use_R=True, seed=0, T=80):
    np.random.seed(seed)
    env = HazardWorld()
    agent = SpriteAgent(env, alpha=0.35, theta=np.pi/4, use_R=use_R)
    trail = [tuple(agent.pos.tolist())]
    records = []
    status = "running"
    for t in range(T):
        rec = agent.step()
        records.append(rec)
        trail.append(rec["pos"])
        if rec["failed"]:
            status = "failed (Scylla)"
            break
        if rec["reached"]:
            status = "reached goal"
            break
    return env, np.array(trail), pd.DataFrame(records), status

# Run
env_R, trail_R, df_R, status_R = run_episode(use_R=True, seed=11)
env_P, trail_P, df_P, status_P = run_episode(use_R=False, seed=11)

# Plot environment & paths
fig, ax = plt.subplots(1,2, figsize=(12,6), sharex=True, sharey=True)

def draw_env(ax, env, title):
    ax.set_xlim(-0.5, env.size-0.5); ax.set_ylim(-0.5, env.size-0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    # Scylla hard
    sc = plt.Circle(env.scylla_center, env.scylla_radius, color='red', alpha=0.35)
    ax.add_patch(sc)
    # Charybdis soft
    ch = plt.Circle(env.chary_center, env.chary_radius, fill=False, linestyle="--", linewidth=2, color='blue')
    ax.add_patch(ch)
    # goal & start
    ax.scatter([env.goal[0]],[env.goal[1]], marker="*", s=160)
    ax.scatter([1],[1], marker="s", s=80)
    # labels
    ax.text(env.scylla_center[0], env.scylla_center[1]+0.5, "Scylla", ha="center", fontsize=9)
    ax.text(env.chary_center[0], env.chary_center[1]-0.7, "Charybdis", ha="center", fontsize=9)
    ax.text(env.goal[0]+0.2, env.goal[1]+0.2, "Goal", fontsize=9)

draw_env(ax[0], env_R, f"Sprite with R (counterfactual tilt)\nStatus: {status_R}")
ax[0].plot(trail_R[:,0], trail_R[:,1], '-o', linewidth=2)

draw_env(ax[1], env_P, f"Sprite without R (pure Bayes)\nStatus: {status_P}")
ax[1].plot(trail_P[:,0], trail_P[:,1], '-o', linewidth=2)

plt.show()

# Show logs (first 10 rows)
import caas_jupyter_tools as cj
cj.display_dataframe_to_user("Scylla–Charybdis (with R) — first steps", df_R.head(10))
cj.display_dataframe_to_user("Scylla–Charybdis (no R) — first steps", df_P.head(10))

