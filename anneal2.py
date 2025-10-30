# %% [markdown]
# # RL & MCTS as Annealing Processes (Tiny Colab Demo)
# - Experiment A: RL on a 1D chain with mid-run goal flip
# - Experiment B: MCTS (UCT) on same chain: fixed-c vs self-annealing c(t)
#
# R-rule is implemented as a real-valued actor-critic:
#   logits <- logits + η * sqrt(p(1-p)) [ α * A  - β * N ]
# with entropy-based gate (no complex numbers).

import numpy as np, math, matplotlib.pyplot as plt
rng = np.random.default_rng(0)

# -------------------------------
# Environment: 1D Chain (len=9)
# -------------------------------
class Chain:
    def __init__(self, n=9, max_steps=30, flip_at=150):
        self.n = n
        self.start = 0
        self.goal_A = n-1
        self.goal_B = 1
        self.max_steps = max_steps
        self.flip_at = flip_at
        self.t = 0
        self.goal = self.goal_A

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self.pos

    def maybe_flip(self):
        if self.t == self.flip_at:
            self.goal = self.goal_B

    def step(self, a):
        # a: 0=left, 1=right ; with 10% slip
        a_eff = a if rng.random() > 0.1 else (1-a)
        if a_eff == 0:
            self.pos = max(0, self.pos - 1)
        else:
            self.pos = min(self.n-1, self.pos + 1)
        self.steps += 1
        done = (self.pos == self.goal) or (self.steps >= self.max_steps)
        r = 1.0 if self.pos == self.goal else -0.01
        return self.pos, r, done

# utils
def softmax2(z0, z1):
    m = max(z0, z1)
    e0, e1 = math.exp(z0-m), math.exp(z1-m)
    s = e0+e1
    return e0/s, e1/s
def entropy2(p):
    if p<=0 or p>=1: return 0.0
    return -(p*math.log(p)+(1-p)*math.log(1-p))

# -------------------------------
# Experiment A: RL
# -------------------------------
def run_ql(env, episodes=300, eps=0.1, alpha=0.2, gamma=0.95):
    Q = np.zeros((env.n, 2))
    returns, steps_list = [], []
    env.t = 0; env.goal = env.goal_A
    for ep in range(episodes):
        env.maybe_flip()
        s = env.reset()
        total = 0.0
        for _ in range(env.max_steps):
            a = rng.integers(0,2) if rng.random()<eps else np.argmax(Q[s])
            s2, r, done = env.step(a)
            Q[s,a] += alpha*(r + gamma*np.max(Q[s2]) - Q[s,a])
            total += r
            s = s2
            env.t += 1
            if done: break
        returns.append(total)
        steps_list.append(env.steps)
    return np.array(returns), np.array(steps_list)

def run_pg(env, episodes=300, eta=0.1, ent_coef=0.02):
    # state-wise logits for two actions
    L = np.zeros((env.n,2))
    baseline = 0.0; beta=0.9
    returns, steps_list, temps = [], [], []
    env.t = 0; env.goal = env.goal_A
    for ep in range(episodes):
        env.maybe_flip()
        s = env.reset()
        total=0.0
        for _ in range(env.max_steps):
            p0, p1 = softmax2(L[s,0], L[s,1])
            a = 0 if rng.random()<p0 else 1
            s2, r, done = env.step(a)
            baseline = beta*baseline + (1-beta)*r
            # REINFORCE-like per-state
            grad0 = (1-p0) if a==0 else (-p0)
            grad1 = (1-p1) if a==1 else (-p1)
            L[s,0] += eta*((r-baseline)*grad0 + ent_coef*(0.5-p0))
            L[s,1] += eta*((r-baseline)*grad1 + ent_coef*(0.5-p1))
            total += r
            s = s2
            env.t += 1
            if done: break
        returns.append(total); steps_list.append(env.steps)
        # Proxy temperature: avg entropy across states visited is similar; store terminal state's entropy for simplicity
        p0e,p1e = softmax2(L[s,0], L[s,1]); temps.append(entropy2(p1e)/math.log(2))
    return np.array(returns), np.array(steps_list), np.array(temps)

def run_r_rule_ac(env, episodes=300, eta=0.12, alpha_c=1.0, beta_c=0.5):
    # Actor logits per state; critic as simple value function
    L = np.zeros((env.n,2))
    L_prev = np.zeros_like(L)
    V = np.zeros(env.n)
    gamma=0.95
    returns, steps_list, temps = [], [], []
    env.t = 0; env.goal = env.goal_A
    for ep in range(episodes):
        env.maybe_flip()
        s = env.reset()
        total=0.0
        while True:
            p0,p1 = softmax2(L[s,0], L[s,1])
            a = 0 if rng.random()<p0 else 1
            s2, r, done = env.step(a)
            td = r + gamma*(0 if done else V[s2]) - V[s]
            V[s] += 0.3*td

            # A: policy gradient advantage (use TD)
            grad0 = (1-p0) if a==0 else (-p0)
            grad1 = (1-p1) if a==1 else (-p1)
            A0, A1 = td*grad0, td*grad1

            # N: coherence penalty (momentum on logits)
            N0 = L[s,0] - L_prev[s,0]
            N1 = L[s,1] - L_prev[s,1]

            H = entropy2(p1)/math.log(2)
            sin_th, cos_th = H, (1-H)
            sigma = math.sqrt(max(1e-9, p1*(1-p1)))
            d0 = eta*sigma*( alpha_c*sin_th*A0 - beta_c*cos_th*N0 )
            d1 = eta*sigma*( alpha_c*sin_th*A1 - beta_c*cos_th*N1 )

            L_prev[s,0], L_prev[s,1] = L[s,0], L[s,1]
            L[s,0] += d0; L[s,1] += d1

            total += r
            s = s2
            env.t += 1
            if done: break
        returns.append(total); steps_list.append(env.steps)
        p0e,p1e = softmax2(L[s,0], L[s,1]); temps.append(math.sqrt(max(1e-9,p1e*(1-p1e))))
    return np.array(returns), np.array(steps_list), np.array(temps)

# Run Experiment A
env = Chain(n=9, max_steps=30, flip_at=150)
R_ql, S_ql = run_ql(env, episodes=300, eps=0.1)
env = Chain(n=9, max_steps=30, flip_at=150)
R_pg, S_pg, T_pg = run_pg(env, episodes=300, eta=0.1, ent_coef=0.02)
env = Chain(n=9, max_steps=30, flip_at=150)
R_rr, S_rr, T_rr = run_r_rule_ac(env, episodes=300, eta=0.12, alpha_c=1.0, beta_c=0.5)

# Plot RL results
x = np.arange(300)
def smooth(y,w=15):
    if len(y)<w: return y
    c = np.cumsum(np.insert(y,0,0.0))
    z = (c[w:]-c[:-w])/w
    return np.concatenate([np.full(w-1,z[0]), z])

plt.figure(figsize=(12,4))
plt.plot(smooth(R_ql), label='Q-learning (ε=0.1)', alpha=0.8)
plt.plot(smooth(R_pg), label='PG + entropy', alpha=0.8)
plt.plot(smooth(R_rr), label='R-rule AC (self-anneal)', alpha=0.95, linewidth=2)
plt.axvline(150/1, color='k', ls='--', alpha=0.5)  # flip roughly after 150 episodes/steps
plt.title('Experiment A — RL on Chain: Episode Reward (smoothed)')
plt.xlabel('Episode'); plt.ylabel('Return'); plt.legend(); plt.show()

plt.figure(figsize=(12,4))
plt.plot(smooth(S_ql), label='Q-learning', alpha=0.8)
plt.plot(smooth(S_pg), label='PG + entropy', alpha=0.8)
plt.plot(smooth(S_rr), label='R-rule AC', alpha=0.95, linewidth=2)
plt.axvline(150/1, color='k', ls='--', alpha=0.5)
plt.title('Experiment A — RL on Chain: Steps per Episode (smoothed)')
plt.xlabel('Episode'); plt.ylabel('Steps'); plt.legend(); plt.show()

plt.figure(figsize=(12,4))
plt.plot(smooth(T_pg), label='PG effective temp (entropy)', alpha=0.8)
plt.plot(smooth(T_rr), label='R-rule effective temp (sqrt p(1-p))', alpha=0.95, linewidth=2)
plt.title('Experiment A — Effective Temperature over Episodes')
plt.xlabel('Episode'); plt.ylabel('T_eff'); plt.legend(); plt.show()

# -------------------------------
# Experiment B: MCTS (UCT) on Chain
# -------------------------------
class Node:
    __slots__ = ("s","parent","N","W","children","untried")
    def __init__(self, s, parent=None):
        self.s = s; self.parent = parent
        self.N = 0; self.W = 0.0
        self.children = {}   # a -> Node
        self.untried = [0,1] # left/right

def rollout(env_state, goal, depth=12):
    s = env_state
    total=0.0
    for _ in range(depth):
        a = rng.integers(0,2)
        s,r,done = sim_step(s,a,goal)
        total += r
        if done: break
    return total

def sim_step(s,a,goal, n=9):
    a_eff = a if rng.random()>0.1 else (1-a)
    s2 = max(0, s-1) if a_eff==0 else min(n-1, s+1)
    done = (s2==goal)
    r = 1.0 if s2==goal else -0.01
    return s2, r, done

def uct_select(node, c):
    best_a, best_u = None, -1e9
    for a,child in node.children.items():
        Q = child.W/max(1,child.N)
        U = c * math.sqrt(math.log(max(1,node.N))/max(1,child.N))
        u = Q + U
        if u > best_u:
            best_u, best_a = u, a
    return best_a

def expand(node, goal):
    if not node.untried: return node
    a = node.untried.pop(rng.integers(0, len(node.untried)))
    s2, r, done = sim_step(node.s, a, goal)
    child = Node(s2, parent=node)
    node.children[a] = child
    return child

def backprop(node, value):
    cur = node
    while cur is not None:
        cur.N += 1
        cur.W += value
        cur = cur.parent

def mcts_plan(start_s, goal, sims=60, depth=12, c=1.4, adaptive=False):
    root = Node(start_s)
    for _ in range(sims):
        node = root
        # selection
        while node.untried==[] and node.children:
            # adaptive c from node's empirical win variance
            c_eff = c
            if adaptive and node.N>1:
                p = max(0.0, min(1.0, node.W/max(1,node.N)))
                c_eff = c * math.sqrt(max(1e-9, p*(1-p)))
            a = uct_select(node, c_eff)
            node = node.children[a]
        # expansion
        if node.untried:
            node = expand(node, goal)
        # rollout
        val = rollout(node.s, goal, depth)
        # backprop
        backprop(node, val)
    # choose best action at root
    if not root.children: return rng.integers(0,2), root
    a_best = max(root.children.items(), key=lambda kv: kv[1].N)[0]
    return a_best, root

def run_mcts_episode(goal, sims=60, adaptive=False, n=9, max_steps=30):
    s = 0; total=0.0
    nodes_expanded = []
    for _ in range(max_steps):
        a, root = mcts_plan(s, goal, sims=sims, depth=12, c=1.4, adaptive=adaptive)
        nodes_expanded.append(sum(ch.N for ch in root.children.values()))
        s2, r, done = sim_step(s, a, goal, n=n)
        total += r
        s = s2
        if done: break
    return total, len(nodes_expanded), np.mean(nodes_expanded)

# Run Experiment B before and after goal flip
def batch(goal, sims, adaptive, trials=50):
    totals=[]; steps=[]; nodes=[]
    for _ in range(trials):
        tot, st, ne = run_mcts_episode(goal, sims=sims, adaptive=adaptive)
        totals.append(tot); steps.append(st); nodes.append(ne)
    return np.array(totals), np.array(steps), np.array(nodes)

tot_fixed_A, st_fixed_A, ne_fixed_A   = batch(goal=8, sims=60, adaptive=False, trials=50)
tot_adapt_A, st_adapt_A, ne_adapt_A   = batch(goal=8, sims=60, adaptive=True,  trials=50)
tot_fixed_B, st_fixed_B, ne_fixed_B   = batch(goal=1, sims=60, adaptive=False, trials=50)
tot_adapt_B, st_adapt_B, ne_adapt_B   = batch(goal=1, sims=60, adaptive=True,  trials=50)

# Plots for MCTS
labels = ['Fixed c (goal=8)','Adaptive c (goal=8)','Fixed c (goal=1)','Adaptive c (goal=1)']
means = [st_fixed_A.mean(), st_adapt_A.mean(), st_fixed_B.mean(), st_adapt_B.mean()]
plt.figure(figsize=(8,4))
plt.bar(labels, means)
plt.ylabel('Avg steps to goal (lower better)'); plt.xticks(rotation=15); plt.title('Experiment B — MCTS Steps')
plt.show()

means_nodes = [ne_fixed_A.mean(), ne_adapt_A.mean(), ne_fixed_B.mean(), ne_adapt_B.mean()]
plt.figure(figsize=(8,4))
plt.bar(labels, means_nodes)
plt.ylabel('Avg nodes expanded per move'); plt.xticks(rotation=15); plt.title('Experiment B — MCTS Effort')
plt.show()

print("Experiment A (RL) — mean return over last 50 episodes:")
print(f"  Q-learning: {R_ql[-50:].mean():.3f}")
print(f"  PG+entropy: {R_pg[-50:].mean():.3f}")
print(f"  R-rule AC : {R_rr[-50:].mean():.3f}")

print("\nExperiment B (MCTS) — avg steps (lower is better):")
print(f"  Fixed c (goal=8): {st_fixed_A.mean():.2f} | Adaptive c: {st_adapt_A.mean():.2f}")
print(f"  Fixed c (goal=1): {st_fixed_B.mean():.2f} | Adaptive c: {st_adapt_B.mean():.2f}")

print("\nExperiment B (MCTS) — avg nodes expanded per move:")
print(f"  Fixed c (goal=8): {ne_fixed_A.mean():.1f} | Adaptive c: {ne_adapt_A.mean():.1f}")
print(f"  Fixed c (goal=1): {ne_fixed_B.mean():.1f} | Adaptive c: {ne_adapt_B.mean():.1f}")

