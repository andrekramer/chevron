# %% [markdown]
# # R-rule as Self-Annealing: Bandit + Double-Well Demos
# Real-valued approximation with two channels (A, N) and intrinsic temperature sqrt(p(1-p)).

import numpy as np
import math
import matplotlib.pyplot as plt
from dataclasses import dataclass

rng = np.random.default_rng(42)

# ----------------------------
# Utilities
# ----------------------------
def softmax2(l0, l1):
    # stable 2-class softmax
    m = max(l0, l1)
    e0 = math.exp(l0 - m)
    e1 = math.exp(l1 - m)
    s = e0 + e1
    return e0/s, e1/s

def entropy2(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p*math.log(p) + (1-p)*math.log(1-p))  # nats

def sign(x):
    return -1.0 if x < 0 else (1.0 if x > 0 else 0.0)

# ----------------------------
# 1) Drifting two-armed bandit
# ----------------------------
@dataclass
class BanditDrift:
    mu1_1: float = 0.6
    mu1_2: float = 0.2
    flip_t: int = 1500
    noise: float = 0.1
    T: int = 3000

    def reward(self, t, a):
        mu1 = self.mu1_1 if t < self.flip_t else self.mu1_2
        mu2 = self.mu1_2 if t < self.flip_t else self.mu1_1
        mu = mu1 if a == 0 else mu2
        return rng.normal(mu, self.noise)

# Baselines: epsilon-greedy and UCB1
def run_epsilon_greedy(bandit, eps=0.1):
    n1 = n2 = 0
    q1 = q2 = 0.0
    rews, temps, actions = [], [], []
    for t in range(bandit.T):
        if rng.random() < eps or (n1 == 0 and n2 == 0):
            a = rng.integers(0, 2)
        else:
            a = 0 if q1 >= q2 else 1
        r = bandit.reward(t, a)
        if a == 0:
            n1 += 1
            q1 += (r - q1)/n1
        else:
            n2 += 1
            q2 += (r - q2)/n2
        rews.append(r)
        temps.append(eps)  # fixed exploration prob as "temp"
        actions.append(a)
    return np.array(rews), np.array(temps), np.array(actions)

def run_ucb1(bandit, c=1.4):
    n1 = n2 = 0
    q1 = q2 = 0.0
    rews, temps, actions = [], [], []
    for t in range(bandit.T):
        if n1 == 0 or n2 == 0:
            a = 0 if n1 == 0 else 1
        else:
            bonus1 = c * math.sqrt(2*math.log(t+1)/n1)
            bonus2 = c * math.sqrt(2*math.log(t+1)/n2)
            u1 = q1 + bonus1
            u2 = q2 + bonus2
            a = 0 if u1 >= u2 else 1
        r = bandit.reward(t, a)
        if a == 0:
            n1 += 1; q1 += (r - q1)/n1
        else:
            n2 += 1; q2 += (r - q2)/n2
        rews.append(r)
        # treat exploration constant as "temp" proxy
        temps.append(c)
        actions.append(a)
    return np.array(rews), np.array(temps), np.array(actions)

# Policy Gradient baseline (REINFORCE) with entropy bonus
def run_pg_entropy(bandit, eta=0.05, ent_coef=0.01):
    l0 = l1 = 0.0  # logits
    b = 0.0        # baseline
    beta = 0.9
    rews, temps, actions, p_hist = [], [], [], []
    for t in range(bandit.T):
        p0, p1 = softmax2(l0, l1)
        a = 0 if rng.random() < p0 else 1
        r = bandit.reward(t, a)
        b = beta*b + (1-beta)*r
        # REINFORCE gradient (two-class)
        grad0 = (1.0 - p0) if a == 0 else (-p0)
        grad1 = (1.0 - p1) if a == 1 else (-p1)
        l0 += eta * (r - b) * grad0 + ent_coef * (0.5 - p0)
        l1 += eta * (r - b) * grad1 + ent_coef * (0.5 - p1)
        rews.append(r)
        # Effective temp as policy entropy
        H = entropy2(p1)
        temps.append(H / math.log(2))  # normalized [0,1]
        actions.append(a)
        p_hist.append(p1)
    return np.array(rews), np.array(temps), np.array(actions), np.array(p_hist)

# R-rule real-valued self-annealing policy
def run_r_rule_bandit(bandit, eta=0.08, alpha=1.0, beta_c=0.6):
    """
    Real-valued approximation:
      logits <- logits + eta * sqrt(p(1-p)) * [ alpha*sinθ * A  - beta*cosθ * N ]
    with:
      p = softmax(logits), entropy H -> sinθ = H/Hmax, cosθ = 1 - sinθ
      A: policy-gradient advantage (REINFORCE core)
      N: coherence term ~ (logits - logits_prev)
    """
    l0 = l1 = 0.0
    l0_prev = l1_prev = 0.0
    base = 0.0
    base_m = 0.9

    rews, temps, actions, p_hist = [], [], [], []
    for t in range(bandit.T):
        p0, p1 = softmax2(l0, l1)
        a = 0 if rng.random() < p0 else 1
        r = bandit.reward(t, a)
        base = base_m*base + (1-base_m)*r

        # A-term: standard two-class PG gradient for chosen action
        grad0 = (1.0 - p0) if a == 0 else (-p0)
        grad1 = (1.0 - p1) if a == 1 else (-p1)
        A0 = (r - base) * grad0
        A1 = (r - base) * grad1

        # N-term: coherence/momentum (penalize rapid changes)
        N0 = (l0 - l0_prev)
        N1 = (l1 - l1_prev)

        # Gate by entropy (uncertainty)
        H = entropy2(p1)
        sin_th = H / math.log(2)          # high when unsure
        cos_th = 1.0 - sin_th             # high when confident

        # Intrinsic temperature (self-annealing)
        sigma = math.sqrt(max(1e-9, p1*(1.0 - p1)))

        dl0 = eta * sigma * ( alpha*sin_th * A0 - beta_c*cos_th * N0 )
        dl1 = eta * sigma * ( alpha*sin_th * A1 - beta_c*cos_th * N1 )

        l0_prev, l1_prev = l0, l1
        l0 += dl0; l1 += dl1

        rews.append(r)
        temps.append(sigma)    # p(1-p)^{1/2} as effective temperature
        actions.append(a)
        p_hist.append(p1)
    return np.array(rews), np.array(temps), np.array(actions), np.array(p_hist)

# Run bandit experiment
bandit = BanditDrift(T=3000, flip_t=1500)

rew_eps, temp_eps, act_eps = run_epsilon_greedy(bandit, eps=0.1)
rew_ucb, temp_ucb, act_ucb   = run_ucb1(bandit, c=1.4)
rew_pg, temp_pg, act_pg, p_pg = run_pg_entropy(bandit, eta=0.05, ent_coef=0.01)
rew_r, temp_r, act_r, p_r     = run_r_rule_bandit(bandit, eta=0.08, alpha=1.0, beta_c=0.6)

def rolling(x, w=50):
    x = np.asarray(x)
    if w <= 1: return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    z = (c[w:] - c[:-w]) / float(w)
    # pad to original length
    head = np.full(w-1, z[0])
    return np.concatenate([head, z])

# Compute regret vs oracle
def oracle_mean(t):
    return bandit.mu1_1 if t < bandit.flip_t else bandit.mu1_2
def oracle_best(t):
    # best of the two arms at time t
    m1 = bandit.mu1_1 if t < bandit.flip_t else bandit.mu1_2
    m2 = bandit.mu1_2 if t < bandit.flip_t else bandit.mu1_1
    return max(m1, m2)

best_series = np.array([oracle_best(t) for t in range(bandit.T)])
cum_reg = lambda r: np.cumsum(best_series - r)

# Plot Bandit results
plt.figure(figsize=(12,5))
plt.plot(rolling(rew_eps,100), label='ε-greedy (avg reward)', alpha=0.8)
plt.plot(rolling(rew_ucb,100), label='UCB1 (avg reward)', alpha=0.8)
plt.plot(rolling(rew_pg,100), label='PG+entropy (avg reward)', alpha=0.8)
plt.plot(rolling(rew_r,100), label='R-rule (avg reward)', alpha=0.9, linewidth=2)
plt.axvline(bandit.flip_t, color='k', linestyle='--', alpha=0.5)
plt.title('Drifting 2-armed Bandit: Rolling average reward (window=100)')
plt.legend(); plt.xlabel('t'); plt.ylabel('reward'); plt.show()

plt.figure(figsize=(12,4))
plt.plot(cum_reg(rew_eps), label='ε-greedy', alpha=0.8)
plt.plot(cum_reg(rew_ucb), label='UCB1', alpha=0.8)
plt.plot(cum_reg(rew_pg), label='PG+entropy', alpha=0.8)
plt.plot(cum_reg(rew_r), label='R-rule', alpha=0.9, linewidth=2)
plt.axvline(bandit.flip_t, color='k', linestyle='--', alpha=0.5)
plt.title('Cumulative regret'); plt.legend(); plt.xlabel('t'); plt.ylabel('regret'); plt.show()

plt.figure(figsize=(12,4))
plt.plot(rolling(temp_pg,50), label='PG effective temp (entropy)', alpha=0.8)
plt.plot(rolling(temp_r,50), label='R-rule effective temp (sqrt p(1-p))', alpha=0.9, linewidth=2)
plt.title('Effective temperature over time'); plt.legend(); plt.xlabel('t'); plt.ylabel('T_eff'); plt.show()

# ----------------------------
# 2) Tilted double-well control
# ----------------------------
@dataclass
class DoubleWellEnv:
    eps: float = 0.2   # tilt
    dt: float = 0.02
    gamma: float = 0.2 # damping
    noise: float = 0.2

    def V(self, q):
        return 0.25*q**4 - 0.5*q**2 + self.eps*q

    def dVdq(self, q):
        return q**3 - q + self.eps

def simulate_langevin(env, T=4000, q0=-0.2, v0=0.0, T_schedule=lambda t: 0.5*math.exp(-t/1500)):
    q = q0; v = v0
    q_hist, T_hist = [], []
    for t in range(T):
        # External temperature schedule -> noise scale
        Tt = max(1e-6, T_schedule(t))
        # Langevin: m dv/dt = -dV/dq - gamma v + sqrt(2 gamma T) * xi
        a = -env.dVdq(q) - env.gamma * v + math.sqrt(2*env.gamma*Tt) * rng.normal()
        v += env.dt * a
        q += env.dt * v
        q_hist.append(q); T_hist.append(Tt)
    return np.array(q_hist), np.array(T_hist)

def simulate_r_rule_control(env, T=4000, q0=-0.2, v0=0.0, u0=0.8,
                            eta=0.08, alpha=1.0, beta_c=0.7):
    """
    Two actions: u in {-u0, +u0}. Policy over actions uses R-rule update.
    A-term aligns with downhill direction; N-term adds momentum coherence on logits.
    Intrinsic temperature sigma = sqrt(p(1-p)).
    """
    q = q0; v = v0
    l_neg = l_pos = 0.0
    l_neg_prev = l_pos_prev = 0.0

    q_hist, T_hist, a_hist = [], [], []
    base = 0.0; base_m = 0.98

    for t in range(T):
        # policy
        p_neg, p_pos = softmax2(l_neg, l_pos)
        a = -u0 if rng.random() < p_neg else +u0

        # physics step with chosen control
        noise = math.sqrt(2*env.gamma*env.noise) * rng.normal()
        a_phys = -env.dVdq(q) - env.gamma*v + a + noise
        v += env.dt * a_phys
        q += env.dt * v

        # reward = -V(q), advantage via control alignment with -dV/dq
        r = -env.V(q)
        base = base_m*base + (1-base_m)*r

        downhill = -env.dVdq(q)
        # A-term: prefer action aligned with downhill
        # gradient wrt logits ~ (indicator - prob)
        g_neg = (1.0 - p_neg) if a < 0 else (-p_neg)
        g_pos = (1.0 - p_pos) if a > 0 else (-p_pos)
        align = sign(downhill)  # +1 means push positive, -1 push negative
        A_neg = (r - base) * (-align) * g_neg
        A_pos = (r - base) * ( align) * g_pos

        # N-term: coherence on logits (momentum-like)
        N_neg = (l_neg - l_neg_prev)
        N_pos = (l_pos - l_pos_prev)

        # gate by entropy
        H = entropy2(p_pos)
        sin_th = H / math.log(2)
        cos_th = 1.0 - sin_th

        sigma = math.sqrt(max(1e-9, p_pos*(1.0 - p_pos)))

        dl_neg = eta * sigma * ( alpha*sin_th * A_neg - beta_c*cos_th * N_neg )
        dl_pos = eta * sigma * ( alpha*sin_th * A_pos - beta_c*cos_th * N_pos )

        l_neg_prev, l_pos_prev = l_neg, l_pos
        l_neg += dl_neg; l_pos += dl_pos

        q_hist.append(q); T_hist.append(sigma); a_hist.append(a)
    return np.array(q_hist), np.array(T_hist), np.array(a_hist)

env = DoubleWellEnv(eps=0.15, dt=0.02, gamma=0.25, noise=0.25)

q_lang, T_lang = simulate_langevin(env, T=4000, q0=-0.1, v0=0.0,
                                   T_schedule=lambda t: 0.6*math.exp(-t/1800))
q_rr, T_rr, a_rr = simulate_r_rule_control(env, T=4000, q0=-0.1, v0=0.0,
                                           u0=0.8, eta=0.08, alpha=1.0, beta_c=0.6)

def well_side(q):  # -1 left, +1 right
    return np.where(q < 0, -1, 1)

plt.figure(figsize=(12,4))
plt.plot(q_lang, label='Langevin (external anneal)', alpha=0.8)
plt.plot(q_rr, label='R-rule (self-anneal)', alpha=0.9, linewidth=2)
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.title('Double-well: position over time'); plt.xlabel('t'); plt.ylabel('q')
plt.legend(); plt.show()

plt.figure(figsize=(12,4))
plt.plot(T_lang, label='External T(t)', alpha=0.8)
plt.plot(T_rr, label='R-rule T_eff ~ sqrt(p(1-p))', alpha=0.9, linewidth=2)
plt.title('Effective temperature over time'); plt.xlabel('t'); plt.ylabel('T')
plt.legend(); plt.show()

# histogram of final well occupancy from multiple runs
def repeat_runs(n=50):
    left_lang = right_lang = left_rr = right_rr = 0
    for k in range(n):
        qL,_ = simulate_langevin(env, T=3000, q0=-0.2+0.4*rng.random(), v0=0.0,
                                 T_schedule=lambda t: 0.6*math.exp(-t/1800))
        qR,_,_ = simulate_r_rule_control(env, T=3000, q0=-0.2+0.4*rng.random(), v0=0.0,
                                         u0=0.8, eta=0.08, alpha=1.0, beta_c=0.6)
        if qL[-1] < 0: left_lang += 1
        else: right_lang += 1
        if qR[-1] < 0: left_rr += 1
        else: right_rr += 1
    return (left_lang, right_lang, left_rr, right_rr)

lL, rL, lR, rR = repeat_runs(50)
plt.figure(figsize=(6,4))
bars = [lL, rL, lR, rR]
labels = ['Lang: Left', 'Lang: Right', 'R-rule: Left', 'R-rule: Right']
plt.bar(labels, bars)
plt.title('Final well occupancy (50 runs)'); plt.ylabel('count'); plt.xticks(rotation=20)
plt.show()

print("Bandit — cumulative regret (lower is better):")
print(f"  ε-greedy:  {cum_reg(rew_eps)[-1]:.2f}")
print(f"  UCB1:      {cum_reg(rew_ucb)[-1]:.2f}")
print(f"  PG+ent:    {cum_reg(rew_pg)[-1]:.2f}")
print(f"  R-rule:    {cum_reg(rew_r)[-1]:.2f}")

print("\nDouble-well — final well occupancy (50 runs):")
print(f"  Langevin:  left={lL}, right={rL}")
print(f"  R-rule:    left={lR}, right={rR}")

