import math
import random
import matplotlib.pyplot as plt

# ----------------------------------------------------
# R-rule style gated update
# ----------------------------------------------------
def r_update(p, target, eta=0.1, noise=0.0):
    """
    Gated probability update towards target in {0,1}.
    Learning strongest at p ~ 0.5, weakest at p ~ 0 or 1.
    """
    g = math.sqrt(max(p * (1 - p), 0.0))   # gate
    dp = eta * g * (target - p)
    p_new = p + dp + noise * random.uniform(-1, 1)
    # clamp to [0, 1]
    return min(max(p_new, 0.0), 1.0)

# ----------------------------------------------------
# Setup: M = meaningful signal, S = neutral signal
# ----------------------------------------------------

# M: meaningful signal -> should produce R
p_R_M = 0.3       # initial prob of R given M

# S: neutral signal -> initially produces Y, not R
p_R_S = 0.01      # tiny chance of R from S
p_Y_S = 0.99      # mostly Y at the start

# Learning rates:
eta_R_fast = 0.15   # fast (A-like) learning for S -> R
eta_Y_slow = 0.03   # slow (N-like) learning to weaken S -> Y
eta_M = 0.20        # M learns R fairly quickly

n_trials = 200

# Histories for plotting
trials = []
pR_S_history = []
pY_S_history = []
pR_M_history = []

for t in range(1, n_trials + 1):

    # 1. Update M: M is always followed by R
    p_R_M = r_update(p_R_M, target=1.0, eta=eta_M)

    # 2. Conditioning phase: S is presented together with M,
    #    and the combined event leads reliably to R.
    #    Fast A-like update: strengthen S -> R
    p_R_S = r_update(p_R_S, target=1.0, eta=eta_R_fast)

    # 3. Slow N-like update: gradually weaken S -> Y
    p_Y_S = r_update(p_Y_S, target=0.0, eta=eta_Y_slow)

    # Optional renormalization (not strictly necessary if independent):
    # ensure they don't exceed 1 when added
    total = p_R_S + p_Y_S
    if total > 1.0:
        p_R_S /= total
        p_Y_S /= total

    # Record histories
    trials.append(t)
    pR_S_history.append(p_R_S)
    pY_S_history.append(p_Y_S)
    pR_M_history.append(p_R_M)

# ----------------------------------------------------
# Plotting
# ----------------------------------------------------
plt.figure(figsize=(9, 5))

plt.plot(trials, pR_S_history, label="p_R(S): S → R (new conditioned response)", linewidth=2)
plt.plot(trials, pY_S_history, label="p_Y(S): S → Y (original response)", linestyle="--", linewidth=2)
plt.plot(trials, pR_M_history, label="p_R(M): M → R", linestyle=":", linewidth=1.5)

plt.title("Dual-Timescale Conditioning via R-Rule\nS transitions from Y → R (fast gain, slow loss)")
plt.xlabel("Trial")
plt.ylabel("Probability")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

