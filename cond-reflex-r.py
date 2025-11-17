import math
import matplotlib.pyplot as plt
import random

# -----------------------------
# R-rule update with epsilon-noise
# -----------------------------
def r_update(p, target, eta=0.1, noise=0.0):
    g = math.sqrt(max(p * (1 - p), 0.0))
    dp = eta * g * (target - p)
    p_new = p + dp + noise * random.uniform(-1, 1)
    return min(max(p_new, 0.0), 1.0)

# -----------------------------
# Setup: M (meaningful), S (neutral)
# -----------------------------
p_R_M = 0.3
p_R_S = 0.01       # <-- tiny uncertainty enables learning
eta_M = 0.2
eta_S = 0.1

n_trials = 120

pR_S_history = []
pY_S_history = []
trials = []

for t in range(1, n_trials + 1):

    # Update M: M -> R
    p_R_M = r_update(p_R_M, target=1.0, eta=eta_M)

    # Conditioning: S paired with M
    p_R_S = r_update(p_R_S, target=1.0, eta=eta_S)
    pY_S = 1 - p_R_S

    pR_S_history.append(p_R_S)
    pY_S_history.append(pY_S)
    trials.append(t)

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(trials, pR_S_history, label="p_R(S): S → R", linewidth=2)
plt.plot(trials, pY_S_history, label="p_Y(S): S → Y", linestyle="--", linewidth=2)

plt.title("Conditioned Reflex via R-Rule\nS transitions from Y → R")
plt.xlabel("Trial")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

