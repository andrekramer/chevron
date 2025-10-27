import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Potential definition: tilted double-well
# ============================================================

# U(q) = a (q^2 - b)^2 + c q
# (q^2 - b)^2 is symmetric double well with minima near +/- sqrt(b)
# c q tilts it so one side is deeper than the other.
a = 1.0
b = 1.0
c = -0.6  # negative tilt: right well is deeper

def U(q):
    return a * (q**2 - b)**2 + c * q

def dU_dq(q):
    # derivative of U wrt q
    # d/dq [a(q^2 - b)^2 + c q] = 4 a q (q^2 - b) + c
    return 4 * a * q * (q**2 - b) + c

def force(q):
    # "actor" drive: downhill direction in U
    return -dU_dq(q)

# Scan landscape to locate shallow vs deeper minima
qgrid = np.linspace(-2.5, 2.5, 2000)
Ugrid = U(qgrid)

left_mask  = qgrid < 0
right_mask = qgrid > 0

left_q  = qgrid[left_mask][np.argmin(Ugrid[left_mask])]
left_U  = U(left_q)
right_q = qgrid[right_mask][np.argmin(Ugrid[right_mask])]
right_U = U(right_q)

# ============================================================
# 2. Boltzmann-like descent (pure gradient fall, overdamped)
# ============================================================

def simulate_boltzmann(q0, steps=400, eta=0.02):
    """
    q update: gradient descent on U
    p just tracked as 0 to keep plotting consistent with 2D phase plots.
    """
    q = q0
    p = 0.0
    qs, ps = [], []
    for t in range(steps):
        qs.append(q); ps.append(p)
        f = force(q)
        dq = eta * f
        q = q + dq
    return np.array(qs), np.array(ps)

# ============================================================
# 3. Hamiltonian-style (HMC-like) dynamics
# ============================================================

def simulate_hmc_like(q0, p0, steps=400, eps=0.05):
    """
    Discrete step approximation of Hamiltonian dynamics:
        dq/dt ~ p
        dp/dt ~ -dU/dq
    No damping.
    This will preserve "energy-like" structure (up to discretization error),
    so it tends to keep moving / orbit / sweep, not settle.
    """
    q = q0
    p = p0
    qs, ps = [], []
    for t in range(steps):
        qs.append(q); ps.append(p)
        # leapfrog-ish Euler step
        dp = -dU_dq(q) * eps       # p update ~ -grad U
        p = p + dp
        dq = p * eps               # q update ~ p
        q = q + dq
    return np.array(qs), np.array(ps)

# ============================================================
# 4. R-rule-style two-phase annealed dynamics
# ============================================================

def simulate_r_rule_two_phase(q0, p0,
                              steps_phase1=300, steps_phase2=500,
                              eta_q1=0.08, eta_p1=0.02,
                              eta_q2=0.05, eta_p2=0.08,
                              p_inject1=1.2, p_inject2=0.0,
                              beta1=0.01, beta2=0.5,
                              theta=np.pi/6):
    """
    Phase 1 ("explore"): 
      - q gets pushed both by downhill force and by p (momentum/self-model)
      - p decays only very weakly (beta1 small), so momentum persists.
      => can cross barriers.

    Phase 2 ("commit"):
      - q mainly follows force, with little/no p injection (p_inject2 small).
      - p is strongly damped (beta2 larger), so motion anneals / settles.
      => can stabilize in a basin.

    We also include a gate in phase 1, which is inspired by sqrt(p(1-p)):
    it boosts updates when the system is in a state of uncertainty,
    and soft-limits them when it's already very certain.
    In phase 2, we drop the gate to allow clean settling in the deep basin.
    """
    q = q0
    p = p0
    qs, ps = [], []

    # -------- Phase 1: coherent exploration
    for t in range(steps_phase1):
        qs.append(q); ps.append(p)

        f = force(q)

        # gating ~ sqrt(p(1-p)) analogue: stronger updates when q vs p are comparable
        amp2 = q**2 + p**2
        if amp2 == 0:
            gate = 0.0
        else:
            prob_like = q**2 / amp2
            gate = np.sqrt(prob_like * (1.0 - prob_like))

        # q gets force + injected p (like "momentum drives position")
        dq = eta_q1 * gate * (np.sin(theta) * f + p_inject1 * p)
        # p gently damps
        dp = eta_p1 * gate * (-beta1 * np.cos(theta) * p)

        q = q + dq
        p = p + dp

    # -------- Phase 2: anneal and land
    for t in range(steps_phase2):
        qs.append(q); ps.append(p)

        f = force(q)

        # now: no gate, heavier damping, and no momentum injection
        dq = eta_q2 * (np.sin(theta) * f + p_inject2 * p)
        dp = eta_p2 * (-beta2 * np.cos(theta) * p)

        q = q + dq
        p = p + dp

    return np.array(qs), np.array(ps)

# ============================================================
# 5. Run all simulations from same initial state
# ============================================================

# initial condition: start near left side, with significant internal momentum for R and HMC
q0 = -0.5
p0 = 2.0

# Boltzmann descent
qs_b, ps_b = simulate_boltzmann(q0)

# HMC-like
qs_hmc, ps_hmc = simulate_hmc_like(q0, p0)

# R-rule two-phase
qs_r, ps_r = simulate_r_rule_two_phase(q0, p0)

# Final/landing diagnostics
final_q_b   = qs_b[-1]
final_q_r   = qs_r[-1]
final_q_hmc = qs_hmc[-1]

final_E_b   = U(final_q_b)
final_E_r   = U(final_q_r)
final_E_hmc = U(final_q_hmc)

print("Boltzmann-like descent:")
print("  final q ≈", final_q_b, "U(q) ≈", final_E_b)
print()
print("HMC-like dynamics:")
print("  final q ≈", final_q_hmc, "U(q) ≈", final_E_hmc,
      "(keeps moving, doesn't necessarily settle)")
print()
print("R-rule two-phase dynamics:")
print("  final q ≈", final_q_r, "U(q) ≈", final_E_r)

# ============================================================
# 6. Plotting
# ============================================================

fig, axes = plt.subplots(2,2, figsize=(12,8))

# ---------- Panel A: Landscape ----------
axA = axes[0,0]
axA.plot(qgrid, Ugrid, linewidth=2, color='goldenrod')
axA.scatter([left_q, right_q], [left_U, right_U], color='red', zorder=5)
axA.scatter([q0], [U(q0)], color='orange', zorder=5)

axA.annotate("local\n(shallow well)",
             xy=(left_q, left_U),
             xytext=(left_q-1.5, left_U+6),
             arrowprops=dict(arrowstyle="->"))

axA.annotate("global\n(deeper well)",
             xy=(right_q, right_U),
             xytext=(right_q+0.5, right_U+8),
             arrowprops=dict(arrowstyle="->"))

axA.annotate("start",
             xy=(q0, U(q0)),
             xytext=(q0-1.0, U(q0)+8),
             arrowprops=dict(arrowstyle="->"))

axA.set_title("Tilted double-well energy landscape U(q)\ncloser basin vs deeper basin")
axA.set_xlabel("q (A / position-like)")
axA.set_ylabel("U(q)")
axA.axvline(0,color='gray',linewidth=0.5,linestyle='--')

# ---------- Panel B: Boltzmann descent ----------
axB = axes[0,1]
axB.plot(qs_b, ps_b, marker='.', linewidth=1, color='tab:blue')
axB.scatter(qs_b[0], ps_b[0], color='orange', zorder=5)
axB.annotate("start", xy=(qs_b[0], ps_b[0]),
             xytext=(qs_b[0]-0.8, ps_b[0]+0.2),
             arrowprops=dict(arrowstyle="->"))
axB.scatter(qs_b[-1], ps_b[-1], color='red', zorder=5)
axB.annotate("settles\n(shallow well)",
             xy=(qs_b[-1], ps_b[-1]),
             xytext=(qs_b[-1]-1.0, ps_b[-1]+0.3),
             arrowprops=dict(arrowstyle="->"))

axB.set_title("Pure descent (Boltzmann-like)\nfalls into nearest shallow well")
axB.set_xlabel("q (A / position-like)")
axB.set_ylabel("p (N / momentum-like)")
axB.axhline(0,color='gray',linewidth=0.5)
axB.axvline(0,color='gray',linewidth=0.5)
axB.set_aspect('equal', 'box')

# ---------- Panel C: HMC-like ----------
axC = axes[1,0]
axC.plot(qs_hmc, ps_hmc, marker='.', linewidth=1, color='tab:purple')
axC.scatter(qs_hmc[0], ps_hmc[0], color='orange', zorder=5)
axC.annotate("start", xy=(qs_hmc[0], ps_hmc[0]),
             xytext=(qs_hmc[0]-0.8, ps_hmc[0]+0.5),
             arrowprops=dict(arrowstyle="->"))
axC.scatter(qs_hmc[-1], ps_hmc[-1], color='red', zorder=5)
axC.annotate("keeps moving\n(explores barrier)",
             xy=(qs_hmc[-1], ps_hmc[-1]),
             xytext=(qs_hmc[-1]-1.2, ps_hmc[-1]+0.5),
             arrowprops=dict(arrowstyle="->"))

axC.set_title("Hamiltonian-style (HMC-like)\ncoherent exploration with momentum\n(no natural settling)")
axC.set_xlabel("q (A / position-like)")
axC.set_ylabel("p (N / momentum-like)")
axC.axhline(0,color='gray',linewidth=0.5)
axC.axvline(0,color='gray',linewidth=0.5)
axC.set_aspect('equal', 'box')

# ---------- Panel D: R-rule two-phase ----------
axD = axes[1,1]
axD.plot(qs_r, ps_r, marker='.', linewidth=1, color='tab:green')
axD.scatter(qs_r[0], ps_r[0], color='orange', zorder=5)
axD.annotate("start", xy=(qs_r[0], ps_r[0]),
             xytext=(qs_r[0]-0.8, ps_r[0]+0.5),
             arrowprops=dict(arrowstyle="->"))
axD.scatter(qs_r[-1], ps_r[-1], color='red', zorder=5)
axD.annotate("stabilizes\n(deeper well)",
             xy=(qs_r[-1], ps_r[-1]),
             xytext=(qs_r[-1]-1.2, ps_r[-1]+0.5),
             arrowprops=dict(arrowstyle="->"))

axD.set_title("R-rule-style two-phase dynamics\nexplores with momentum, then anneals\ncrosses barrier and commits deeper")
axD.set_xlabel("q (A / position-like)")
axD.set_ylabel("p (N / momentum-like)")
axD.axhline(0,color='gray',linewidth=0.5)
axD.axvline(0,color='gray',linewidth=0.5)
axD.set_aspect('equal', 'box')

plt.tight_layout()
plt.savefig("r_rule_vs_boltzmann_hmc.png", dpi=200)
plt.show()

# ============================================================
# 7. Stability check: Does R-rule actually stay in new minimum?
# ============================================================

def stability_diagnostics(qs, label, q_target, fraction=0.2):
    """Compute mean/std of q over last fraction of trajectory."""
    tail = qs[int(len(qs)*(1-fraction)):]
    mean_q = np.mean(tail)
    std_q = np.std(tail)
    print(f"{label}: mean q={mean_q:.4f}, std={std_q:.4f}, distance to target={abs(mean_q - q_target):.4f}")
    return mean_q, std_q

print("\n--- Stability diagnostics ---")
mean_q_r, std_q_r = stability_diagnostics(qs_r, "R-rule", right_q)
mean_q_h, std_q_h = stability_diagnostics(qs_hmc, "HMC-like", right_q)

# Plot q(t) for both HMC and R-rule to visualize
plt.figure(figsize=(8,4))
plt.plot(qs_hmc, label='HMC-like (momentum only)', color='purple', alpha=0.7)
plt.plot(qs_r, label='R-rule (annealed)', color='green', alpha=0.8)
plt.axhline(right_q, color='gray', linestyle='--', label='Right-hand minimum')
plt.axhline(left_q, color='lightgray', linestyle='--', label='Left-hand minimum')
plt.title("q(t) trajectories: HMC vs R-rule")
plt.xlabel("time steps")
plt.ylabel("q position")
plt.legend()
plt.tight_layout()
plt.savefig("r_rule_stability_check.png", dpi=200)
plt.show()

