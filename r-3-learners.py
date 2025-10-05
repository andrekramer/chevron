# Unified simulation starting from the core dynamic:
#   ψ'(a) = ψ(a) + η * sqrt(p(a)(1-p(a))) * [ α sinθ A(a) - i β cosθ N(a) ]
#   p'(a) = |ψ'(a)|^2 / sum_b |ψ'(b)|^2
#
# We provide three aligned "readings":
# 1) Pavlovian conditioning (CS=B, US=R): write–store–read–forget.
# 2) Predictive Processing (PP) diagnostics: evidence vs. normative heads.
# 3) Levin-style biochemical conditioning with a slow trace W that feeds N.
#
# All three use the same core_step() function so the math lines up.
#
# Plots (one per figure, no seaborn, default colors):
#  - Pavlovian: R-index probability vs phases, plus CS/US schedules
#  - PP: surprise, norm alignment, gate energy, step-KL
#  - Levin: slow trace W and R-response across phases
#
# CSVs are saved for each experiment in /mnt/data.

# Retry unified simulation with careful imports and minimal state.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(123)

def core_step(psi, A_vec, N_vec, eta, alpha, beta, theta):
    p = np.abs(psi)**2
    p = p / (p.sum() + 1e-12)
    gate = np.sqrt(np.clip(p * (1 - p), 0.0, 1.0))
    drive_real = alpha * np.sin(theta) * A_vec
    drive_imag = -beta * np.cos(theta) * N_vec
    update = gate * (drive_real + 1j * drive_imag)
    psi_new = psi + eta * update
    p_new = np.abs(psi_new)**2
    p_new = p_new / (p_new.sum() + 1e-12)
    return psi_new, p_new

outdir = Path("/mnt/data")
outdir.mkdir(parents=True, exist_ok=True)

# Common params
K = 5
eta = 0.22
alpha = 1.0
beta  = 1.0
theta = np.deg2rad(30)

# 1) Pavlovian
T_pav = 380
CS_idx, R_idx = 0, 2
baseline_T, pair_T, test_T, ext_T = 80, 160, 80, 60
CS = np.zeros(T_pav); US = np.zeros(T_pav)
for t in range(0, baseline_T, 20): CS[t:t+5] = 1.0
off = baseline_T
for t in range(off, off+pair_T, 20):
    CS[t:t+5] = 1.0; US[t:t+5] = 1.0
off += pair_T
for t in range(off, off+test_T, 20): CS[t:t+5] = 1.0
off += test_T
for t in range(off, off+ext_T, 10): CS[t:t+5] = 1.0

psi = (rng.normal(size=K) + 1j * rng.normal(size=K)); psi /= np.linalg.norm(psi)
p_hist = np.zeros((T_pav, K)); R_prob = np.zeros(T_pav)
for t in range(T_pav):
    A_vec = np.zeros(K); N_vec = np.zeros(K)
    A_vec[CS_idx] = CS[t]; N_vec[R_idx] = US[t]
    psi, p = core_step(psi, A_vec, N_vec, eta, alpha, beta, theta)
    p_hist[t] = p; R_prob[t] = p[R_idx]

pav_df = pd.DataFrame(p_hist, columns=[f"p_{i}" for i in range(K)])
pav_df["R_prob"] = R_prob; pav_df["CS"] = CS; pav_df["US"] = US
pav_csv = outdir / "pavlovian_unified.csv"; pav_df.to_csv(pav_csv, index=False)

plt.figure(figsize=(8,4.5))
plt.plot(R_prob, label="p[R]"); plt.plot(CS, label="CS"); plt.plot(US, label="US")
plt.title("Pavlovian via unified core"); plt.xlabel("time step"); plt.ylabel("level / prob")
plt.legend(loc="best"); plt.tight_layout(); plt.show()

# 2) PP diagnostics
T_pp = 300; risky_idx, safe_idx = 1, 3
A_pp = np.zeros((T_pp,K)); N_pp = np.zeros((T_pp,K))
for t in range(T_pp):
    target = (t // 60) % K
    vec = -0.2*np.ones(K); vec[target]=1.0; vec += 0.12*rng.normal(size=K)
    A_pp[t] = vec / max(1e-8, np.linalg.norm(vec))
    nvec = np.zeros(K); nvec[risky_idx] = -1.0; nvec[safe_idx]=0.7; nvec += 0.05*rng.normal(size=K)
    N_pp[t] = nvec / max(1e-8, np.linalg.norm(nvec))

psi = (rng.normal(size=K) + 1j * rng.normal(size=K)); psi /= np.linalg.norm(psi)
p = np.abs(psi)**2; p /= p.sum()
surprise = np.zeros(T_pp); norm_align = np.zeros(T_pp); gate_energy = np.zeros(T_pp); step_kl = np.zeros(T_pp)

def kl(p_new, p_old):
    eps=1e-12; return float(np.sum(np.where(p_new>0, p_new*(np.log(p_new+eps)-np.log(p_old+eps)),0.0)))

for t in range(T_pp):
    surprise[t] = -float(np.dot(p, A_pp[t]))
    norm_align[t] = float(np.dot(p, N_pp[t]))
    gate_energy[t] = float(np.mean(p*(1-p)))
    p_prev = p.copy()
    psi, p = core_step(psi, A_pp[t], N_pp[t], eta, alpha, beta, theta)
    step_kl[t] = kl(p, p_prev)

pp_df = pd.DataFrame({"t":np.arange(T_pp),"surprise_proxy":surprise,"norm_alignment":norm_align,"gate_energy":gate_energy,"step_kl":step_kl})
pp_csv = outdir / "pp_unified.csv"; pp_df.to_csv(pp_csv, index=False)

plt.figure(figsize=(8,4.5))
plt.plot(surprise, label="−⟨p,A⟩"); plt.plot(norm_align, label="⟨p,N⟩")
plt.plot(gate_energy, label="mean p(1−p)"); plt.plot(step_kl, label="step KL")
plt.title("PP diagnostics via unified core"); plt.xlabel("time step"); plt.ylabel("value")
plt.legend(loc="best"); plt.tight_layout(); plt.show()

# 3) Levin-style with slow trace W feeding N
T_lev = 420; B_idx, R_idx = 0, 2
B_drive = np.zeros(T_lev); R_drive = np.zeros(T_lev)
baseline_T, pair_T, test_T, ext_T = 100, 180, 80, 60
for t in range(0, baseline_T, 20): B_drive[t:t+5] = 1.0
off = baseline_T
for t in range(off, off+pair_T, 20): B_drive[t:t+5] = 1.0; R_drive[t:t+5] = 1.0
off += pair_T
for t in range(off, off+test_T, 20): B_drive[t:t+5] = 1.0
off += test_T
for t in range(off, off+ext_T, 10): B_drive[t:t+5] = 1.0

W = 0.0; k_pair = 0.7; k_decay = 0.02; dt = 0.25
psi = (rng.normal(size=K) + 1j * rng.normal(size=K)); psi /= np.linalg.norm(psi)
p = np.abs(psi)**2; p /= p.sum()
W_hist = np.zeros(T_lev); R_prob_hist = np.zeros(T_lev)

for t in range(T_lev):
    A_vec = np.zeros(K); A_vec[B_idx] = B_drive[t]; A_vec[R_idx] = 0.7*R_drive[t]
    N_vec = np.zeros(K); N_vec[R_idx] = W
    psi, p = core_step(psi, A_vec, N_vec, eta, alpha, beta, theta)
    B_level = p[B_idx]; R_level = p[R_idx]
    W = W + dt * (k_pair * B_level * R_level - k_decay * W)
    W_hist[t] = W; R_prob_hist[t] = p[R_idx]

lev_df = pd.DataFrame({"t":np.arange(T_lev),"W":W_hist,"p_R":R_prob_hist,"B_drive":B_drive,"R_drive":R_drive})
lev_csv = outdir / "levin_unified.csv"; lev_df.to_csv(lev_csv, index=False)

plt.figure(figsize=(8,4.5))
plt.plot(B_drive, label="B drive"); plt.plot(R_drive, label="R drive")
plt.plot(W_hist, label="slow trace W"); plt.plot(R_prob_hist, label="p[R]")
plt.title("Levin-style (slow trace feeds N) via unified core"); plt.xlabel("time step"); plt.ylabel("level / prob")
plt.legend(loc="best"); plt.tight_layout(); plt.show()

(str(pav_csv), str(pp_csv), str(lev_csv))

