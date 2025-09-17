# Comparison between Pure Bayes and Sprite R (improved)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse previously defined classes/functions from session:
# - HazardWorld, SpriteAgent, run_episode (with use_R/improved flags)
# If not present, this cell will raise; the session currently has them.

def find_seed_where_both_succeed(start_seed=1, max_tries=500):
    for s in range(start_seed, start_seed+max_tries):
        envA, trailA, stepsA, statusA = run_episode(use_R=False, improved=False, seed=s)
        envB, trailB, stepsB, statusB = run_episode(use_R=True, improved=True, seed=s)
        if statusA=="reached goal" and statusB=="reached goal":
            return s, (envA, trailA, stepsA, statusA), (envB, trailB, stepsB, statusB)
    return None, None, None

seed, pure_pack, imp_pack = find_seed_where_both_succeed(1, 500)
if seed is None:
    raise RuntimeError("Could not find a seed where both agents succeed within the search window.")

envA, trailA, stepsA, statusA = pure_pack
envB, trailB, stepsB, statusB = imp_pack

# Plot side-by-side path for the first found seed
fig, ax = plt.subplots(1,2, figsize=(12,6), sharex=True, sharey=True)

def draw_env(ax, env, title):
    ax.set_xlim(-0.5, env.size-0.5); ax.set_ylim(-0.5, env.size-0.5)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    sc = plt.Circle(env.scylla_center, env.scylla_radius, color='red', alpha=0.35); ax.add_patch(sc)
    ch = plt.Circle(env.chary_center, env.chary_radius, fill=False, linestyle="--", linewidth=2, color='blue'); ax.add_patch(ch)
    ax.scatter([env.goal[0]],[env.goal[1]], marker="*", s=160); ax.scatter([1],[1], marker="s", s=80)

draw_env(ax[0], envA, f"Pure Bayes — {statusA}, steps={stepsA}\n(seed={seed})")
ax[0].plot(trailA[:,0], trailA[:,1], '-o', linewidth=2)

draw_env(ax[1], envB, f"Sprite R (improved) — {statusB}, steps={stepsB}\n(seed={seed})")
ax[1].plot(trailB[:,0], trailB[:,1], '-o', linewidth=2)

plt.show()

# Aggregate comparison to ensure totals show improvement
def eval_two_agents(trials=300, seed0=1000):
    rows = []
    for name, (use_R, improved) in {
        "Pure Bayes": (False, False),
        "Sprite R (improved)": (True, True),
    }.items():
        succ=fail=tout=0
        steps_succ=[]
        for k in range(trials):
            _, _, steps, status = run_episode(use_R=use_R, improved=improved, seed=seed0+k)
            if status=="reached goal":
                succ+=1; steps_succ.append(steps)
            elif status.startswith("failed"):
                fail+=1
            else:
                tout+=1
        rows.append({
            "Agent": name,
            "Trials": trials,
            "Success": succ,
            "Fail (Scylla)": fail,
            "Timeout": tout,
            "Success Rate": succ/trials if trials else float('nan'),
            "Median Steps (success)": (np.median(steps_succ) if steps_succ else float('nan'))
        })
    return pd.DataFrame(rows)

df = eval_two_agents(trials=300, seed0=2000)
import caas_jupyter_tools as cj
cj.display_dataframe_to_user("Pure Bayes vs Sprite R (improved): 300-trial summary", df.round(3))

# Bar charts for success rate and median steps
fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].bar(df["Agent"], df["Success Rate"])
ax[0].set_ylim(0,1); ax[0].set_ylabel("success rate"); ax[0].set_title("Success rate (300 trials)")
ax[1].bar(df["Agent"], df["Median Steps (success)"])
ax[1].set_ylabel("median steps"); ax[1].set_title("Efficiency among successes")
plt.tight_layout()
plt.show()

