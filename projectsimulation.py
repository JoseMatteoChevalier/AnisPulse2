# ========================================
# Standalone Python Project Simulation
# ========================================

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# ----------------------------
# Project Setup
# ----------------------------
tasks = ["Requirements", "Database Design", "Backend API",
         "Third-party Integration", "Frontend UI", "Testing & Deployment"]
num_tasks = len(tasks)
baseline_durations = np.array([2, 3, 4, 3, 5, 2])  # in weeks

# Dependencies (adjacency matrix)
adjacency = np.array([
    [0,1,0,0,0,0],
    [0,0,1,0,1,0],
    [0,0,0,1,0,0],
    [0,0,0,0,0,1],
    [0,0,0,0,0,1],
    [0,0,0,0,0,0]
], dtype=float)

T = 20  # total simulation time in weeks
dt = 0.05
steps = int(T/dt)
time = np.linspace(0, T, steps+1)
diffusion = 0.01

# Risk levels (simulate some delays)
risk_levels = np.array([0, 0, 1, 0, 2, 0])  # e.g., task 3 and 5 are risky

# ----------------------------
# PDE Simulation Function
# ----------------------------
def reaction(u_i, duration):
    return 1.0 / duration

def run_pde(adjacency, durations, diffusion, risk_levels):
    u = np.zeros((num_tasks, steps+1))
    for t in range(steps):
        du = np.zeros(num_tasks)
        for i in range(num_tasks):
            preds = np.where(adjacency[:, i] > 0)[0]
            if len(preds) == 0 or all(u[p, t] >= 1.0 for p in preds):
                dur = durations[i] * max(1.0, risk_levels[i])
                du[i] += reaction(u[i, t], dur)
                for j in preds:
                    du[i] += adjacency[j, i] * (u[j, t] - u[i, t]) * diffusion
        u[:, t+1] = u[:, t] + du * dt
        u[:, t+1] = np.clip(u[:, t+1], 0, 1)
    return u

# ----------------------------
# Compute Curves
# ----------------------------
u_baseline = run_pde(adjacency, baseline_durations, diffusion, np.zeros(num_tasks))
baseline_curve = u_baseline.mean(axis=0)

u_risk = run_pde(adjacency, baseline_durations, diffusion, risk_levels)
risk_curve = u_risk.mean(axis=0)

# ----------------------------
# Create output folder
# ----------------------------
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# 3D Plot: Baseline vs Risk
# ----------------------------
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection="3d")

ax.plot(time, np.zeros_like(time), baseline_curve, color="blue", lw=2, label="Baseline")
ax.plot(time, np.ones_like(time), risk_curve, color="red", lw=2, linestyle="--", label="With Risk")

ax.set_xlabel("Time (weeks)")
ax.set_ylabel("Scenario")
ax.set_yticks([0,1])
ax.set_yticklabels(["Baseline","Risk"])
ax.set_zlabel("Completion (0–1)")
ax.set_title("Project Completion: Baseline vs Risk (3D)")
ax.view_init(elev=25, azim=-60)
ax.legend()

fig_path = os.path.join(output_folder, "3d_completion.png")
plt.savefig(fig_path)
print(f"Saved 3D completion plot to: {fig_path}")
plt.close()

# ----------------------------
# 2D Gantt-like Plot
# ----------------------------
fig2, ax2 = plt.subplots(figsize=(10,6))
y_pos = np.arange(num_tasks)
bar_height = 0.4

# Compute simple start/finish times
start_baseline = np.zeros(num_tasks)
finish_baseline = baseline_durations.copy()
for i in range(1, num_tasks):
    preds = np.where(adjacency[:, i] > 0)[0]
    if len(preds) > 0:
        start_baseline[i] = max(finish_baseline[preds])
    finish_baseline[i] = start_baseline[i] + baseline_durations[i]

start_risk = np.zeros(num_tasks)
finish_risk = baseline_durations * np.maximum(1, risk_levels)
for i in range(1, num_tasks):
    preds = np.where(adjacency[:, i] > 0)[0]
    if len(preds) > 0:
        start_risk[i] = max(finish_risk[preds])
    finish_risk[i] = start_risk[i] + baseline_durations[i]*np.maximum(1,risk_levels[i])

# Plot baseline
ax2.barh(y_pos - bar_height/2, finish_baseline-start_baseline, left=start_baseline,
         color="skyblue", edgecolor="black", height=bar_height, label="Baseline")
# Plot risk
ax2.barh(y_pos + bar_height/2, finish_risk-start_risk, left=start_risk,
         color="salmon", edgecolor="black", height=bar_height, label="With Risk")

# Add task names
for i, task in enumerate(tasks):
    ax2.text(finish_risk[i]+0.1, y_pos[i]+bar_height/2, task, va='center')

ax2.set_xlabel("Time (weeks)")
ax2.set_ylabel("Tasks")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(tasks)
ax2.set_title("Gantt-like Project Timeline: Baseline vs Risk")
ax2.legend()

gantt_path = os.path.join(output_folder, "gantt_comparison.png")
plt.savefig(gantt_path)
print(f"Saved Gantt plot to: {gantt_path}")
plt.close()

# ----------------------------
# Completion Times
# ----------------------------
baseline_completion = T * np.argmax(baseline_curve>=1)/steps
risk_completion = T * np.argmax(risk_curve>=1)/steps

print(f"Baseline completion time: {baseline_completion:.2f} weeks")
print(f"Risk-weighted completion time: {risk_completion:.2f} weeks")

