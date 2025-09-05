# streamlit_pde_tabs.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# Default Project Setup
# ----------------------------
default_tasks = ["Requirements", "Database Design", "Backend API",
                 "Third-party Integration", "Frontend UI", "Testing & Deployment"]
default_durations = np.array([2, 3, 4, 3, 5, 2])  # weeks
num_tasks = len(default_tasks)

default_adjacency = np.array([
    [0,1,0,0,0,0],
    [0,0,1,0,1,0],
    [0,0,0,1,0,0],
    [0,0,0,0,0,1],
    [0,0,0,0,0,1],
    [0,0,0,0,0,0]
], dtype=float)

T = 20
dt = 0.05
steps = int(T/dt)
time = np.linspace(0, T, steps+1)
diffusion = 0.01

# ----------------------------
# PDE Simulation Function
# ----------------------------
def reaction(u_i, duration):
    return 1.0 / duration

def run_pde(adjacency, durations, diffusion, risk_levels):
    num_tasks = len(durations)
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
# Sidebar for Risk Input
# ----------------------------
st.sidebar.header("Default Risk Levels")
risk_levels = np.array([st.sidebar.slider(task, 0, 5, 0) for task in default_tasks])

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2 = st.tabs(["Project Visualization", "Edit Tasks & Durations"])

# ----------------------------
# Tab 1: Visualization
# ----------------------------
with tab1:
    st.header("Project Completion Curves & Gantt")

    # Compute curves
    u_baseline = run_pde(default_adjacency, default_durations, diffusion, np.zeros(num_tasks))
    baseline_curve = u_baseline.mean(axis=0)
    u_risk = run_pde(default_adjacency, default_durations, diffusion, risk_levels)
    risk_curve = u_risk.mean(axis=0)

    # 3D Plot
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
    st.pyplot(fig)

    # 2D Gantt
    fig2, ax2 = plt.subplots(figsize=(10,6))
    y_pos = np.arange(num_tasks)
    bar_height = 0.4

    # Baseline start/finish
    start_baseline = np.zeros(num_tasks)
    finish_baseline = default_durations.copy()
    for i in range(1,num_tasks):
        preds = np.where(default_adjacency[:,i]>0)[0]
        if len(preds) > 0:
            start_baseline[i] = max(finish_baseline[preds])
        finish_baseline[i] = start_baseline[i] + default_durations[i]

    # Risk start/finish
    adjusted_durations = default_durations * np.maximum(1, risk_levels)
    start_risk = np.zeros(num_tasks)
    finish_risk = adjusted_durations.copy()
    for i in range(1,num_tasks):
        preds = np.where(default_adjacency[:,i]>0)[0]
        if len(preds) > 0:
            start_risk[i] = max(finish_risk[preds])
        finish_risk[i] = start_risk[i] + adjusted_durations[i]

    ax2.barh(y_pos - bar_height/2, finish_baseline-start_baseline, left=start_baseline,
             color="skyblue", edgecolor="black", height=bar_height, label="Baseline")
    ax2.barh(y_pos + bar_height/2, finish_risk-start_risk, left=start_risk,
             color="salmon", edgecolor="black", height=bar_height, label="With Risk")
    for i, task in enumerate(default_tasks):
        ax2.text(finish_risk[i]+0.1, y_pos[i]+bar_height/2, task, va='center')
    ax2.set_xlabel("Time (weeks)")
    ax2.set_ylabel("Tasks")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(default_tasks)
    ax2.set_title("Gantt-like Project Timeline: Baseline vs Risk")
    ax2.legend()
    st.pyplot(fig2)

# ----------------------------
# Tab 2: Edit Tasks & Durations
# ----------------------------
with tab2:
    st.header("Edit Task Durations & Risks")

    # Editable tasks/durations
    tasks_input = st.text_area("Task Names (comma separated)", value=",".join(default_tasks))
    durations_input = st.text_area("Durations (weeks, comma separated)", value=",".join(map(str, default_durations)))
    new_tasks = [t.strip() for t in tasks_input.split(",")]
    new_durations = np.array([float(d) for d in durations_input.split(",")])

    # Slider for risks
    st.subheader("Adjust Risks")
    new_risks = np.array([st.slider(task, 0, 5, 0) for task in new_tasks])

    st.write("**Tasks:**", new_tasks)
    st.write("**Durations:**", new_durations)
    st.write("**Risks:**", new_risks)
