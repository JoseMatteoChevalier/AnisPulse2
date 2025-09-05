import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from st_aggrid import AgGrid, GridOptionsBuilder
import json
from io import BytesIO
import aspose.tasks as tasks

# ----------------------------
# PDE Simulation Function
# ----------------------------
def reaction(u_i, duration):
    return 1.0 / duration if duration > 0 else 0

def run_pde(adjacency, durations, diffusion, risk_levels, T=20, dt=0.05):
    num_tasks = len(durations)
    steps = int(T/dt)
    u = np.zeros((num_tasks, steps+1))
    for t in range(steps):
        du = np.zeros(num_tasks)
        for i in range(num_tasks):
            preds = np.where(adjacency[:,i] > 0)[0]
            if len(preds) == 0 or all(u[p,t] >= 1.0 for p in preds):
                dur = durations[i] * max(1.0, risk_levels[i])
                if dur > 0:
                    du[i] += reaction(u[i,t], dur)
                for j in preds:
                    du[i] += adjacency[j,i] * (u[j,t] - u[i,t]) * diffusion
        u[:,t+1] = np.clip(u[:,t] + du * dt, 0, 1)
    return u

# ----------------------------
# Defaults
# ----------------------------
default_tasks = ["Requirements", "Database Design", "Backend API",
                 "Third-party Integration", "Frontend UI", "Testing & Deployment"]
default_durations = np.array([2,3,4,3,5,2], dtype=float)
default_risks = np.array([0,0,1,0,2,0], dtype=float)
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
time = np.linspace(0, T, int(T/dt)+1)
diffusion = 0.01

# ----------------------------
# Streamlit Layout
# ----------------------------
st.set_page_config(page_title="Pulse Wave Simulation", layout="wide")
st.title("📊 Pulse Wave Simulation of Project Schedule")

tab1, tab2 = st.tabs(["Default Simulation", "Custom Project"])
with tab1:
    st.write("Default project simulation view here")
with tab2:
    st.write("Custom project simulation view here")

simulation_mode = st.sidebar.selectbox("Simulation Mode", ["Default", "Custom Project"])
recalculate = st.sidebar.button("Recalculate")
time_step = st.sidebar.slider("Time Step", min_value=0.1, max_value=5.0, value=1.0)

with st.expander("Advanced Options"):
    diffusion = st.slider("Diffusion Factor", 0.0, 1.0, 0.5)
    risk_weight = st.slider("Risk Weight", 0, 100, 50)

# ----------------------------
# TAB 1: Default Example
# ----------------------------
with tab1:
    st.subheader("Default Example: Software Project")

    col1, col2, col3 = st.columns(3)

    # Baseline simulation
    u_baseline = run_pde(default_adjacency, default_durations, diffusion, np.zeros(len(default_tasks)))
    baseline_curve = u_baseline.mean(axis=0)

    # Risk simulation
    u_risk = run_pde(default_adjacency, default_durations, diffusion, default_risks)
    risk_curve = u_risk.mean(axis=0)

    # Delay simulation (e.g., Task 3 delayed by 50%)
    delay_durations = default_durations.copy()
    delay_durations[2] *= 1.5  # Increase duration of Task 3 (Backend API)
    u_delay = run_pde(default_adjacency, delay_durations, diffusion, np.zeros(len(default_tasks)))
    delay_curve = u_delay.mean(axis=0)

    # 3D Plot
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(time, np.zeros_like(time), baseline_curve, color="blue", lw=2, label="Baseline")
        ax.plot(time, np.ones_like(time)*0.5, risk_curve, color="red", lw=2, linestyle="--", label="With Risk")
        ax.plot(time, np.ones_like(time), delay_curve, color="green", lw=2, linestyle=":", label="With Delay")
        ax.set_xlabel("Time (weeks)")
        ax.set_ylabel("Scenario")
        ax.set_yticks([0,0.5,1])
        ax.set_yticklabels(["Baseline","Risk","Delay"])
        ax.set_zlabel("Completion (0–1)")
        ax.set_title("Baseline vs Risk vs Delay")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    # Heatmap for Baseline
    with col2:
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(6,4))
        sns.heatmap(u_baseline * 100, cmap="YlGnBu", cbar_kws={'label': 'Completion %'}, ax=ax_heatmap)
        ax_heatmap.set_xticks(range(0, len(default_tasks), 1))
        ax_heatmap.set_xticklabels(default_tasks, rotation=45, ha="right")
        ax_heatmap.set_yticks(np.arange(0, len(time), int(1/dt)))
        ax_heatmap.set_yticklabels([f"{t:.1f}" for t in time[::int(1/dt)]], rotation=0)
        ax_heatmap.set_title("Baseline Completion Heatmap")
        ax_heatmap.set_xlabel("Tasks")
        ax_heatmap.set_ylabel("Time (weeks)")
        st.pyplot(fig_heatmap, use_container_width=True)

    # Heatmap for Delay
    with col3:
        fig_heatmap_delay, ax_heatmap_delay = plt.subplots(figsize=(6,4))
        sns.heatmap(u_delay * 100, cmap="YlOrRd", cbar_kws={'label': 'Completion %'}, ax=ax_heatmap_delay)
        ax_heatmap_delay.set_xticks(range(0, len(default_tasks), 1))
        ax_heatmap_delay.set_xticklabels(default_tasks, rotation=45, ha="right")
        ax_heatmap_delay.set_yticks(np.arange(0, len(time), int(1/dt)))
        ax_heatmap_delay.set_yticklabels([f"{t:.1f}" for t in time[::int(1/dt)]], rotation=0)
        ax_heatmap_delay.set_title("Delay Scenario Heatmap")
        ax_heatmap_delay.set_xlabel("Tasks")
        ax_heatmap_delay.set_ylabel("Time (weeks)")
        st.pyplot(fig_heatmap_delay, use_container_width=True)

# ----------------------------
# TAB 2: Custom Project Editor
# ----------------------------
with tab2:
    st.subheader("Custom Project Editor (like MS Project Lite)")

    if "task_df" not in st.session_state:
        st.session_state.task_df = pd.DataFrame({
            "ID": list(range(1, len(default_tasks)+1)),
            "Task": default_tasks,
            "Duration (weeks)": default_durations,
            "Dependencies (IDs)": ["" for _ in default_tasks],
            "Risk (0-5)": default_risks
        })

    st.write("Edit tasks below. Use **IDs** in Dependencies, separated by commas (e.g., `1,2`).")
    task_df = st.data_editor(
        st.session_state.task_df,
        num_rows="dynamic",
        use_container_width=True
    )
    st.session_state.task_df = task_df

    # --- Recalculate button ---
    if st.button("🔄 Recalculate Simulation"):
        # Convert table into inputs
        tasks = task_df["Task"].tolist()
        durations = task_df["Duration (weeks)"].to_numpy(dtype=float)
        risks = task_df["Risk (0-5)"].to_numpy(dtype=float)

        num_tasks = len(tasks)
        adjacency = np.zeros((num_tasks, num_tasks))

        for i, dep in enumerate(task_df["Dependencies (IDs)"]):
            if dep:
                dep_list = [d.strip() for d in dep.split(",")]
                for d in dep_list:
                    if d.isdigit():
                        j = int(d) - 1
                        if 0 <= j < num_tasks:
                            adjacency[j, i] = 1

        # Baseline simulation
        u_baseline = run_pde(adjacency, durations, diffusion, np.zeros(num_tasks))
        baseline_curve = u_baseline.mean(axis=0)

        # Risk simulation
        u_risk = run_pde(adjacency, durations, diffusion, risks)
        risk_curve = u_risk.mean(axis=0)

        # Delay simulation (e.g., first task with duration > 0 delayed by 50%)
        delay_durations = durations.copy()
        delay_idx = next((i for i, d in enumerate(durations) if d > 0), 0)
        delay_durations[delay_idx] *= 1.5
        u_delay = run_pde(adjacency, delay_durations, diffusion, np.zeros(num_tasks))
        delay_curve = u_delay.mean(axis=0)

        col1, col2, col3 = st.columns(3)

        # 3D Plot
        with col1:
            fig2 = plt.figure(figsize=(6,4))
            ax2 = fig2.add_subplot(111, projection="3d")
            ax2.plot(time, np.zeros_like(time), baseline_curve, color="blue", lw=2, label="Baseline")
            ax2.plot(time, np.ones_like(time)*0.5, risk_curve, color="red", lw=2, linestyle="--", label="With Risk")
            ax2.plot(time, np.ones_like(time), delay_curve, color="green", lw=2, linestyle=":", label="With Delay")
            ax2.set_xlabel("Time (weeks)")
            ax2.set_ylabel("Scenario")
            ax2.set_yticks([0,0.5,1])
            ax2.set_yticklabels(["Baseline","Risk","Delay"])
            ax2.set_zlabel("Completion (0–1)")
            ax2.set_title("Custom Project Simulation", fontsize=12)
            ax2.legend(fontsize=8)
            st.pyplot(fig2, use_container_width=True)

        # Heatmap for Baseline
        with col2:
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(6,4))
            sns.heatmap(u_baseline * 100, cmap="YlGnBu", cbar_kws={'label': 'Completion %'}, ax=ax_heatmap)
            ax_heatmap.set_xticks(range(0, len(tasks), 1))
            ax_heatmap.set_xticklabels(tasks, rotation=45, ha="right")
            ax_heatmap.set_yticks(np.arange(0, len(time), int(1/dt)))
            ax_heatmap.set_yticklabels([f"{t:.1f}" for t in time[::int(1/dt)]], rotation=0)
            ax_heatmap.set_title("Baseline Completion Heatmap")
            ax_heatmap.set_xlabel("Tasks")
            ax_heatmap.set_ylabel("Time (weeks)")
            st.pyplot(fig_heatmap, use_container_width=True)

        # Heatmap for Delay
        with col3:
            fig_heatmap_delay, ax_heatmap_delay = plt.subplots(figsize=(6,4))
            sns.heatmap(u_delay * 100, cmap="YlOrRd", cbar_kws={'label': 'Completion %'}, ax=ax_heatmap_delay)
            ax_heatmap_delay.set_xticks(range(0, len(tasks), 1))
            ax_heatmap_delay.set_xticklabels(tasks, rotation=45, ha="right")
            ax_heatmap_delay.set_yticks(np.arange(0, len(time), int(1/dt)))
            ax_heatmap_delay.set_yticklabels([f"{t:.1f}" for t in time[::int(1/dt)]], rotation=0)
            ax_heatmap_delay.set_title("Delay Scenario Heatmap")
            ax_heatmap_delay.set_xlabel("Tasks")
            ax_heatmap_delay.set_ylabel("Time (weeks)")
            st.pyplot(fig_heatmap_delay, use_container_width=True)

        # Gantt Chart
        with col1:
            fig_gantt, axg = plt.subplots(figsize=(6,4))
            start_times = np.zeros(num_tasks)
            finish_times = np.zeros(num_tasks)

            for i in range(num_tasks):
                preds = np.where(adjacency[:,i] > 0)[0]
                if len(preds) > 0:
                    start_times[i] = max(finish_times[p] for p in preds)
                finish_times[i] = start_times[i] + durations[i]
                axg.barh(i, durations[i], left=start_times[i], height=0.4,
                         align="center", color="skyblue", edgecolor="black")
                axg.text(start_times[i] + durations[i]/2, i,
                         f"{tasks[i]} ({durations[i]:.0f}w)",
                         ha="center", va="center", fontsize=8)

            for i in range(num_tasks):
                preds = np.where(adjacency[:,i] > 0)[0]
                for p in preds:
                    axg.annotate("",
                        xy=(start_times[i], i),
                        xytext=(finish_times[p], p),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1)
                    )

            axg.set_yticks(range(num_tasks))
            axg.set_yticklabels([f"{i+1}" for i in range(num_tasks)])
            axg.invert_yaxis()
            axg.set_xlabel("Time (weeks)")
            axg.set_ylabel("Task ID")
            axg.set_title("Gantt Chart with Dependencies", fontsize=12)
            st.pyplot(fig_gantt, use_container_width=True)