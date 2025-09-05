import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
from io import BytesIO
import networkx as nx
import seaborn as sns
import json
import os
import time
from matplotlib.animation import FuncAnimation, FFMpegWriter
import tempfile
import shutil

# Custom CSS for Enterprise Look with Deep Navy, Orange Highlights, and Tab Transitions
st.markdown("""
    <style>
    .main { background-color: #1a2a44; padding: 20px; border-radius: 15px; }
    .stButton>button {
        background-color: #f28c38; color: #fff; border-radius: 10px; padding: 10px 20px;
        transition: all 0.3s ease; border: 2px solid #f28c38; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #d8701f; transform: scale(1.05); box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2a4066; color: #fff; font-size: 16px; font-weight: bold; border-radius: 10px 10px 0 0;
        border: 1px solid #f28c38; margin: 2px; transition: opacity 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { background-color: #35548a; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #f28c38; color: #fff; }
    .stSidebar { background-color: #1a2a44; border-right: 1px solid #f28c38; color: #fff; border-radius: 15px 0 0 15px; }
    h1 { color: #f28c38; font-family: 'Arial', sans-serif; font-size: 28px; font-weight: bold; text-shadow: 1px 1px #2a4066; }
    h2, h3 { color: #f28c38; font-family: 'Arial', sans-serif; font-size: 20px; font-weight: bold; text-shadow: 1px 1px #2a4066; }
    .stAlert { border-radius: 10px; background-color: #2a4066; color: #fff; }
    .left-column { background-color: #fff; padding: 10px; border-radius: 10px; color: #1a2a44; }
    .left-column * { color: #1a2a44 !important; }
    .right-column { background-color: #2a4066; padding: 10px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("Ani's Pulse: Project Schedule Simulator")
st.write("**Ready to feel the pulse**")

# Autosave function
def save_task_list(task_df):
    save_path = "task_list.json"
    task_dict = task_df.to_dict(orient="records")
    with open(save_path, "w") as f:
        json.dump(task_dict, f)

def load_task_list():
    save_path = "task_list.json"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            task_dict = json.load(f)
            return pd.DataFrame(task_dict)
    return None

def save_project(task_df, u_matrix, risk_curve, classical_risk):
    save_path = "project_data.json"
    project_data = {
        "task_df": task_df.to_dict(orient="records"),
        "u_matrix": u_matrix.tolist(),
        "risk_curve": risk_curve.tolist(),
        "classical_risk": classical_risk.tolist()
    }
    with open(save_path, "w") as f:
        json.dump(project_data, f)

def export_mpp(task_df, start_times, finish_times):
    mpp_data = {
        "Tasks": task_df["Task"].tolist(),
        "Duration": task_df["Duration (days)"].tolist(),
        "Start": start_times.tolist(),
        "Finish": finish_times.tolist(),
        "Predecessors": task_df["Dependencies (IDs)"].tolist(),
        "Risk": task_df["Risk (0-5)"].tolist(),
        "Parent ID": task_df["Parent ID"].tolist()
    }
    mpp_df = pd.DataFrame(mpp_data)
    return mpp_df.to_csv(index=False)

# ----------------------------
# PDE-Based Project Simulation
# ----------------------------
def reaction(u_i, duration, reaction_multiplier=2.0):
    return reaction_multiplier / duration if duration > 0 else 0

def run_pde(adjacency, durations, diffusion, risk_levels, start_times, parent_ids,
            T=140, dt=0.01, reaction_multiplier=2.0, max_delay=0.05):
    num_tasks = len(durations)
    steps = int(T/dt)
    u = np.zeros((num_tasks, steps+1))
    durations_risk = durations * np.maximum(1.0, risk_levels)

    for t in range(steps):
        du = np.zeros(num_tasks)
        for i in range(num_tasks):
            preds = np.where(adjacency[:,i] > 0)[0]
            elapsed_time = t * dt - start_times[i]
            if elapsed_time < 0:
                continue

            base_progress = reaction(u[i,t], durations_risk[i], reaction_multiplier) if u[i,t] < 1.0 else 0
            delay = 0
            if len(preds) > 0:
                avg_pred_delay = diffusion * np.mean([1.0 - u[p,t] for p in preds])
                delay = min(avg_pred_delay, max_delay)

            # Roll-up progress for summary tasks
            if parent_ids[i] == 0 and u[i,t] < 1.0:  # Summary task
                subtasks = [j for j in range(num_tasks) if parent_ids[j] == i + 1]
                if subtasks:
                    subtask_durations = [durations_risk[j] for j in subtasks]
                    total_duration = sum(subtask_durations)
                    weighted_progress = sum(u[j,t] * (durations_risk[j] / total_duration) for j in subtasks)
                    base_progress = min(base_progress, weighted_progress)

            net_progress = base_progress - delay
            max_progress = min(1.0, elapsed_time / durations_risk[i])
            if net_progress > 0 or u[i,t] >= max_progress - 0.01:
                du[i] += net_progress

        u[:,t+1] = np.clip(u[:,t] + du * dt, 0, 1.0)

    for i in range(num_tasks):
        finish_time = start_times[i] + durations_risk[i]
        finish_step = int(finish_time / dt)
        if finish_step < steps:
            transition_steps = int(0.1 / dt)
            for t in range(finish_step, min(finish_step + transition_steps, steps)):
                u[i,t] = min(1.0, u[i,t] + (1.0 - u[i,t]) * (t - finish_step + 1) / transition_steps)
            u[i, finish_step + transition_steps:] = 1.0

    return u

# ----------------------------
# Classical Completion Function
# ----------------------------
def compute_classical_completion(start_times, durations, risk_levels=None, parent_ids=None, simulation_time=None):
    if risk_levels is None:
        durations_risk = np.array(durations)
    else:
        durations_risk = np.array(durations) * np.maximum(1.0, np.array(risk_levels))

    num_tasks = len(durations_risk)
    if simulation_time is None:
        max_time = np.max(start_times + durations_risk) * 1.1
        simulation_time = np.linspace(0, max_time, int(max_time*100))

    u = np.zeros((num_tasks, len(simulation_time)))

    for i in range(num_tasks):
        for j, t in enumerate(simulation_time):
            if t < start_times[i]:
                u[i, j] = 0
            elif t < start_times[i] + durations_risk[i]:
                if parent_ids[i] == 0:  # Summary task
                    subtasks = [j for j in range(num_tasks) if parent_ids[j] == i + 1]
                    if subtasks:
                        subtask_durations = [durations_risk[j] for j in subtasks]
                        total_duration = sum(subtask_durations)
                        weighted_progress = sum(u[j,k] * (durations_risk[j] / total_duration) for j in subtasks for k in range(j) if k <= j and t >= start_times[j])
                        u[i, j] = min((t - start_times[i]) / durations_risk[i], weighted_progress)
                    else:
                        u[i, j] = (t - start_times[i]) / durations_risk[i]
                else:
                    u[i, j] = (t - start_times[i]) / durations_risk[i]
            else:
                u[i, j] = 1

    return u, simulation_time

# ----------------------------
# MPP Import Function
# ----------------------------
def import_mpp(file):
    if mppx is None:
        return None, "MPP import requires 'python-mppx' library. Please install it or use manual entry."
    try:
        project = mppx.read(file)
        tasks = []
        durations = []
        dependencies = []
        risks = []
        parent_ids = []
        for task in project.tasks:
            tasks.append(task.name or f"Task {task.id}")
            durations.append(task.duration or 7.0)
            deps = ",".join(str(dep.id) for dep in task.predecessors) if task.predecessors else ""
            dependencies.append(deps)
            risks.append(0.0 if task.is_summary else task.risk or 0.0)
            parent_ids.append(task.parent_id or 0)
        df = pd.DataFrame({
            "Select": [False] * len(tasks),
            "ID": list(range(1, len(tasks)+1)),
            "Task": tasks,
            "Duration (days)": durations,
            "Dependencies (IDs)": dependencies,
            "Risk (0-5)": risks,
            "Parent ID": parent_ids
        })
        return df, None
    except Exception as e:
        return None, f"Error importing .mpp file: {str(e)}"

# ----------------------------
# Defaults
# ----------------------------
default_tasks = ["Project Phase 1", "Requirements", "Database Design", "Project Phase 2", "Backend API", "Testing"]
default_durations = np.array([0, 14, 21, 0, 28, 14], dtype=float)  # 0 for summary tasks
default_risks = np.array([0, 0, 0, 0, 1, 0], dtype=float)  # Risk only for subtasks
default_adjacency = np.array([
    [0,1,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0],
    [0,0,0,0,1,0],
    [0,0,0,0,0,0],
    [0,0,0,0,0,0]
], dtype=float)
default_deps = ["", "", "", "", "1", "5"]
default_parent_ids = [0, 1, 1, 0, 4, 4]  # Phase 1 (1,2), Phase 2 (4,5)

dt = 0.01
T = 150

# Initialize session state
if "simulation_data" not in st.session_state:
    st.session_state.simulation_data = {"tasks": None, "adjacency": None, "u_matrix": None, "num_tasks": None}
if "task_df" not in st.session_state:
    loaded_df = load_task_list()
    st.session_state.task_df = loaded_df if loaded_df is not None else pd.DataFrame({
        "Select": [False] * len(default_tasks),
        "ID": list(range(1, len(default_tasks)+1)),
        "Task": default_tasks,
        "Duration (days)": default_durations,
        "Dependencies (IDs)": default_deps,
        "Risk (0-5)": default_risks,
        "Parent ID": default_parent_ids
    })
if "diffusion" not in st.session_state:
    st.session_state.diffusion = 0.001
if "reaction_multiplier" not in st.session_state:
    st.session_state.reaction_multiplier = 2.0
if "max_delay" not in st.session_state:
    st.session_state.max_delay = 0.05

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Editor & Results", "🔗 Task Dependencies", "🔮 Eigenvalue Analysis", "📊 Completion Heatmap"])

with tab1:
    with st.sidebar.expander("⚙️ Simulation Parameters", expanded=True):
        st.subheader("Simulation Parameters")
        diffusion = st.slider("Diffusion Coefficient", 0.001, 0.05, st.session_state.diffusion, 0.001, help="Finely tune delay propagation (0.001–0.05).")
        reaction_multiplier = st.slider("Reaction Multiplier", 1.0, 5.0, st.session_state.reaction_multiplier, 0.1, help="Scales the base reaction rate (1.0–5.0).")
        max_delay = st.slider("Max Delay", 0.01, 0.1, st.session_state.max_delay, 0.01, help="Max delay from predecessors (0.01–0.1).")
        col_diff1, col_diff2, col_diff3 = st.columns(3)
        with col_diff1:
            if st.button("Low (0.01)"):
                st.session_state.diffusion = 0.01
                st.rerun()
        with col_diff2:
            if st.button("Medium (0.025)"):
                st.session_state.diffusion = 0.025
                st.rerun()
        with col_diff3:
            if st.button("High (0.05)"):
                st.session_state.diffusion = 0.05
                st.rerun()
        st.session_state.diffusion = diffusion
        st.session_state.reaction_multiplier = reaction_multiplier
        st.session_state.max_delay = max_delay
        st.write("**Diffusion Factor**: Controls delay propagation from risk points (0.001–0.05).")
        st.write("**Reaction Multiplier**: Scales the base progress rate (1.0–5.0).")
        st.write("**Max Delay**: Caps delay from predecessors (0.01–0.1).")
        st.write("**Risk**: Values (0–5) multiply subtask durations; summary tasks inherit from subtasks.")

    st.subheader("Project Editor")
    st.info("Edit tasks or import an .mpp file. Use **Task IDs** in Dependencies and Parent ID (0 for top-level). Update dependencies after changes.")

    st.write("📥 **Import .mpp File**")
    mpp_file = st.file_uploader("Upload Microsoft Project (.mpp) file", type=["mpp"])
    if mpp_file is not None:
        df, error = import_mpp(mpp_file)
        if error:
            st.error(error)
        else:
            st.session_state.task_df = df
            save_task_list(df)
            st.success("MPP file imported successfully.")
            st.rerun()

    task_df = st.data_editor(
        st.session_state.task_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", help="Select rows to delete"),
            "ID": st.column_config.NumberColumn(disabled=True),
            "Task": st.column_config.TextColumn(required=True),
            "Duration (days)": st.column_config.NumberColumn(min_value=0.0, step=1.0, help="0 for summary tasks"),
            "Dependencies (IDs)": st.column_config.TextColumn(),
            "Risk (0-5)": st.column_config.NumberColumn(min_value=0, max_value=5, step=0.1, help="0 for summary tasks"),
            "Parent ID": st.column_config.NumberColumn(
                min_value=0,
                max_value=len(st.session_state.task_df),
                step=1,
                help="ID of parent task (0 for top-level)"
            )
        },
        on_change=save_task_list, args=(st.session_state.task_df,)
    )
    st.session_state.task_df = task_df

    col_add, col_delete = st.columns(2)
    with col_add:
        if st.button("➕ Add New Row"):
            new_id = len(st.session_state.task_df) + 1
            new_row = {
                "Select": False,
                "ID": new_id,
                "Task": "",
                "Duration (days)": 7.0,
                "Dependencies (IDs)": "",
                "Risk (0-5)": 0.0,
                "Parent ID": 0
            }
            st.session_state.task_df = pd.concat([st.session_state.task_df, pd.DataFrame([new_row])], ignore_index=True)
            save_task_list(st.session_state.task_df)
            st.rerun()
    with col_delete:
        if st.button("🗑️ Delete Selected Rows"):
            selected_indices = st.session_state.task_df[st.session_state.task_df["Select"] == True].index
            if not selected_indices.empty:
                st.session_state.task_df = st.session_state.task_df.drop(selected_indices).reset_index(drop=True)
                st.session_state.task_df["ID"] = range(1, len(st.session_state.task_df) + 1)
                save_task_list(st.session_state.task_df)
                st.rerun()

    if st.button("🔄 Recalculate Simulation", type="primary"):
        valid = True
        tasks = task_df["Task"].tolist()
        num_tasks = len(tasks)
        st.session_state.simulation_data["num_tasks"] = num_tasks
        parent_ids = task_df["Parent ID"].tolist()
        # Validate hierarchy
        for i, parent_id in enumerate(parent_ids):
            if parent_id > 0:
                if parent_id > num_tasks or parent_id == i + 1:
                    valid = False
                    st.error(f"Invalid Parent ID for task {i+1}: {parent_id} is not valid or self-referencing.")
                current = i
                visited = set()
                while parent_ids[current] > 0:
                    if parent_ids[current] - 1 in visited:
                        valid = False
                        st.error(f"Cyclic hierarchy detected involving task {i+1}.")
                        break
                    visited.add(current)
                    current = parent_ids[current] - 1
        # Validate dependencies and durations
        for i, dep in enumerate(task_df["Dependencies (IDs)"]):
            if dep:
                dep_list = [d.strip() for d in dep.split(",")]
                for d in dep_list:
                    if not d.isdigit() or int(d) < 1 or int(d) > num_tasks or int(d) == i + 1:
                        valid = False
                        st.error(f"Invalid dependency for task {i+1}: '{d}'.")
            if task_df["Duration (days)"][i] < 0:
                valid = False
                st.error(f"Invalid duration for task {i+1}: Duration must be >= 0.")
        if valid:
            progress_bar = st.progress(0)
            durations = task_df["Duration (days)"].to_numpy(dtype=float)
            risks = task_df["Risk (0-5)"].to_numpy(dtype=float)
            # Set duration and risk to 0 for summary tasks
            for i in range(num_tasks):
                if parent_ids[i] > 0:  # Subtask
                    parent_idx = parent_ids[i] - 1
                    durations[parent_idx] = 0
                    risks[parent_idx] = 0
            adjacency = np.zeros((num_tasks, num_tasks))
            for i, dep in enumerate(task_df["Dependencies (IDs)"]):
                if dep:
                    dep_list = [d.strip() for d in dep.split(",")]
                    for d in dep_list:
                        j = int(d) - 1
                        if 0 <= j < num_tasks:
                            adjacency[j, i] = 1
            st.session_state.simulation_data["tasks"] = tasks
            st.session_state.simulation_data["adjacency"] = adjacency

            start_times_risk = np.zeros(num_tasks)
            finish_times_risk = np.zeros(num_tasks)
            durations_risk = durations * np.maximum(1.0, risks)
            for i in range(num_tasks):
                preds = np.where(adjacency[:, i] > 0)[0]
                if len(preds) > 0:
                    start_times_risk[i] = max(finish_times_risk[p] for p in preds)
                if parent_ids[i] == 0:  # Summary task
                    subtasks = [j for j in range(num_tasks) if parent_ids[j] == i + 1]
                    if subtasks:
                        finish_times_risk[i] = max(finish_times_risk[j] for j in subtasks)
                else:
                    finish_times_risk[i] = start_times_risk[i] + durations_risk[i]
            T = int(np.max(finish_times_risk) + 10)
            simulation_time = np.linspace(0, T, int(T/dt)+1)

            u_risk = run_pde(adjacency, durations, st.session_state.diffusion, risks, start_times_risk, parent_ids, T, dt,
                           st.session_state.reaction_multiplier, st.session_state.max_delay)
            risk_curve = u_risk.mean(axis=0)
            st.session_state.simulation_data["u_matrix"] = u_risk
            progress_bar.progress(50)

            classical_u, sim_time = compute_classical_completion(start_times_risk, durations, risks, parent_ids, simulation_time)
            classical_risk = classical_u.mean(axis=0)
            progress_bar.progress(100)

            classical_completion_time = simulation_time[np.argmax(classical_risk >= 0.99)] if np.max(classical_risk) >= 0.99 else T
            pde_completion_time = simulation_time[np.argmax(risk_curve >= 0.99)] if np.max(risk_curve) >= 0.99 else T

            st.subheader("Simulation Results")
            st.write(f"**Time to Completion:** Classical: {classical_completion_time:.1f} days, PDE: {pde_completion_time:.1f} days")

            col1, col2 = st.columns([1, 1])
            with col1.container():
                fig, ax = plt.subplots(figsize=(6,4), facecolor='#fff')
                ax.plot(simulation_time, risk_curve, color="#d32f2f", lw=2, label="Diffusion Risk")
                ax.plot(simulation_time, classical_risk, color="#1976d2", lw=2, label="Classical Risk")
                ax.set_xlabel("Time (days)")
                ax.set_ylabel("Average Completion (0–1)")
                ax.set_title("Completion: Diffusion vs Classical (Risk)")
                ax.legend(frameon=True, facecolor='#fff', edgecolor='#f28c38', loc='upper left')
                ax.grid(True, linestyle="--", alpha=0.7, color='#2a4066')
                ax.set_facecolor('#fff')
                st.pyplot(fig, use_container_width=True)

            with col2.container():
                st.markdown('<div class="right-column">', unsafe_allow_html=True)
                fig_gantt_classical, axg_classical = plt.subplots(figsize=(6,4), facecolor='#fff')
                colors = plt.cm.Oranges(task_df["Risk (0-5)"].to_numpy() / 5.0)
                for i in range(num_tasks):
                    is_summary = parent_ids[i] == 0
                    indent = 0.2 * (len([p for p in parent_ids[:i] if p > 0 and parent_ids.index(p) < i]) if not is_summary else 0)
                    bar = axg_classical.barh(i + indent, durations_risk[i], left=start_times_risk[i], height=0.4,
                                          align="center", color=colors[i], edgecolor="#1a2a44", alpha=0.9,
                                          hatch='/' if not is_summary else None)
                    axg_classical.text(start_times_risk[i] + durations_risk[i]/2, i + indent,
                                     f"{tasks[i]} ({durations_risk[i]:.0f}d)" if not is_summary else f"**{tasks[i]}**",
                                     ha="center", va="center", fontsize=8, fontweight='bold' if is_summary else 'normal',
                                     fontfamily='Arial', color='#1a2a44')
                    mplcursors.cursor(bar, hover=True).connect("add", lambda sel, idx=i: sel.annotation.set_text(
                        f"Task: {tasks[idx]}\nDuration: {durations_risk[idx]:.1f} days\nRisk: {task_df['Risk (0-5)'][idx]:.1f}"
                    ).set_alpha(0.0).set_alpha(1.0, duration=0.5))
                for i in range(num_tasks):
                    preds = np.where(adjacency[:,i] > 0)[0]
                    for p in preds:
                        axg_classical.annotate("", xy=(start_times_risk[i], i + 0.2 * (parent_ids[i] > 0)),
                                             xytext=(finish_times_risk[p], p + 0.2 * (parent_ids[p] > 0)),
                                             arrowprops=dict(arrowstyle="->", color="#1a2a44", lw=1.5))
                axg_classical.set_yticks(range(num_tasks))
                axg_classical.set_yticklabels([f"{i+1}" for i in range(num_tasks)], fontsize=10, fontfamily='Arial')
                axg_classical.invert_yaxis()
                axg_classical.set_xlabel("Time (days)")
                axg_classical.set_ylabel("Task ID")
                axg_classical.set_title("Gantt Chart (Classical)")
                axg_classical.grid(True, axis="x", linestyle="--", alpha=0.7, color='#2a4066')
                axg_classical.set_facecolor('#fff')
                st.pyplot(fig_gantt_classical, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="right-column">', unsafe_allow_html=True)
            fig_gantt_pde, axg_pde = plt.subplots(figsize=(6,4), facecolor='#fff')
            start_times_pde = np.zeros(num_tasks)
            finish_times_pde = np.zeros(num_tasks)
            for i in range(num_tasks):
                preds = np.where(adjacency[:,i] > 0)[0]
                if len(preds) > 0:
                    start_times_pde[i] = max(finish_times_pde[p] for p in preds)
                if parent_ids[i] == 0:
                    subtasks = [j for j in range(num_tasks) if parent_ids[j] == i + 1]
                    if subtasks:
                        finish_times_pde[i] = max(finish_times_pde[j] for j in subtasks)
                else:
                    finish_times_pde[i] = start_times_pde[i] + durations_risk[i]
                is_summary = parent_ids[i] == 0
                indent = 0.2 * (len([p for p in parent_ids[:i] if p > 0 and parent_ids.index(p) < i]) if not is_summary else 0)
                bar = axg_pde.barh(i + indent, durations_risk[i], left=start_times_pde[i], height=0.4,
                                align="center", color=colors[i], edgecolor="#1a2a44", alpha=0.9,
                                hatch='/' if not is_summary else None)
                axg_pde.text(start_times_pde[i] + durations_risk[i]/2, i + indent,
                           f"{tasks[i]} ({durations_risk[i]:.0f}d)" if not is_summary else f"**{tasks[i]}**",
                           ha="center", va="center", fontsize=8, fontweight='bold' if is_summary else 'normal',
                           fontfamily='Arial', color='#1a2a44')
            for i in range(num_tasks):
                preds = np.where(adjacency[:,i] > 0)[0]
                for p in preds:
                    axg_pde.annotate("", xy=(start_times_pde[i], i + 0.2 * (parent_ids[i] > 0)),
                                   xytext=(finish_times_pde[p], p + 0.2 * (parent_ids[p] > 0)),
                                   arrowprops=dict(arrowstyle="->", color="#1a2a44", lw=1.5))
            axg_pde.set_yticks(range(num_tasks))
            axg_pde.set_yticklabels([f"{i+1}" for i in range(num_tasks)], fontsize=10, fontfamily='Arial')
            axg_pde.invert_yaxis()
            axg_pde.set_xlabel("Time (days)")
            axg_pde.set_ylabel("Task ID")
            axg_pde.set_title("Gantt Chart (PDE)")
            axg_pde.grid(True, axis="x", linestyle="--", alpha=0.7, color='#2a4066')
            axg_pde.set_facecolor('#fff')
            st.pyplot(fig_gantt_pde, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.subheader("3D Completion Plot")
            fig_3d = plt.figure(figsize=(8,6), facecolor='#fff')
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            ax_3d.plot(simulation_time, [0]*len(simulation_time), risk_curve, color="#d32f2f", lw=2, label="Diffusion Risk")
            ax_3d.plot(simulation_time, [1]*len(simulation_time), classical_risk, color="#1976d2", lw=2, label="Classical Risk")
            ax_3d.set_xlabel("Time (days)")
            ax_3d.set_ylabel("Model (0=PDE, 1=Classical)")
            ax_3d.set_zlabel("Average Completion (0–1)")
            ax_3d.set_title("3D Completion: Diffusion vs Classical (Risk)")
            ax_3d.legend(frameon=True, facecolor='#fff', edgecolor='#f28c38')
            ax_3d.set_facecolor('#fff')
            ax_3d.grid(True, color='#2a4066', alpha=0.7)
            st.pyplot(fig_3d, use_container_width=True)

            col_export1, col_export2, col_export3 = st.columns(3)
            with col_export1:
                csv = st.session_state.task_df.to_csv(index=False)
                st.download_button("📄 Download Task Data (CSV)", csv, "project_tasks.csv", "text/csv")
            with col_export2:
                buf = BytesIO()
                fig_gantt_pde.savefig(buf, format="png", bbox_inches="tight")
                st.download_button("🖼️ Download PDE Gantt Chart (PNG)", buf.getvalue(), "gantt_chart_pde.png", "image/png")
            with col_export3:
                mpp_data = export_mpp(task_df, start_times_risk, finish_times_risk)
                st.download_button("📤 Export as .mpp (CSV)", mpp_data, "project_tasks.mpp.csv", "text/csv")

            if st.button("💾 Save Project"):
                save_project(task_df, u_matrix, risk_curve, classical_risk)

with tab2:
    st.subheader("Task Dependency Diagram")
    if st.session_state.simulation_data["tasks"] is None:
        st.info("Please run the simulation in the 'Editor & Results' tab.")
    else:
        tasks = st.session_state.simulation_data["tasks"]
        adjacency = st.session_state.simulation_data["adjacency"]
        parent_ids = task_df["Parent ID"].tolist()
        G = nx.DiGraph()
        for i, task in enumerate(tasks):
            G.add_node(i, label=f"{i+1}: {task}")
        for i in range(len(tasks)):
            preds = np.where(adjacency[:,i] > 0)[0]
            for p in preds:
                edge_style = '--' if parent_ids[p] == i + 1 or parent_ids[i] == p + 1 else '-'
                G.add_edge(p, i, style=edge_style)
        fig_dep, ax_dep = plt.subplots(figsize=(6,4), facecolor='#2a4066')
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
                node_color='#f28c38', node_size=800, font_size=9, font_family='Arial',
                font_weight='bold', edge_color='#1a2a44', arrows=True, arrowsize=15,
                style=nx.get_edge_attributes(G, 'style'))
        ax_dep.set_title("Task Dependency Graph")
        ax_dep.set_facecolor('#2a4066')
        st.pyplot(fig_dep, use_container_width=True)

with tab3:
    st.subheader("Eigenvalue Analysis")
    if st.session_state.simulation_data["tasks"] is None:
        st.info("Please run the simulation in the 'Editor & Results' tab.")
    else:
        adjacency = st.session_state.simulation_data["adjacency"]
        num_tasks = st.session_state.simulation_data["num_tasks"]
        eigenvalues, _ = np.linalg.eig(adjacency.astype(float))
        degree = np.sum(adjacency, axis=1)
        D = np.diag(degree)
        laplacian = D - adjacency
        eigvals_lap, _ = np.linalg.eig(laplacian.astype(float))
        second_eigenvalue = np.sort(eigvals_lap)[1] if len(eigvals_lap) > 1 else 0
        st.write("### Eigenvalues of Adjacency Matrix")
        df_eigen = pd.DataFrame({"Eigenvalue": np.real(eigenvalues), "Imaginary": np.imag(eigenvalues)})
        st.table(df_eigen)
        st.write(f"### Second Eigenvalue (Fiedler Value): {second_eigenvalue:.4f}")
        st.write("Note: Indicates graph connectivity.")

        fig_bar, ax_bar = plt.subplots(figsize=(6,4), facecolor='#fff')
        ax_bar.bar(range(len(np.real(eigenvalues))), np.real(eigenvalues), color='#f28c38', edgecolor='#1a2a44')
        ax_bar.set_xlabel("Task Index")
        ax_bar.set_ylabel("Real Eigenvalue")
        ax_bar.set_title("Real Part of Eigenvalues")
        ax_bar.grid(True, linestyle="--", alpha=0.7, color='#2a4066')
        ax_bar.set_facecolor('#fff')
        st.pyplot(fig_bar, use_container_width=True)

        df_connectivity = pd.DataFrame(adjacency, index=[f"Task {i+1}" for i in range(num_tasks)],
                                      columns=[f"Task {i+1}" for i in range(num_tasks)])
        fig_heat, ax_heat = plt.subplots(figsize=(6,4), facecolor='#fff')
        sns.heatmap(df_connectivity, annot=True, cmap="Oranges", cbar_kws={'label': 'Dependency Strength'}, ax=ax_heat)
        ax_heat.set_xlabel("Task ID")
        ax_heat.set_ylabel("Task ID")
        ax_heat.set_title("Adjacency Matrix Heatmap")
        st.pyplot(fig_heat, use_container_width=True)

with tab4:
    st.subheader("Completion Heatmap")
    if st.session_state.simulation_data["tasks"] is None:
        st.info("Please run the simulation in the 'Editor & Results' tab.")
    else:
        tasks = st.session_state.simulation_data["tasks"]
        u_risk = st.session_state.simulation_data["u_matrix"]
        classical_u, sim_time = compute_classical_completion(start_times_risk, durations, risks, parent_ids, simulation_time)

        if shutil.which('ffmpeg') is None:
            st.error("FFmpeg is not installed. Please install it to generate animations. On macOS, use `brew install ffmpeg` or visit https://ffmpeg.org/download.html.")
        else:
            st.subheader("PDE Completion Animation")
            fig_pde, ax_pde = plt.subplots(figsize=(10, 6), facecolor='#fff')
            heatmap_pde = ax_pde.imshow(u_risk[:, :1], cmap="Oranges", aspect='auto', vmin=0, vmax=1)
            ax_pde.set_xticks(np.arange(0, len(simulation_time), int(len(simulation_time)/10)))
            ax_pde.set_xticklabels([f"{x:.1f}" for x in simulation_time[::int(len(simulation_time)/10)]])
            ax_pde.set_yticks(range(num_tasks))
            ax_pde.set_yticklabels([f"{i+1}: {tasks[i]}" for i in range(num_tasks)])
            ax_pde.set_xlabel("Time (days)")
            ax_pde.set_ylabel("Tasks")
            ax_pde.set_title("PDE Task Completion Progression")
            plt.colorbar(heatmap_pde, ax=ax_pde, label="Completion %")

            def update(frame):
                heatmap_pde.set_array(u_risk[:, :frame+1])
                return [heatmap_pde]

            ani_pde = FuncAnimation(fig_pde, update, frames=range(1, u_risk.shape[1]), interval=50, blit=True)
            writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                ani_pde.save(tmp_file.name, writer=writer, dpi=100)
                tmp_file.seek(0)
                with open(tmp_file.name, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)

            st.subheader("Classical Completion Animation")
            fig_classical, ax_classical = plt.subplots(figsize=(10, 6), facecolor='#fff')
            heatmap_classical = ax_classical.imshow(classical_u[:, :1], cmap="Oranges", aspect='auto', vmin=0, vmax=1)
            ax_classical.set_xticks(np.arange(0, len(simulation_time), int(len(simulation_time)/10)))
            ax_classical.set_xticklabels([f"{x:.1f}" for x in simulation_time[::int(len(simulation_time)/10)]])
            ax_classical.set_yticks(range(num_tasks))
            ax_classical.set_yticklabels([f"{i+1}: {tasks[i]}" for i in range(num_tasks)])
            ax_classical.set_xlabel("Time (days)")
            ax_classical.set_ylabel("Tasks")
            ax_classical.set_title("Classical Task Completion Progression")
            plt.colorbar(heatmap_classical, ax=ax_classical, label="Completion %")

            def update(frame):
                heatmap_classical.set_array(classical_u[:, :frame+1])
                return [heatmap_classical]

            ani_classical = FuncAnimation(fig_classical, update, frames=range(1, classical_u.shape[1]), interval=50, blit=True)
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                ani_classical.save(tmp_file.name, writer=writer, dpi=100)
                tmp_file.seek(0)
                with open(tmp_file.name, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
