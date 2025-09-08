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
        "Risk": task_df["Risk (0-5)"].tolist()
    }
    mpp_df = pd.DataFrame(mpp_data)
    return mpp_df.to_csv(index=False)

# PDE-Based Project Simulation
def reaction(u_i, duration, reaction_multiplier=2.0):
    return reaction_multiplier / duration if duration > 0 else 0

def run_pde(adjacency, durations, diffusion, risk_levels, start_times,
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

# Classical Completion Function
def compute_classical_completion(start_times, durations, risk_levels=None, simulation_time=None):
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
                u[i, j] = (t - start_times[i]) / durations_risk[i]
            else:
                u[i, j] = 1
    return u, simulation_time

# MPP Import Function
def import_mpp(file):
    try:
        import mppx
        project = mppx.read(file)
        tasks = []
        durations = []
        dependencies = []
        risks = []
        for task in project.tasks:
            tasks.append(task.name or f"Task {task.id}")
            durations.append(task.duration or 7.0)
            deps = ",".join(str(dep.id) for dep in task.predecessors) if task.predecessors else ""
            dependencies.append(deps)
            risks.append(0.0)
        df = pd.DataFrame({
            "Select": [False] * len(tasks),
            "ID": list(range(1, len(tasks)+1)),
            "Task": tasks,
            "Duration (days)": durations,
            "Dependencies (IDs)": dependencies,
            "Risk (0-5)": risks
        })
        return df, None
    except ImportError:
        return None, "MPP import requires 'python-mppx' library. Please install it or use manual entry."
    except Exception as e:
        return None, f"Error importing .mpp file: {str(e)}"

# Defaults
default_tasks = ["Requirements", "Database Design", "Backend API",
                 "Third-party Integration", "Frontend UI", "Testing & Deployment"]
default_durations = np.array([14, 21, 28, 21, 35, 14], dtype=float)
default_risks = np.array([0, 0, 1, 0, 2, 0], dtype=float)
default_adjacency = np.array([
    [0,1,0,0,0,0],
    [0,0,1,0,1,0],
    [0,0,0,1,0,0],
    [0,0,0,0,0,1],
    [0,0,0,0,0,1],
    [0,0,0,0,0,0]
], dtype=float)
default_deps = ["", "1", "2", "3", "2", "4,5"]

# Dynamically set T based on classical completion
dt = 0.01
T = 150  # Initial guess, will be updated dynamically

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
        "Risk (0-5)": default_risks
    })
if "diffusion" not in st.session_state:
    st.session_state.diffusion = 0.001
if "reaction_multiplier" not in st.session_state:
    st.session_state.reaction_multiplier = 2.0
if "max_delay" not in st.session_state:
    st.session_state.max_delay = 0.05

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Editor & Results", "🔗 Task Dependencies", "📊 Classical Gantt", "📊 PDE Gantt", "🔮 Eigenvalue Analysis"])

with tab1:
    # Sidebar
    with st.sidebar.expander("⚙️ Simulation Parameters", expanded=True):
        st.subheader("Simulation Parameters")
        diffusion = st.slider("Diffusion Coefficient", 0.001, 0.05, st.session_state.diffusion, 0.001,
                              help="Finely tune delay propagation (0.001–0.05). Higher values increase delay from risk points.")
        reaction_multiplier = st.slider("Reaction Multiplier", 1.0, 5.0, st.session_state.reaction_multiplier, 0.1,
                                        help="Scales the base reaction rate (1.0–5.0). Higher values speed up completion.")
        max_delay = st.slider("Max Delay", 0.01, 0.1, st.session_state.max_delay, 0.01,
                              help="Max delay from predecessors (0.01–0.1). Higher values increase delay impact.")
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
        st.write("**Diffusion Factor**: Controls delay propagation from risk points (0.001–0.05), decreasing completion rate.")
        st.write("**Reaction Multiplier**: Scales the base progress rate (1.0–5.0).")
        st.write("**Max Delay**: Caps delay from predecessors (0.01–0.1).")
        st.write("**Risk**: Values (0–5) directly multiply task durations (e.g., risk=2 means duration * 2).")

    # Project Editor
    st.subheader("Project Editor")
    st.info("Edit tasks or import an .mpp file. Use **Task IDs** in Dependencies (e.g., `1,2`). Update dependencies after adding/deleting rows.")

    # .mpp File Upload
    mpp_file = st.file_uploader("Upload Microsoft Project (.mpp) file", type=["mpp"])
    if mpp_file is not None:
        df, error = import_mpp(mpp_file)
        if error:
            st.error(error)
        else:
            st.session_state.task_df = df
            save_task_list(df)
            st.success("MPP file imported successfully. Edit tasks below if needed.")
            st.rerun()

    # Data editor
    task_df = st.data_editor(
        st.session_state.task_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", help="Select rows to delete"),
            "ID": st.column_config.NumberColumn(disabled=True),
            "Task": st.column_config.TextColumn(required=True),
            "Duration (days)": st.column_config.NumberColumn(min_value=1.0, step=1.0),
            "Dependencies (IDs)": st.column_config.TextColumn(),
            "Risk (0-5)": st.column_config.NumberColumn(min_value=0, max_value=5, step=0.1)
        },
        on_change=save_task_list, args=(st.session_state.task_df,)
    )
    st.session_state.task_df = task_df

    # Add/Delete Rows
    col_add, col_delete = st.columns(2)
    with col_add:
        if st.button("➕ Add New Row"):
            new_id = len(st.session_state.task_df) + 1
            new_row = {"Select": False, "ID": new_id, "Task": "", "Duration (days)": 7.0, "Dependencies (IDs)": "", "Risk (0-5)": 0.0}
            st.session_state.task_df = pd.concat([st.session_state.task_df, pd.DataFrame([new_row])], ignore_index=True)
            save_task_list(st.session_state.task_df)
            st.rerun()
    with col_delete:
        if st.button("🗑️ Delete Selected Rows"):
            selected_indices = st.session_state.task_df[st.session_state.task_df["Select"] == True].index
            if not selected_indices.empty:
                st.session_state.task_df = st.session_state.task_df.drop(selected_indices).reset_index(drop=True)
                st.session_state.task_df["ID"] = pd.Series(range(1, len(st.session_state.task_df) + 1))
                save_task_list(st.session_state.task_df)
                st.rerun()

    # Recalculate Simulation
    if st.button("🔄 Recalculate Simulation", type="primary"):
        valid = True
        tasks = task_df["Task"].tolist()
        num_tasks = len(tasks)
        st.session_state.simulation_data["num_tasks"] = num_tasks

        # Validate dependencies and durations
        for i, dep in enumerate(task_df["Dependencies (IDs)"]):
            if dep:
                dep_list = [d.strip() for d in dep.split(",")]
                for d in dep_list:
                    if not d.isdigit() or int(d) < 1 or int(d) > num_tasks or int(d) == i+1:
                        valid = False
                        st.error(f"Invalid dependency for task {i+1}: '{d}'")
                        break
            if task_df["Duration (days)"][i] <= 0:
                valid = False
                st.error(f"Invalid duration for task {i+1}")
        if not valid:
            st.stop()

        progress_bar = st.progress(0)

        # Arrays
        durations = task_df["Duration (days)"].to_numpy(dtype=float)
        risks = task_df["Risk (0-5)"].to_numpy(dtype=float)
        adjacency = np.zeros((num_tasks, num_tasks))
        for i, dep in enumerate(task_df["Dependencies (IDs)"]):
            if dep:
                for d in dep.split(","):
                    try:
                        j = int(d.strip()) - 1
                        adjacency[j, i] = 1
                    except:
                        st.error(f"Invalid dependency ID '{d}' for task {i+1}")
                        valid = False
        if not valid:
            st.stop()

        st.session_state.simulation_data["tasks"] = tasks
        st.session_state.simulation_data["adjacency"] = adjacency

        # Compute start/finish times and risk-adjusted durations
        start_times_risk = np.zeros(num_tasks)
        finish_times_risk = np.zeros(num_tasks)
        durations_risk = durations * np.maximum(1.0, risks)
        for i in range(num_tasks):
            preds = np.where(adjacency[:, i] > 0)[0]
            if len(preds) > 0:
                start_times_risk[i] = max(finish_times_risk[p] for p in preds)
            finish_times_risk[i] = start_times_risk[i] + durations_risk[i]

        # Save in session_state for Gantt tabs
        st.session_state.simulation_data["start_times_risk"] = start_times_risk
        st.session_state.simulation_data["finish_times_risk"] = finish_times_risk
        st.session_state.simulation_data["durations_risk"] = durations_risk

        # Time array
        T = int(np.max(finish_times_risk) + 10)
        simulation_time = np.linspace(0, T, int(T/dt)+1)

        # PDE Simulation
        u_risk = run_pde(adjacency, durations, st.session_state.diffusion, risks, start_times_risk, T, dt,
                         st.session_state.reaction_multiplier, st.session_state.max_delay)
        risk_curve = u_risk.mean(axis=0)
        st.session_state.simulation_data["u_matrix"] = u_risk
        progress_bar.progress(50)

        # Classical Schedule
        classical_u, sim_time = compute_classical_completion(start_times_risk, durations, risks, simulation_time)
        classical_risk = classical_u.mean(axis=0)
        progress_bar.progress(100)

        # Completion Times
        classical_completion_time = simulation_time[np.argmax(classical_risk >= 0.99)] if np.max(classical_risk) >= 0.99 else T
        pde_completion_time = simulation_time[np.argmax(risk_curve >= 0.99)] if np.max(risk_curve) >= 0.99 else T

        # Plots side-by-side
        st.subheader("Simulation Results")
        st.write(f"**Time to Completion:** Classical: {classical_completion_time:.1f} days, PDE: {pde_completion_time:.1f} days")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2D Completion Plot")
            fig, ax = plt.subplots(figsize=(6,4), facecolor='#fff')
            ax.plot(simulation_time, risk_curve, color="#d32f2f", lw=2, label="Diffusion Risk")
            ax.plot(simulation_time, classical_risk, color="#1976d2", lw=2, label="Classical Risk")
            ax.set_xlabel("Time (days)", fontsize=12, fontfamily='Arial', color='#1a2a44')
            ax.set_ylabel("Average Completion (0–1)", fontsize=12, fontfamily='Arial', color='#1a2a44')
            ax.set_title("Completion: Diffusion vs Classical (Risk)", fontsize=14, fontfamily='Arial', pad=10, color='#1a2a44')
            ax.legend(frameon=True, facecolor='#fff', edgecolor='#f28c38', fontsize=10, loc='upper left')
            ax.grid(True, linestyle="--", alpha=0.7, color='#2a4066')
            ax.set_facecolor('#fff')
            ax.tick_params(colors='#1a2a44')
            st.pyplot(fig, use_container_width=True)

        with col2:
            st.subheader("3D Completion Plot")
            fig_3d = plt.figure(figsize=(6,4), facecolor='#fff')
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            ax_3d.plot(simulation_time, [0]*len(simulation_time), risk_curve, color="#d32f2f", lw=2, label="Diffusion Risk")
            ax_3d.plot(simulation_time, [1]*len(simulation_time), classical_risk, color="#1976d2", lw=2, label="Classical Risk")
            ax_3d.set_xlabel("Time (days)", fontsize=12, fontfamily='Arial', color='#1a2a44')
            ax_3d.set_ylabel("Model (0=PDE, 1=Classical)", fontsize=12, fontfamily='Arial', color='#1a2a44')
            ax_3d.set_zlabel("Average Completion (0–1)", fontsize=12, fontfamily='Arial', color='#1a2a44')
            ax_3d.set_title("3D Completion: Diffusion vs Classical (Risk)", fontsize=14, fontfamily='Arial', pad=10, color='#1a2a44')
            ax_3d.legend(frameon=True, facecolor='#fff', edgecolor='#f28c38', fontsize=10)
            ax_3d.set_facecolor('#fff')
            ax_3d.grid(True, color='#2a4066', alpha=0.7)
            ax_3d.tick_params(colors='#1a2a44')
            st.pyplot(fig_3d, use_container_width=True)

with tab2:
    st.subheader("Task Dependency Diagram")
    if st.session_state.simulation_data["tasks"] is None:
        st.info("Please run the simulation in the 'Editor & Results' tab to generate the dependency graph.")
    else:
        tasks = st.session_state.simulation_data["tasks"]
        adjacency = st.session_state.simulation_data["adjacency"]
        st.write("Directed graph showing task dependencies based on the adjacency matrix.")
        G = nx.DiGraph()
        for i, task in enumerate(tasks):
            G.add_node(i, label=f"{i+1}: {task}")
        for i in range(len(tasks)):
            preds = np.where(adjacency[:,i] > 0)[0]
            for p in preds:
                G.add_edge(p, i)
        fig_dep, ax_dep = plt.subplots(figsize=(6,4), facecolor='#2a4066')
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
                node_color='#f28c38', node_size=800, font_size=9, font_family='Arial',
                font_weight='bold', edge_color='#1a2a44', arrows=True, arrowsize=15)
        ax_dep.set_title("Task Dependency Graph", fontsize=14, fontfamily='Arial', pad=10, color='#f28c38')
        ax_dep.set_facecolor('#2a4066')
        st.pyplot(fig_dep, use_container_width=True)

with tab3:
    st.subheader("Classical Gantt Chart")
    if st.session_state.simulation_data["tasks"] is None:
        st.info("Please run the simulation in the 'Editor & Results' tab to generate the Gantt chart.")
    else:
        tasks = st.session_state.simulation_data["tasks"]
        adjacency = st.session_state.simulation_data["adjacency"]
        num_tasks = st.session_state.simulation_data["num_tasks"]
        start_times = st.session_state.simulation_data["start_times_risk"]
        finish_times = st.session_state.simulation_data["finish_times_risk"]
        durations_risk = st.session_state.simulation_data["durations_risk"]
        task_df = st.session_state.task_df

        colors = plt.cm.Oranges(np.linspace(0.3, 1, num_tasks))
        fig_gantt_classical, axg_classical = plt.subplots(figsize=(8,5), facecolor='#fff')
        for i in range(num_tasks):
            bar = axg_classical.barh(i, durations_risk[i], left=start_times[i], height=0.4,
                                     align="center", color=colors[i], edgecolor="#1a2a44", alpha=0.9)
            axg_classical.text(start_times[i] + durations_risk[i]/2, i,
                               f"{tasks[i]} ({durations_risk[i]:.0f}d)",
                               ha="center", va="center", fontsize=8, fontfamily='Arial', color='#1a2a44')
            mplcursors.cursor(bar, hover=True).connect("add", lambda sel: sel.annotation.set_text(
                f"Task: {tasks[sel.index]}\nDuration: {durations_risk[sel.index]:.1f} days\nRisk: {task_df['Risk (0-5)'][sel.index]:.1f}"
            ))

        for i in range(num_tasks):
            preds = np.where(adjacency[:,i] > 0)[0]
            for p in preds:
                axg_classical.annotate("", xy=(start_times[i], i), xytext=(finish_times[p], p),
                                       arrowprops=dict(arrowstyle="->", color="#1a2a44", lw=1.5))

        axg_classical.set_yticks(range(num_tasks))
        axg_classical.set_yticklabels([f"{i+1}" for i in range(num_tasks)], fontsize=10, fontfamily='Arial')
        axg_classical.invert_yaxis()
        axg_classical.set_xlabel("Time (days)", fontsize=12, fontfamily='Arial')
        axg_classical.set_ylabel("Task ID", fontsize=12, fontfamily='Arial')
        axg_classical.set_title("Gantt Chart (Classical)", fontsize=14, fontfamily='Arial', pad=10)
        axg_classical.grid(True, axis="x", linestyle="--", alpha=0.7, color='#2a4066')
        axg_classical.set_facecolor('#fff')
        st.pyplot(fig_gantt_classical, use_container_width=True)

with tab4:
    st.subheader("PDE Gantt Chart")
    if st.session_state.simulation_data["tasks"] is None:
        st.info("Please run the simulation in the 'Editor & Results' tab to generate the Gantt chart.")
    else:
        tasks = st.session_state.simulation_data["tasks"]
        adjacency = st.session_state.simulation_data["adjacency"]
        num_tasks = st.session_state.simulation_data["num_tasks"]
        start_times = st.session_state.simulation_data["start_times_risk"]
        finish_times = st.session_state.simulation_data["finish_times_risk"]
        durations_risk = st.session_state.simulation_data["durations_risk"]
        task_df = st.session_state.task_df

        colors = plt.cm.Oranges(np.linspace(0.3, 1, num_tasks))
        fig_gantt_pde, axg_pde = plt.subplots(figsize=(8,5), facecolor='#fff')
        for i in range(num_tasks):
            bar = axg_pde.barh(i, durations_risk[i], left=start_times[i], height=0.4,
                               align="center", color=colors[i], edgecolor="#1a2a44", alpha=0.9)
            axg_pde.text(start_times[i] + durations_risk[i]/2, i,
                         f"{tasks[i]} ({durations_risk[i]:.0f}d)",
                         ha="center", va="center", fontsize=8, fontfamily='Arial', color='#1a2a44')
            mplcursors.cursor(bar, hover=True).connect("add", lambda sel: sel.annotation.set_text(
                f"Task: {tasks[sel.index]}\nDuration: {durations_risk[sel.index]:.1f} days\nRisk: {task_df['Risk (0-5)'][sel.index]:.1f}"
            ))

        for i in range(num_tasks):
            preds = np.where(adjacency[:,i] > 0)[0]
            for p in preds:
                axg_pde.annotate("", xy=(start_times[i], i), xytext=(finish_times[p], p),
                                 arrowprops=dict(arrowstyle="->", color="#1a2a44", lw=1.5))

        axg_pde.set_yticks(range(num_tasks))
        axg_pde.set_yticklabels([f"{i+1}" for i in range(num_tasks)], fontsize=10, fontfamily='Arial')
        axg_pde.invert_yaxis()
        axg_pde.set_xlabel("Time (days)", fontsize=12, fontfamily='Arial')
        axg_pde.set_ylabel("Task ID", fontsize=12, fontfamily='Arial')
        axg_pde.set_title("Gantt Chart (PDE)", fontsize=14, fontfamily='Arial', pad=10)
        axg_pde.grid(True, axis="x", linestyle="--", alpha=0.7, color='#2a4066')
        axg_pde.set_facecolor('#fff')
        st.pyplot(fig_gantt_pde, use_container_width=True)

with tab5:
    st.subheader("Eigenvalue Analysis")
    if st.session_state.simulation_data["tasks"] is None:
        st.info("Please run the simulation in the 'Editor & Results' tab to generate eigenvalue data.")
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
        st.write("The eigenvalues represent the spectral properties of the task dependency structure.")
        df_eigen = pd.DataFrame({"Eigenvalue": np.real(eigenvalues), "Imaginary": np.imag(eigenvalues)})
        st.table(df_eigen)
        st.write(f"### Second Eigenvalue (Fiedler Value) for Centrality: {second_eigenvalue:.4f}")
        st.write("Note: The second smallest eigenvalue of the Laplacian indicates graph connectivity/centrality. A larger value suggests better connectivity.")

        fig_bar, ax_bar = plt.subplots(figsize=(6,4), facecolor='#fff')
        ax_bar.bar(range(len(np.real(eigenvalues))), np.real(eigenvalues), color='#f28c38', edgecolor='#1a2a44')
        ax_bar.set_xlabel("Task Index", fontsize=12, fontfamily='Arial', color='#1a2a44')
        ax_bar.set_ylabel("Real Eigenvalue", fontsize=12, fontfamily='Arial', color='#1a2a44')
        ax_bar.set_title("Real Part of Eigenvalues", fontsize=14, fontfamily='Arial', pad=10, color='#1a2a44')
        ax_bar.grid(True, linestyle="--", alpha=0.7, color='#2a4066')
        ax_bar.set_facecolor('#fff')
        ax_bar.tick_params(colors='#1a2a44')
        st.pyplot(fig_bar, use_container_width=True)

        st.write("### Connectivity Matrix (Adjacency)")
        st.write("Visual representation of task dependencies.")
        df_connectivity = pd.DataFrame(adjacency, index=[f"Task {i+1}" for i in range(num_tasks)], columns=[f"Task {i+1}" for i in range(num_tasks)])
        fig_heat, ax_heat = plt.subplots(figsize=(6,4), facecolor='#fff')
        sns.heatmap(df_connectivity, annot=True, cmap="Oranges", cbar_kws={'label': 'Dependency Strength'}, ax=ax_heat)
        ax_heat.set_xlabel("Task ID", fontsize=12, fontfamily='Arial', color='#1a2a44')
        ax_heat.set_ylabel("Task ID", fontsize=12, fontfamily='Arial', color='#1a2a44')
        ax_heat.set_title("Adjacency Matrix Heatmap", fontsize=14, fontfamily='Arial', pad=10, color='#1a2a44')
        st.pyplot(fig_heat, use_container_width=True)