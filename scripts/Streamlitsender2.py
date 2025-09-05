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

# ----------------------------
# PDE Simulation Function
# ----------------------------
def run_pde(adjacency, durations, diffusion, risk_levels, T=140, dt=0.01):
    num_tasks = len(durations)
    steps = int(T/dt) + 1
    u = np.zeros((num_tasks, steps))
    for t in range(steps-1):
        du = np.zeros(num_tasks)
        for i in range(num_tasks):
            preds = np.where(adjacency[:,i] > 0)[0]
            if len(preds) == 0 or all(u[p,t] >= 1.0 - 1e-6 for p in preds):
                dur = durations[i] * max(1.0, risk_levels[i])
                if dur > 0:
                    du[i] += 1.0 / dur
                for j in preds:
                    if u[j,t] >= 1.0 - 1e-6:
                        du[i] += adjacency[j,i] * (u[j,t] - u[i,t]) * diffusion
        u[:,t+1] = np.clip(u[:,t] + du * dt, 0, 1)
    return u

# ----------------------------
# Classical Completion Function
# ----------------------------
def compute_classical_completion(start_times, finish_times, durations, time):
    num_tasks = len(durations)
    u = np.zeros((num_tasks, len(time)))
    for i in range(num_tasks):
        for j, t in enumerate(time):
            if t < start_times[i]:
                u[i, j] = 0
            elif t < finish_times[i]:
                u[i, j] = (t - start_times[i]) / durations[i]
            else:
                u[i, j] = 1
    return u.mean(axis=0)

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
    except Exception as e:
        return None, f"Error importing .mpp file: {str(e)}"

# ----------------------------
# Defaults
# ----------------------------
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

T = 140
dt = 0.01
time = np.linspace(0, T, int(T/dt)+1)

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

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Editor & Results", "🔗 Task Dependencies", "🔮 Eigenvalue Analysis", "🌊 Risk Visualization"])

with tab1:
    # Sidebar
    with st.sidebar.expander("⚙️ Simulation Parameters", expanded=True):
        st.subheader("Simulation Parameters")
        st.write("**Diffusion Factor**: Fixed at 0.001 for minimal task influence, closely matching the classical schedule with slight smoothing.")
        diffusion = 0.001
        st.write("**Risk**: Values (0–5) directly multiply task durations (e.g., risk=2 means duration * 2).")

    # Project Editor
    st.subheader("Project Editor")
    st.info("Edit tasks or import an .mpp file. Use **Task IDs** in Dependencies (e.g., `1,2`). Update dependencies after adding/deleting rows.")

    # .mpp File Upload
    st.write("📥 **Import .mpp File**")
    mpp_file = st.file_uploader("Upload Microsoft Project (.mpp) file", type=["mpp"])
    if mpp_file is not None:
        df, error = import_mpp(mpp_file)
        if error:
            st.error(error)
        else:
            st.session_state.task_df = df
            save_task_list(df)  # Autosave on import
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
        on_change=save_task_list, args=(st.session_state.task_df,)  # Autosave on edit
    )

    # Persist edits
    st.session_state.task_df = task_df

    # Add/Delete Buttons
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
                "Risk (0-5)": 0.0
            }
            st.session_state.task_df = pd.concat([st.session_state.task_df, pd.DataFrame([new_row])], ignore_index=True)
            save_task_list(st.session_state.task_df)  # Autosave on add
            st.rerun()
    with col_delete:
        if st.button("🗑️ Delete Selected Rows"):
            selected_indices = st.session_state.task_df[st.session_state.task_df["Select"] == True].index
            if not selected_indices.empty:
                st.session_state.task_df = st.session_state.task_df.drop(selected_indices).reset_index(drop=True)
                st.session_state.task_df["ID"] = pd.Series(range(1, len(st.session_state.task_df) + 1))
                save_task_list(st.session_state.task_df)  # Autosave on delete
                st.rerun()

    # Recalculate Simulation
    if st.button("🔄 Recalculate Simulation", type="primary"):
        valid = True
        tasks = task_df["Task"].tolist()
        num_tasks = len(tasks)
        st.session_state.simulation_data["num_tasks"] = num_tasks
        for i, dep in enumerate(task_df["Dependencies (IDs)"]):
            if dep:
                dep_list = [d.strip() for d in dep.split(",")]
                for d in dep_list:
                    if not d.isdigit() or int(d) < 1 or int(d) > num_tasks or int(d) == i+1:
                        valid = False
                        st.error(f"Invalid dependency for task {i+1}: '{d}' is not a valid ID or references itself.")
                        break
            if task_df["Duration (days)"][i] <= 0:
                valid = False
                st.error(f"Invalid duration for task {i+1}: Duration must be greater than 0.")
        if valid:
            progress_bar = st.progress(0)
            durations = task_df["Duration (days)"].to_numpy(dtype=float)
            risks = task_df["Risk (0-5)"].to_numpy(dtype=float)
            adjacency = np.zeros((num_tasks, num_tasks))
            for i, dep in enumerate(task_df["Dependencies (IDs)"]):
                if dep:
                    dep_list = [d.strip() for d in dep.split(",")]
                    for d in dep_list:
                        try:
                            j = int(d) - 1
                            if 0 <= j < num_tasks:
                                adjacency[j, i] = 1
                        except ValueError:
                            st.error(f"Invalid dependency ID '{d}' for task {i+1}")
                            valid = False
            if not valid:
                st.stop()
            st.session_state.simulation_data["tasks"] = tasks
            st.session_state.simulation_data["adjacency"] = adjacency
            st.write("Debug: Adjacency Matrix:", adjacency)  # Debug output

            # Diffusion-based Simulation
            u_risk = run_pde(adjacency, durations, diffusion, risks)
            risk_curve = u_risk.mean(axis=0)
            st.session_state.simulation_data["u_matrix"] = u_risk
            progress_bar.progress(50)

            # Classical Schedule
            start_times_risk = np.zeros(num_tasks)
            finish_times_risk = np.zeros(num_tasks)
            durations_risk = durations * np.maximum(1, risks)
            for i in range(num_tasks):
                preds = np.where(adjacency[:, i] > 0)[0]
                if len(preds) > 0:
                    start_times_risk[i] = max(finish_times_risk[p] for p in preds)
                finish_times_risk[i] = start_times_risk[i] + durations_risk[i]
            classical_risk = compute_classical_completion(start_times_risk, finish_times_risk, durations_risk, time)
            progress_bar.progress(100)

            st.subheader("Simulation Results")
            st.write("The plot shows risk-influenced schedules: diffusion-based (PDE, red, slightly smoother) and classical (blue, linear). Risk multiplies task durations.")

            col1, col2 = st.columns([1, 1])
            with col1.container():
                fig, ax = plt.subplots(figsize=(6,4), facecolor='#fff')
                ax.plot(time, risk_curve, color="#d32f2f", lw=2, label="Diffusion Risk")
                ax.plot(time, classical_risk, color="#1976d2", lw=2, label="Classical Risk")
                ax.set_xlabel("Time (days)", fontsize=12, fontfamily='Arial', color='#1a2a44')
                ax.set_ylabel("Average Completion (0–1)", fontsize=12, fontfamily='Arial', color='#1a2a44')
                ax.set_title("Completion: Diffusion vs Classical (Risk)", fontsize=14, fontfamily='Arial', pad=10, color='#1a2a44')
                ax.legend(frameon=True, facecolor='#fff', edgecolor='#f28c38', fontsize=10, loc='upper left')
                ax.grid(True, linestyle="--", alpha=0.7, color='#2a4066')
                ax.set_facecolor('#fff')
                ax.tick_params(colors='#1a2a44')
                st.pyplot(fig, use_container_width=True)

            with col2.container():
                st.markdown('<div class="right-column">', unsafe_allow_html=True)
                fig_gantt, axg = plt.subplots(figsize=(6,4), facecolor='#fff')
                start_times = start_times_risk
                finish_times = finish_times_risk
                durations_gantt = durations_risk
                colors = plt.cm.Oranges(task_df["Risk (0-5)"].to_numpy() / 5.0)
                for i in range(num_tasks):
                    bar = axg.barh(i, durations_gantt[i], left=start_times[i], height=0.4,
                                   align="center", color=colors[i], edgecolor="#1a2a44", alpha=0.9)
                    axg.text(start_times[i] + durations_gantt[i]/2, i,
                             f"{tasks[i]} ({durations_gantt[i]:.0f}d)",
                             ha="center", va="center", fontsize=8, fontfamily='Arial', color='#1a2a44')
                    mplcursors.cursor(bar, hover=True).connect("add", lambda sel: sel.annotation.set_text(
                        f"Task: {tasks[sel.index]}\nDuration: {durations_gantt[sel.index]:.1f} days\nRisk: {task_df['Risk (0-5)'][sel.index]:.1f}"
                    ).set_alpha(0.0).set_alpha(1.0, duration=0.5))  # Pulsing tooltip
                for i in range(num_tasks):
                    preds = np.where(adjacency[:,i] > 0)[0]
                    for p in preds:
                        axg.annotate("", xy=(start_times[i], i), xytext=(finish_times[p], p),
                                     arrowprops=dict(arrowstyle="->", color="#1a2a44", lw=1.5))
                axg.set_yticks(range(num_tasks))
                axg.set_yticklabels([f"{i+1}" for i in range(num_tasks)], fontsize=10, fontfamily='Arial')
                axg.invert_yaxis()
                axg.set_xlabel("Time (days)", fontsize=12, fontfamily='Arial')
                axg.set_ylabel("Task ID", fontsize=12, fontfamily='Arial')
                axg.set_title("Gantt Chart (Risk)", fontsize=14, fontfamily='Arial', pad=10)
                axg.grid(True, axis="x", linestyle="--", alpha=0.7, color='#2a4066')
                axg.set_facecolor('#fff')
                st.pyplot(fig_gantt, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 3D Completion Plot
            st.subheader("3D Completion Plot")
            st.write("3D view of risk-influenced schedules: PDE (red) and classical (blue).")
            fig_3d = plt.figure(figsize=(8,6), facecolor='#fff')
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            ax_3d.plot(time, [0]*len(time), risk_curve, color="#d32f2f", lw=2, label="Diffusion Risk")
            ax_3d.plot(time, [1]*len(time), classical_risk, color="#1976d2", lw=2, label="Classical Risk")
            ax_3d.set_xlabel("Time (days)", fontsize=12, fontfamily='Arial', color='#1a2a44')
            ax_3d.set_ylabel("Model (0=PDE, 1=Classical)", fontsize=12, fontfamily='Arial', color='#1a2a44')
            ax_3d.set_zlabel("Average Completion (0–1)", fontsize=12, fontfamily='Arial', color='#1a2a44')
            ax_3d.set_title("3D Completion: Diffusion vs Classical (Risk)", fontsize=14, fontfamily='Arial', pad=10, color='#1a2a44')
            ax_3d.legend(frameon=True, facecolor='#fff', edgecolor='#f28c38', fontsize=10)
            ax_3d.set_facecolor('#fff')
            ax_3d.grid(True, color='#2a4066', alpha=0.7)
            ax_3d.tick_params(colors='#1a2a44')
            st.pyplot(fig_3d, use_container_width=True)

            # Export Options
            col_export1, col_export2 = st.columns(2)
            with col_export1:
                csv = st.session_state.task_df.to_csv(index=False)
                st.download_button("📄 Download Task Data (CSV)", csv, "project_tasks.csv", "text/csv")
            with col_export2:
                buf = BytesIO()
                fig_gantt.savefig(buf, format="png", bbox_inches="tight")
                st.download_button("🖼️ Download Gantt Chart (PNG)", buf.getvalue(), "gantt_chart.png", "image/png")

with tab2:
    # Task Dependency Diagram
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
    # Eigenvalue Analysis
    st.subheader("Eigenvalue Analysis")
    if st.session_state.simulation_data["tasks"] is None:
        st.info("Please run the simulation in the 'Editor & Results' tab to generate eigenvalue data.")
    else:
        adjacency = st.session_state.simulation_data["adjacency"]
        num_tasks = st.session_state.simulation_data["num_tasks"]
        eigenvalues, _ = np.linalg.eig(adjacency.astype(float))
        # Compute Laplacian for second eigenvalue (centrality)
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

        # Bar Chart of Real Eigenvalues
        fig_bar, ax_bar = plt.subplots(figsize=(6,4), facecolor='#fff')
        ax_bar.bar(range(len(np.real(eigenvalues))), np.real(eigenvalues), color='#f28c38', edgecolor='#1a2a44')
        ax_bar.set_xlabel("Task Index", fontsize=12, fontfamily='Arial', color='#1a2a44')
        ax_bar.set_ylabel("Real Eigenvalue", fontsize=12, fontfamily='Arial', color='#1a2a44')
        ax_bar.set_title("Real Part of Eigenvalues", fontsize=14, fontfamily='Arial', pad=10, color='#1a2a44')
        ax_bar.grid(True, linestyle="--", alpha=0.7, color='#2a4066')
        ax_bar.set_facecolor('#fff')
        ax_bar.tick_params(colors='#1a2a44')
        st.pyplot(fig_bar, use_container_width=True)

        # Styled Connectivity Matrix
        st.write("### Connectivity Matrix (Adjacency)")
        st.write("Visual representation of task dependencies.")
        df_connectivity = pd.DataFrame(adjacency, index=[f"Task {i+1}" for i in range(num_tasks)], columns=[f"Task {i+1}" for i in range(num_tasks)])
        fig_heat, ax_heat = plt.subplots(figsize=(6,4), facecolor='#fff')
        sns.heatmap(df_connectivity, annot=True, cmap="Oranges", cbar_kws={'label': 'Dependency Strength'}, ax=ax_heat)
        ax_heat.set_xlabel("Task ID", fontsize=12, fontfamily='Arial', color='#1a2a44')
        ax_heat.set_ylabel("Task ID", fontsize=12, fontfamily='Arial', color='#1a2a44')
        ax_heat.set_title("Adjacency Matrix Heatmap", fontsize=14, fontfamily='Arial', pad=10, color='#1a2a44')
        st.pyplot(fig_heat, use_container_width=True)

with tab4:
    # Risk Visualization
    st.subheader("Risk Visualization")
    if st.session_state.simulation_data["tasks"] is None or st.session_state.simulation_data["u_matrix"] is None:
        st.info("Please run the simulation in the 'Editor & Results' tab to generate risk data.")
    else:
        tasks = st.session_state.simulation_data["tasks"]
        u_matrix = st.session_state.simulation_data["u_matrix"]
        num_tasks = st.session_state.simulation_data["num_tasks"]
        steps = u_matrix.shape[1]
        task_df = st.session_state.task_df

        st.write("### Risk Impact by Task (Bar Chart)")
        st.write("Bar heights reflect risk levels (0–5) and their impact on task durations, styled like MS Project.")
        fig_risk, ax_risk = plt.subplots(figsize=(6,4), facecolor='#fff')
        risks = task_df["Risk (0-5)"].to_numpy()
        durations = task_df["Duration (days)"].to_numpy()
        risk_impact = durations * np.maximum(1, risks) - durations  # Additional days due to risk
        bars = ax_risk.bar(range(num_tasks), risk_impact, color='#f28c38', edgecolor='#1a2a44', alpha=0.9)
        ax_risk.set_xticks(range(num_tasks))
        ax_risk.set_xticklabels([f"Task {i+1}" for i in range(num_tasks)], rotation=45, fontsize=10, fontfamily='Arial')
        ax_risk.set_xlabel("Task ID", fontsize=12, fontfamily='Arial', color='#1a2a44')
        ax_risk.set_ylabel("Risk Impact (Additional Days)", fontsize=12, fontfamily='Arial', color='#1a2a44')
        ax_risk.set_title("Risk Impact on Task Durations", fontsize=14, fontfamily='Arial', pad=10, color='#1a2a44')
        ax_risk.grid(True, axis="y", linestyle="--", alpha=0.7, color='#2a4066')
        ax_risk.set_facecolor('#fff')
        ax_risk.tick_params(colors='#1a2a44')
        for bar, risk in zip(bars, risks):
            height = bar.get_height()
            ax_risk.text(bar.get_x() + bar.get_width()/2, height,
                         f"{risk:.1f}", ha="center", va="bottom", fontsize=8, color='#1a2a44')
            mplcursors.cursor(bar, hover=True).connect("add", lambda sel: sel.annotation.set_text(
                f"Task: {tasks[sel.index]}\nDuration: {durations[sel.index]:.1f} days\nRisk: {risks[sel.index]:.1f}"
            ).set_alpha(0.0).set_alpha(1.0, duration=0.5))  # Pulsing tooltip
        st.pyplot(fig_risk, use_container_width=True)

        # Animation Controls (Slider-based)
        st.write("### Completion Ripple Animation")
        frame = st.slider("Select Frame", 0, steps-1, 0, step=max(1, int(steps / 100)))
        fig_anim, ax_anim = plt.subplots(figsize=(6,4), facecolor='#fff')
        line, = ax_anim.plot([], [], 'o-', color='#f28c38', lw=2, marker='o', markersize=10, markeredgecolor='#1a2a44', markeredgewidth=2)
        ripple_marker, = ax_anim.plot([], [], 'o', color='#d8701f', markersize=15, alpha=0.7)
        ax_anim.set_xlim(0, num_tasks - 1)
        ax_anim.set_ylim(0, 1)
        ax_anim.set_xlabel("Task Index", fontsize=12, fontfamily='Arial', color='#1a2a44')
        ax_anim.set_ylabel("Completion (0–1)", fontsize=12, fontfamily='Arial', color='#1a2a44')
        ax_anim.set_title("Completion Ripple Animation", fontsize=14, fontfamily='Arial', pad=10, color='#f28c38')
        ax_anim.grid(True, linestyle="--", alpha=0.7, color='#2a4066')
        ax_anim.set_facecolor('#fff')
        ax_anim.tick_params(colors='#1a2a44')

        line.set_data(range(num_tasks), u_matrix[:, frame])
        max_completion_idx = np.argmax(u_matrix[:, frame])
        ripple_marker.set_xdata([max_completion_idx])
        ripple_marker.set_ydata([u_matrix[max_completion_idx, frame]])
        st.pyplot(fig_anim, use_container_width=True)
