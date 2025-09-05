import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
from io import BytesIO
import networkx as nx
import seaborn as sns
import time
try:
    import mppx
except ImportError:
    mppx = None

# Custom CSS for Halloween Theme and Aesthetics
st.markdown("""
    <style>
    .main { background-color: #1a1a1a; padding: 20px; border-radius: 10px; }
    .stButton>button {
        background-color: #e65100; color: #fff; border-radius: 8px; padding: 8px 16px;
        transition: background-color 0.3s; border: 2px solid #ffca28;
    }
    .stButton>button:hover { background-color: #ff9800; }
    .stTabs [data-baseweb="tab"] { background-color: #333; color: #fff; font-size: 16px; font-weight: bold; border: 1px solid #ffca28; }
    .stTabs [data-baseweb="tab"]:hover { background-color: #444; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: #e65100; color: #fff; }
    .stSidebar { background-color: #212121; border-right: 1px solid #ffca28; color: #fff; }
    h1 { color: #ffca28; font-family: 'Roboto', sans-serif; text-shadow: 2px 2px #333; }
    h2, h3 { color: #ffca28; font-family: 'Roboto', sans-serif; text-shadow: 1px 1px #333; }
    .stAlert { border-radius: 8px; background-color: #424242; color: #fff; }
    .left-column { background-color: #ffffff; padding: 10px; border-radius: 8px; color: #ffffff; }
    .left-column * { color: #212121 !important; }
    .right-column { background-color: #1a1a1a; padding: 10px; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("🎃 Ani's Pulse: Project Schedule Simulation")

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

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Editor & Results", "🔗 Task Dependencies", "🔮 Eigenvalue Analysis", "🌊 Ripple Visualization"])

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
            st.success("MPP file imported successfully. Edit tasks below if needed.")
            st.rerun()

    if "task_df" not in st.session_state:
        st.session_state.task_df = pd.DataFrame({
            "Select": [False] * len(default_tasks),
            "ID": list(range(1, len(default_tasks)+1)),
            "Task": default_tasks,
            "Duration (days)": default_durations,
            "Dependencies (IDs)": default_deps,
            "Risk (0-5)": default_risks
        })

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
        }
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
            st.rerun()
    with col_delete:
        if st.button("🗑️ Delete Selected Rows"):
            selected_indices = st.session_state.task_df[st.session_state.task_df["Select"] == True].index
            if not selected_indices.empty:
                st.session_state.task_df = st.session_state.task_df.drop(selected_indices).reset_index(drop=True)
                st.session_state.task_df["ID"] = pd.Series(range(1, len(st.session_state.task_df) + 1))
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
                fig, ax = plt.subplots(figsize=(6,4), facecolor='#ffffff')
                ax.plot(time, risk_curve, color="#d32f2f", lw=3, label="Diffusion Risk")
                ax.plot(time, classical_risk, color="#1976d2", lw=3, label="Classical Risk")
                ax.set_xlabel("Time (days)", fontsize=12, fontfamily='Roboto', color='#212121')
                ax.set_ylabel("Average Completion (0–1)", fontsize=12, fontfamily='Roboto', color='#212121')
                ax.set_title("Completion: Diffusion vs Classical (Risk)", fontsize=14, fontfamily='Roboto', pad=10, color='#212121')
                ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#90caf9', fontsize=10, loc='upper left')
                ax.grid(True, linestyle="--", alpha=0.7, color='#90caf9')
                ax.set_facecolor('#ffffff')
                ax.tick_params(colors='#212121')
                st.pyplot(fig, use_container_width=True)

            with col2.container():
                st.markdown('<div class="right-column">', unsafe_allow_html=True)
                fig_gantt, axg = plt.subplots(figsize=(6,4), facecolor='#ffffff')
                start_times = start_times_risk
                finish_times = finish_times_risk
                durations_gantt = durations_risk
                colors = plt.cm.Reds(task_df["Risk (0-5)"].to_numpy() / 5.0)
                for i in range(num_tasks):
                    bar = axg.barh(i, durations_gantt[i], left=start_times[i], height=0.4,
                                   align="center", color=colors[i], edgecolor="black", alpha=0.9)
                    axg.text(start_times[i] + durations_gantt[i]/2, i,
                             f"{tasks[i]} ({durations_gantt[i]:.0f}d)",
                             ha="center", va="center", fontsize=8, fontfamily='Roboto', color='white')
                    mplcursors.cursor(bar).connect("add", lambda sel: sel.annotation.set_text(
                        f"Task: {tasks[sel.index]}\nDuration: {durations_gantt[sel.index]:.1f} days\nRisk: {task_df['Risk (0-5)'][sel.index]:.1f}"
                    ))
                for i in range(num_tasks):
                    preds = np.where(adjacency[:,i] > 0)[0]
                    for p in preds:
                        axg.annotate("", xy=(start_times[i], i), xytext=(finish_times[p], p),
                                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
                axg.set_yticks(range(num_tasks))
                axg.set_yticklabels([f"{i+1}" for i in range(num_tasks)], fontsize=10, fontfamily='Roboto')
                axg.invert_yaxis()
                axg.set_xlabel("Time (days)", fontsize=12, fontfamily='Roboto')
                axg.set_ylabel("Task ID", fontsize=12, fontfamily='Roboto')
                axg.set_title("Gantt Chart (Risk)", fontsize=14, fontfamily='Roboto', pad=10)
                axg.grid(True, axis="x", linestyle="--", alpha=0.7, color='#90caf9')
                axg.set_facecolor('#ffffff')
                st.pyplot(fig_gantt, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 3D Completion Plot
            st.subheader("3D Completion Plot")
            st.write("3D view of risk-influenced schedules: PDE (red) and classical (blue).")
            fig_3d = plt.figure(figsize=(8,6), facecolor='#ffffff')
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            ax_3d.plot(time, [0]*len(time), risk_curve, color="#d32f2f", lw=3, label="Diffusion Risk")
            ax_3d.plot(time, [1]*len(time), classical_risk, color="#1976d2", lw=3, label="Classical Risk")
            ax_3d.set_xlabel("Time (days)", fontsize=12, fontfamily='Roboto', color='#212121')
            ax_3d.set_ylabel("Model (0=PDE, 1=Classical)", fontsize=12, fontfamily='Roboto', color='#212121')
            ax_3d.set_zlabel("Average Completion (0–1)", fontsize=12, fontfamily='Roboto', color='#212121')
            ax_3d.set_title("3D Completion: Diffusion vs Classical (Risk)", fontsize=14, fontfamily='Roboto', pad=10, color='#212121')
            ax_3d.legend(frameon=True, facecolor='#ffffff', edgecolor='#90caf9', fontsize=10)
            ax_3d.set_facecolor('#ffffff')
            ax_3d.grid(True, color='#90caf9', alpha=0.7)
            ax_3d.tick_params(colors='#212121')
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
        fig_dep, ax_dep = plt.subplots(figsize=(6,4), facecolor='#1a1a1a')
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
                node_color='#ffca28', node_size=800, font_size=9, font_family='Roboto',
                font_weight='bold', edge_color='#e65100', arrows=True, arrowsize=15)
        ax_dep.set_title("Task Dependency Graph", fontsize=14, fontfamily='Roboto', pad=10, color='#ffca28')
        ax_dep.set_facecolor('#333')
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
        fig_bar, ax_bar = plt.subplots(figsize=(6,4), facecolor='#ffffff')
        ax_bar.bar(range(len(np.real(eigenvalues))), np.real(eigenvalues), color='#e65100', edgecolor='#ffca28')
        ax_bar.set_xlabel("Task Index", fontsize=12, fontfamily='Roboto', color='#212121')
        ax_bar.set_ylabel("Real Eigenvalue", fontsize=12, fontfamily='Roboto', color='#212121')
        ax_bar.set_title("Real Part of Eigenvalues", fontsize=14, fontfamily='Roboto', pad=10, color='#212121')
        ax_bar.grid(True, linestyle="--", alpha=0.7, color='#90caf9')
        ax_bar.set_facecolor('#ffffff')
        ax_bar.tick_params(colors='#212121')
        st.pyplot(fig_bar, use_container_width=True)

        # Styled Connectivity Matrix
        st.write("### Connectivity Matrix (Adjacency)")
        st.write("Visual representation of task dependencies.")
        df_connectivity = pd.DataFrame(adjacency, index=[f"Task {i+1}" for i in range(num_tasks)], columns=[f"Task {i+1}" for i in range(num_tasks)])
        fig_heat, ax_heat = plt.subplots(figsize=(6,4), facecolor='#ffffff')
        sns.heatmap(df_connectivity, annot=True, cmap="YlOrRd", cbar_kws={'label': 'Dependency Strength'}, ax=ax_heat)
        ax_heat.set_xlabel("Task ID", fontsize=12, fontfamily='Roboto', color='#212121')
        ax_heat.set_ylabel("Task ID", fontsize=12, fontfamily='Roboto', color='#212121')
        ax_heat.set_title("Adjacency Matrix Heatmap", fontsize=14, fontfamily='Roboto', pad=10, color='#212121')
        st.pyplot(fig_heat, use_container_width=True)

with tab4:
    # Ripple Visualization
    st.subheader("Ripple Visualization")
    if st.session_state.simulation_data["tasks"] is None or st.session_state.simulation_data["u_matrix"] is None:
        st.info("Please run the simulation in the 'Editor & Results' tab to generate ripple data.")
    else:
        tasks = st.session_state.simulation_data["tasks"]
        u_matrix = st.session_state.simulation_data["u_matrix"]
        num_tasks = st.session_state.simulation_data["num_tasks"]
        steps = u_matrix.shape[1]
        time_quarters = [int(i * steps / 4) for i in range(1, 5)]  # Quarters at 25%, 50%, 75%, 100%

        st.write("### Completion Over Time (Quarters)")
        st.write("Heatmap showing task completion (0–1) at quarter intervals of the project timeline (140 days).")
        for idx, t in enumerate(time_quarters):
            if t < steps:
                fig_ripple, ax_ripple = plt.subplots(figsize=(6,4), facecolor='#ffffff')
                sns.heatmap(u_matrix[:, t:t+1].T, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': 'Completion'},
                            xticklabels=[f"T{idx * 25 + 25}%"], yticklabels=[], ax=ax_ripple)
                ax_ripple.set_xlabel("Time Quarter", fontsize=12, fontfamily='Roboto', color='#212121')
                ax_ripple.set_ylabel("", fontsize=12, fontfamily='Roboto', color='#212121')  # Remove y-axis label
                ax_ripple.set_title(f"Completion at {idx * 25 + 25}% (Day {int(t * dt):.0f})", fontsize=14, fontfamily='Roboto', pad=10, color='#212121')
                ax_ripple.set_facecolor('#ffffff')
                st.pyplot(fig_ripple, use_container_width=True)

        # Animation Controls
        use_delay = st.checkbox("Use Delay (0.5s per frame)", value=True)
        if st.button("▶️ Play Animation"):
            fig_anim, ax_anim = plt.subplots(figsize=(6,4), facecolor='#ffffff')
            line, = ax_anim.plot([], [], 'o-', color='#d32f2f')
            ax_anim.set_xlim(0, num_tasks - 1)
            ax_anim.set_ylim(0, 1)
            ax_anim.set_xlabel("Task Index", fontsize=12, fontfamily='Roboto', color='#212121')
            ax_anim.set_ylabel("Completion (0–1)", fontsize=12, fontfamily='Roboto', color='#212121')
            ax_anim.set_title("Completion Animation", fontsize=14, fontfamily='Roboto', pad=10, color='#212121')
            ax_anim.grid(True, linestyle="--", alpha=0.7, color='#90caf9')
            ax_anim.set_facecolor('#ffffff')
            ax_anim.tick_params(colors='#212121')

            for frame in range(0, steps, max(1, int(steps / 100))):
                line.set_data(range(num_tasks), u_matrix[:, frame])
                st.pyplot(fig_anim, use_container_width=True)
                if use_delay:
                    time.sleep(0.5)  # 0.5s delay
                else:
                    time.sleep(0.01)  # Minimal delay for smooth playback
                plt.close(fig_anim)  # Close to avoid memory buildup
                st.experimental_rerun()  # Refresh to update plot
