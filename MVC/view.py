import streamlit as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
import networkx as nx
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.patches as patches
from sde_solver import SDEParameters


# -------------------------------
# Sidebar
# -------------------------------
def render_sidebar(model):
    with st.sidebar.expander("⚙️ Simulation Parameters", expanded=True):
        st.subheader("Simulation Parameters")
        diffusion = st.slider("Diffusion Coefficient", 0.001, 0.05,
                              st.session_state.get("diffusion", 0.001), 0.001,
                              help="Finely tune delay propagation (0.001–0.05).")
        reaction_multiplier = st.slider("Reaction Multiplier", 1.0, 5.0,
                                        st.session_state.get("reaction_multiplier", 2.0), 0.1,
                                        help="Scales the base reaction rate (1.0–5.0).")
        max_delay = st.slider("Max Delay", 0.01, 0.1,
                              st.session_state.get("max_delay", 0.05), 0.01,
                              help="Max delay from predecessors (0.01–0.1).")
        col_diff1, col_diff2, col_diff3 = st.columns(3)
        with col_diff1:
            st.button("Low (0.01)", key="diff_low")
        with col_diff2:
            st.button("Medium (0.025)", key="diff_medium")
        with col_diff3:
            st.button("High (0.05)", key="diff_high")
        st.write("**Diffusion Factor**: Controls delay propagation from risk points.")
        st.write("**Reaction Multiplier**: Scales the base progress rate.")
        st.write("**Max Delay**: Caps delay from predecessors.")
        st.write("**Risk**: Values (0–5) multiply task durations.")
        return diffusion, reaction_multiplier, max_delay


# -------------------------------
# Editor Tab
# -------------------------------
def render_editor_tab(model):
    st.subheader("Project Editor")
    st.info("Edit tasks or import an .mpp file. Use **Task IDs** in Dependencies (e.g., `1,2`).")
    mpp_file = st.file_uploader("Upload Microsoft Project (.mpp) file", type=["mpp"])
    task_df = st.data_editor(
        model.task_df,
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
    col_add, col_delete = st.columns(2)
    with col_add:
        st.button("➕ Add New Row", key="add_row")
    with col_delete:
        st.button("🗑️ Delete Selected Rows", key="delete_rows")
    st.button("🔄 Recalculate Simulation", type="primary", key="recalculate")
    return task_df, mpp_file



# -------------------------------
# Basic Schedule Tab (NEW)
# -------------------------------


# -------------------------------
# Basic Schedule Tab (ADDITION ONLY)
# -------------------------------
def render_basic_schedule_tab(model):
    """
    Naive baseline schedule - just dependencies and durations.
    No risk, no PDE, no fancy calculations.
    This is our sanity check baseline.
    """
    st.subheader("📅 Basic Schedule (Naive Baseline)")
    st.info(
        "Simple sequential scheduling based only on task durations and dependencies. This is our baseline sanity check.")

    # Check if we have tasks
    if len(model.task_df) == 0:
        st.warning("No tasks defined. Please add tasks in the Editor tab.")
        return

    # Calculate basic schedule
    basic_schedule = calculate_basic_schedule(model.task_df)

    if basic_schedule is None:
        st.error("Cannot calculate schedule. Check for dependency errors.")
        return

    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tasks", len(model.task_df))
    with col2:
        st.metric("Project Duration", f"{basic_schedule['project_duration']:.1f} days")
    with col3:
        st.metric("Critical Path Length", len(basic_schedule['critical_path']))
    with col4:
        st.metric("Total Work", f"{basic_schedule['total_work']:.1f} days")

    # Show schedule table
    st.subheader("📋 Task Schedule")
    schedule_df = create_schedule_dataframe(model.task_df, basic_schedule)

    # Make the table interactive and sortable
    st.dataframe(
        schedule_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Task ID": st.column_config.NumberColumn("ID", width="small"),
            "Task Name": st.column_config.TextColumn("Task", width="large"),
            "Duration": st.column_config.NumberColumn("Duration (days)", format="%.1f"),
            "Start Day": st.column_config.NumberColumn("Start", format="%.1f"),
            "Finish Day": st.column_config.NumberColumn("Finish", format="%.1f"),
            "Dependencies": st.column_config.TextColumn("Deps", width="small"),
            "Critical": st.column_config.CheckboxColumn("Critical Path"),
            "Early Start": st.column_config.NumberColumn("Early Start", format="%.1f"),
            "Late Start": st.column_config.NumberColumn("Late Start", format="%.1f"),
            "Float": st.column_config.NumberColumn("Float (days)", format="%.1f"),
        }
    )

    # Basic Gantt Chart
    st.subheader("📊 Basic Gantt Chart")
    render_basic_gantt_chart(basic_schedule, model.task_df)

    # Critical Path Analysis
    st.subheader("🔍 Critical Path Analysis")
    render_critical_path_info(basic_schedule)


def calculate_basic_schedule(task_df):
    """Calculate the most basic schedule possible"""
    try:
        num_tasks = len(task_df)
        early_start = np.zeros(num_tasks)
        early_finish = np.zeros(num_tasks)
        late_start = np.zeros(num_tasks)
        late_finish = np.zeros(num_tasks)

        dependency_matrix = build_dependency_matrix(task_df)
        task_order = topological_sort(task_df, dependency_matrix)
        if task_order is None:
            return None

        for task_idx in task_order:
            predecessors = np.where(dependency_matrix[:, task_idx] == 1)[0]
            if len(predecessors) > 0:
                early_start[task_idx] = np.max(early_finish[predecessors])
            else:
                early_start[task_idx] = 0
            duration = task_df.iloc[task_idx]["Duration (days)"]
            early_finish[task_idx] = early_start[task_idx] + duration

        project_duration = np.max(early_finish)

        for i in range(num_tasks):
            successors = np.where(dependency_matrix[i, :] == 1)[0]
            if len(successors) == 0:
                late_finish[i] = project_duration
            else:
                late_finish[i] = project_duration

        for task_idx in reversed(task_order):
            successors = np.where(dependency_matrix[task_idx, :] == 1)[0]
            if len(successors) > 0:
                late_finish[task_idx] = np.min(late_start[successors])
            duration = task_df.iloc[task_idx]["Duration (days)"]
            late_start[task_idx] = late_finish[task_idx] - duration

        total_float = late_start - early_start
        critical_tasks = np.where(np.abs(total_float) < 0.001)[0]
        critical_path = find_critical_path(critical_tasks, dependency_matrix, task_order)

        return {
            'early_start': early_start,
            'early_finish': early_finish,
            'late_start': late_start,
            'late_finish': late_finish,
            'total_float': total_float,
            'critical_path': critical_path,
            'project_duration': project_duration,
            'total_work': np.sum(task_df["Duration (days)"]),
            'task_order': task_order
        }
    except Exception as e:
        st.error(f"Error calculating basic schedule: {str(e)}")
        return None


def build_dependency_matrix(task_df):
    """Build adjacency matrix from dependencies"""
    num_tasks = len(task_df)
    matrix = np.zeros((num_tasks, num_tasks))
    for i, row in task_df.iterrows():
        deps = str(row['Dependencies (IDs)']).strip()
        if deps and deps != "":
            dep_list = [d.strip() for d in deps.split(",")]
            for dep in dep_list:
                if dep.isdigit():
                    dep_id = int(dep)
                    if 1 <= dep_id <= num_tasks:
                        matrix[dep_id - 1, i] = 1
    return matrix


def topological_sort(task_df, dependency_matrix):
    """Topological sort using Kahn's algorithm"""
    num_tasks = len(task_df)
    in_degree = np.sum(dependency_matrix, axis=0)
    queue = [i for i in range(num_tasks) if in_degree[i] == 0]
    result = []

    while queue:
        current = queue.pop(0)
        result.append(current)
        successors = np.where(dependency_matrix[current, :] == 1)[0]
        for successor in successors:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    if len(result) != num_tasks:
        return None
    return result


def find_critical_path(critical_tasks, dependency_matrix, task_order):
    """Find the actual critical path sequence"""
    if len(critical_tasks) == 0:
        return []

    critical_set = set(critical_tasks)
    path = []
    start_candidates = []
    for task in critical_tasks:
        predecessors = np.where(dependency_matrix[:, task] == 1)[0]
        critical_predecessors = [p for p in predecessors if p in critical_set]
        if len(critical_predecessors) == 0:
            start_candidates.append(task)

    if start_candidates:
        current = start_candidates[0]
        path.append(current)
        while True:
            successors = np.where(dependency_matrix[current, :] == 1)[0]
            critical_successors = [s for s in successors if s in critical_set]
            if critical_successors:
                current = critical_successors[0]
                path.append(current)
            else:
                break
    return path


def create_schedule_dataframe(task_df, basic_schedule):
    """Create a comprehensive schedule dataframe"""
    schedule_data = []
    for i, row in task_df.iterrows():
        schedule_data.append({
            "Task ID": row["ID"],
            "Task Name": row["Task"],
            "Duration": row["Duration (days)"],
            "Start Day": basic_schedule['early_start'][i],
            "Finish Day": basic_schedule['early_finish'][i],
            "Dependencies": row["Dependencies (IDs)"],
            "Critical": i in basic_schedule['critical_path'],
            "Early Start": basic_schedule['early_start'][i],
            "Late Start": basic_schedule['late_start'][i],
            "Float": basic_schedule['total_float'][i],
        })
    return pd.DataFrame(schedule_data)


def render_basic_gantt_chart(basic_schedule, task_df):
    """Render a simple Gantt chart for the basic schedule"""
    num_tasks = len(task_df)
    fig, ax = plt.subplots(figsize=(12, max(6, num_tasks * 0.4)))

    normal_color = '#4CAF50'  # Green
    critical_color = '#F44336'  # Red

    for i in range(num_tasks):
        start = basic_schedule['early_start'][i]
        duration = task_df.iloc[i]["Duration (days)"]
        is_critical = i in basic_schedule['critical_path']
        color = critical_color if is_critical else normal_color

        rect = patches.Rectangle(
            (start, i - 0.3), duration, 0.6,
            linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
        )
        ax.add_patch(rect)

        task_name = task_df.iloc[i]["Task"]
        ax.text(start + duration / 2, i, f"{task_name}\n({duration:.1f}d)",
                ha='center', va='center', fontsize=8, fontweight='bold')

        if not is_critical and basic_schedule['total_float'][i] > 0:
            float_rect = patches.Rectangle(
                (start + duration, i - 0.15), basic_schedule['total_float'][i], 0.3,
                linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.5
            )
            ax.add_patch(float_rect)

    ax.set_xlim(0, basic_schedule['project_duration'] * 1.1)
    ax.set_ylim(-0.5, num_tasks - 0.5)
    ax.set_xlabel('Time (Days)')
    ax.set_ylabel('Tasks')
    ax.set_title('Basic Schedule Gantt Chart\n(Green=Normal, Red=Critical Path)')
    ax.set_yticks(range(num_tasks))
    ax.set_yticklabels([f"Task {i + 1}" for i in range(num_tasks)])
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

    normal_patch = patches.Patch(color=normal_color, alpha=0.7, label='Normal Tasks')
    critical_patch = patches.Patch(color=critical_color, alpha=0.7, label='Critical Path')
    float_patch = patches.Patch(color='lightgray', alpha=0.5, label='Float/Slack')
    ax.legend(handles=[normal_patch, critical_patch, float_patch], loc='upper right')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)


def render_critical_path_info(basic_schedule):
    """Display critical path information"""
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Critical Path Tasks:**")
        if basic_schedule['critical_path']:
            critical_info = []
            for task_idx in basic_schedule['critical_path']:
                task_id = task_idx + 1
                critical_info.append(f"Task {task_id}")
            st.write(" → ".join(critical_info))
        else:
            st.write("No critical path identified")

    with col2:
        st.write("**Key Metrics:**")
        st.write(f"• Critical Path Length: {len(basic_schedule['critical_path'])} tasks")
        st.write(f"• Project Duration: {basic_schedule['project_duration']:.1f} days")
        total_float = np.sum(basic_schedule['total_float'])
        st.write(f"• Total Float in Project: {total_float:.1f} days")

    st.write("**Task Float Analysis:**")
    float_df = pd.DataFrame({
        'Task ID': range(1, len(basic_schedule['total_float']) + 1),
        'Float (days)': basic_schedule['total_float'],
        'Status': ['Critical' if f < 0.001 else f'Float: {f:.1f}d' for f in basic_schedule['total_float']]
    })
    st.dataframe(float_df, use_container_width=True, hide_index=True)














# -------------------------------
# Simulation Results
# -------------------------------
# Replace your current render_simulation_results function in view.py with this ORIGINAL version

def render_simulation_results(model):
    if not model.simulation_data.get("tasks"):
        st.info("Please run the simulation to view results.")
        return

    simulation_time = model.simulation_data.get("simulation_time", np.array([]))
    risk_curve = model.simulation_data.get("risk_curve", np.array([]))
    classical_risk = model.simulation_data.get("classical_risk", np.array([]))

    # Safety check
    if simulation_time is None or risk_curve is None or classical_risk is None:
        st.info("Simulation data is incomplete. Cannot render results.")
        return

    # Make sure arrays match in length
    min_len = min(len(simulation_time), len(risk_curve), len(classical_risk))
    simulation_time = simulation_time[:min_len]
    risk_curve = risk_curve[:min_len]
    classical_risk = classical_risk[:min_len]

    st.subheader("Simulation Results")
    col1, col2 = st.columns(2)

    # --- 2D Plot ---
    with col1:
        st.subheader("2D Completion Plot")
        # OPTION 2: Use consistent figsize that works well for both
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='#fff')
        # Blue Classical curve (SOLID)
        ax.plot(simulation_time, classical_risk, color="#1976d2", lw=2, label="Classical Risk")
        # Red PDE curve (DOTTED)
        ax.plot(simulation_time, risk_curve, color="#d32f2f", lw=2, linestyle="--", label="Diffusion Risk")

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Average Completion (0–1)")
        ax.set_title("Completion: Classical vs Diffusion")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig, use_container_width=True)

    # --- 3D Plot ---
    with col2:
        st.subheader("3D Completion Plot")
        # OPTION 2: Same figsize for consistency
        fig_3d = plt.figure(figsize=(8, 6), facecolor='#fff')
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # Blue Classical (SOLID)
        ax_3d.plot(simulation_time, [0] * len(simulation_time), classical_risk, color="#1976d2", lw=2,
                   label="Classical Risk")
        # Red PDE (DOTTED)
        ax_3d.plot(simulation_time, [1] * len(simulation_time), risk_curve, color="#d32f2f", lw=2, linestyle='--',
                   label="Diffusion Risk")

        ax_3d.set_xlabel("Time (days)")
        ax_3d.set_ylabel("Model (0=Classical,1=Diffusion)")
        ax_3d.set_zlabel("Average Completion (0–1)")
        ax_3d.set_title("3D Completion: Classical vs Diffusion")
        ax_3d.legend()

        # Better 3D layout
        fig_3d.tight_layout()
        st.pyplot(fig_3d, use_container_width=True)# -------------------------------
# Dependency Tab
# -------------------------------
def render_dependency_tab(model):
    if not model.simulation_data.get("tasks"):
        st.info("Please run the simulation to view the dependency graph.")
        return
    tasks = model.simulation_data["tasks"]
    adjacency = model.simulation_data["adjacency"]
    st.subheader("Task Dependency Diagram")
    st.write("Directed graph showing task dependencies.")
    G = nx.DiGraph()
    for i, task in enumerate(tasks):
        G.add_node(i, label=f"{i + 1}: {task}")
    for i in range(len(tasks)):
        preds = np.where(adjacency[:, i] > 0)[0]
        for p in preds:
            G.add_edge(p, i)
    fig_dep, ax_dep = plt.subplots(figsize=(6, 4), facecolor='#2a4066')
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
            node_color='#f28c38', node_size=800, font_size=9, font_family='Arial',
            font_weight='bold', edge_color='#1a2a44', arrows=True, arrowsize=15)
    ax_dep.set_title("Task Dependency Graph", fontsize=14, fontfamily='Arial', pad=10, color='#f28c38')
    ax_dep.set_facecolor('#2a4066')
    st.pyplot(fig_dep, use_container_width=True)


# -------------------------------
# Critical Path Helper
# -------------------------------
def compute_critical_path(adjacency, durations):
    """
    Compute critical path using adjacency matrix and task durations.
    Returns: list of task indices on critical path, earliest start times array.
    """
    num_tasks = len(durations)
    adj = np.array(adjacency)
    in_degree = adj.sum(axis=0)
    order = []
    zero_in = [i for i in range(num_tasks) if in_degree[i] == 0]

    # Topological sort
    while zero_in:
        n = zero_in.pop(0)
        order.append(n)
        for i in range(num_tasks):
            if adj[n, i]:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    zero_in.append(i)

    earliest_start = np.zeros(num_tasks)
    prev_task = [None] * num_tasks

    for i in order:
        preds = np.where(adj[:, i] > 0)[0]
        if preds.size > 0:
            est = earliest_start[preds] + durations[preds]
            earliest_start[i] = est.max()
            prev_task[i] = preds[est.argmax()]

    # Backtrack to get critical path
    end_task = (earliest_start + durations).argmax()
    critical_path = []
    t = end_task
    while t is not None:
        critical_path.insert(0, t)
        t = prev_task[t]
    return critical_path, earliest_start


# -------------------------------
# Helper functions
# -------------------------------
def update_parent_durations(task_df):
    """
    Update parent task durations as the sum of child durations if they exist.
    """
    durations = task_df['Duration'].copy()
    for idx, row in task_df.iterrows():
        if row['Level'] == 0:
            # Children are all rows immediately below with Level 1
            children = task_df[(task_df['Level'] == 1) & (task_df.index > idx)]
            if not children.empty:
                durations[idx] = children['Duration'].sum()
    return durations


# -------------------------------
# Gantt Chart with Critical Path & Subtasks
# -------------------------------
def render_classical_gantt(model):
    # SAFER DEFENSIVE VALIDATION
    print(f"🔍 Classical Gantt called with {len(model.task_df)} tasks")

    # Check if classical data exists using 'is None' instead of truthiness
    start_times = model.simulation_data.get("start_times_classical")
    finish_times = model.simulation_data.get("finish_times_classical")

    if start_times is None:
        st.info("No classical simulation data. Run simulation first.")
        print("🔍 No classical simulation data found")
        return

    if finish_times is None:
        st.error("Classical finish times missing. Run simulation first.")
        print("🔍 No classical finish times found")
        return

    print(f"🔍 Classical arrays: start_times={len(start_times)}, finish_times={len(finish_times)}")

    # Validate array sizes match task count
    if len(start_times) != len(model.task_df):
        st.error(
            f"❌ Data mismatch detected: {len(model.task_df)} tasks but {len(start_times)} classical results. Please re-run simulation.")
        print(f"🔍 MISMATCH: {len(model.task_df)} tasks vs {len(start_times)} classical results")
        return

    if len(finish_times) != len(model.task_df):
        st.error(
            f"❌ Data mismatch detected: {len(model.task_df)} tasks but {len(finish_times)} finish times. Please re-run simulation.")
        print(f"🔍 MISMATCH: {len(model.task_df)} tasks vs {len(finish_times)} finish times")
        return

    print("✅ Classical Gantt validation passed")

    # Continue with your existing code but REMOVE these duplicate lines if they exist:
    # start_times = model.simulation_data.get("start_times_classical")  # REMOVE THIS
    # finish_times = model.simulation_data.get("finish_times_classical")  # REMOVE THIS

    # Your existing code continues here...
    if not model.simulation_data.get("tasks"):
        st.info("Run the simulation to view the Gantt chart.")
        return

    task_df = model.task_df.copy()
    tasks = task_df["Task"].tolist()
    num_tasks = len(tasks)
    durations_risk = task_df["Duration (days)"].values

    classical_completion_time = np.max(finish_times)
    pde_completion_time = np.max(model.simulation_data.get("finish_times_risk", finish_times))

    st.subheader("Classical Gantt Chart")
    st.write(
        f"**Time to Completion:** Classical: {classical_completion_time:.1f} days, PDE: {pde_completion_time:.1f} days")

    # Get the stored classical schedule data
    start_times = model.simulation_data.get("start_times_classical")
    finish_times = model.simulation_data.get("finish_times_classical")

    if start_times is None or finish_times is None:
        st.error("Classical schedule data not available. Run simulation first.")
        return

    task_df = model.task_df.copy()
    tasks = task_df["Task"].tolist()
    num_tasks = len(tasks)
    durations_risk = task_df["Duration (days)"].values

    classical_completion_time = np.max(finish_times)
    pde_completion_time = np.max(model.simulation_data.get("finish_times_risk", finish_times))

    st.subheader("Classical Gantt Chart")
    st.write(
        f"**Time to Completion:** Classical: {classical_completion_time:.1f} days, PDE: {pde_completion_time:.1f} days")

    colors = plt.cm.Oranges(np.linspace(0.3, 1, num_tasks))
    fig_gantt, ax_gantt = plt.subplots(figsize=(8, 5), facecolor='#fff')

    # Plot each task using the stored start_times
    for i, row in task_df.iterrows():
        color = colors[i]
        bar = ax_gantt.barh(i, durations_risk[i], left=start_times[i], height=0.4,
                            align="center", color=color, edgecolor="#1a2a44", alpha=0.9)
        # Add task name + duration at the end of the bar
        ax_gantt.text(start_times[i] + durations_risk[i] + 0.5, i,
                      f"{row['Task']} ({durations_risk[i]:.0f}d)",
                      ha="left", va="center", fontsize=8, fontfamily='Arial')
        # Hover info
        mplcursors.cursor(bar, hover=True).connect("add", lambda sel: sel.annotation.set_text(
            f"Task: {tasks[sel.index]}\nDuration: {durations_risk[sel.index]:.1f} days\nRisk: {task_df['Risk (0-5)'][sel.index]:.1f}"
        ))

    # Draw dependency arrows using stored finish_times
    adjacency = np.zeros((num_tasks, num_tasks))
    for i, row in task_df.iterrows():
        deps = str(row['Dependencies (IDs)']).split(",") if row['Dependencies (IDs)'] else []
        deps = [int(d.strip()) - 1 for d in deps if d.strip().isdigit()]
        for d in deps:
            adjacency[d, i] = 1

    for i in range(num_tasks):
        preds = np.where(adjacency[:, i] > 0)[0]
        for p in preds:
            ax_gantt.annotate("",
                              xy=(start_times[i], i),
                              xytext=(finish_times[p], p),
                              arrowprops=dict(arrowstyle="->", color="#1a2a44", lw=1.5))

    ax_gantt.set_yticks(range(num_tasks))
    ax_gantt.set_yticklabels([f"{i + 1}" for i in range(num_tasks)], fontsize=10, fontfamily='Arial')
    ax_gantt.invert_yaxis()
    ax_gantt.set_xlabel("Time (days)", fontsize=12, fontfamily='Arial')
    ax_gantt.set_ylabel("Task ID", fontsize=12, fontfamily='Arial')
    ax_gantt.set_title("Gantt Chart (Classical)", fontsize=14, fontfamily='Arial', pad=10)
    ax_gantt.grid(True, axis="x", linestyle="--", alpha=0.7, color='#2a4066')
    ax_gantt.set_facecolor('#fff')
    st.pyplot(fig_gantt, use_container_width=True)
# -------------------------------
# Eigenvalue Tab
# -------------------------------

# Monte Carlo GANTT#



#Fixed below#

def render_monte_carlo_gantt_chart(model):
    """Render Monte Carlo Gantt chart with confidence bands - FIXED VERSION"""

    #TRoubleshooting Code for Testing the Switch Off#
    #if model.simulation_data.get("monte_carlo_results"):
    #    render_monte_carlo_debug_info(model)  # Add this line
    #    render_monte_carlo_gantt_chart(model)  # Your existing call

    if not model.simulation_data.get("monte_carlo_results"):
        st.info("🎲 Run Monte Carlo analysis to view probabilistic Gantt chart")
        return

    st.subheader("📊 Monte Carlo Gantt Chart")

    # Get Monte Carlo results
    mc_results = model.simulation_data["monte_carlo_results"]
    task_df = model.task_df
    num_tasks = len(task_df)

    # DEBUG: Print array sizes to understand the mismatch
    st.write("🔍 **Debug Info:**")
    st.write(f"Number of tasks in DataFrame: {num_tasks}")

    # Check if task_start_percentiles exists and its structure
    if "task_start_percentiles" in mc_results:
        task_start_perc = mc_results["task_start_percentiles"]
        st.write(f"task_start_percentiles keys: {list(task_start_perc.keys())}")
        for key, values in task_start_perc.items():
            if hasattr(values, '__len__'):
                st.write(f"  {key}: length = {len(values)}")
            else:
                st.write(f"  {key}: not an array")
    else:
        st.error("❌ task_start_percentiles not found in Monte Carlo results")
        return

    # Check if arrays are the right size
    confidence_levels = mc_results.get("confidence_levels", [90])
    if not confidence_levels:
        st.error("❌ No confidence levels found")
        return

    # Get percentile keys
    confidence_level = confidence_levels[0]  # Use first available
    lower_key = f"P{int((100 - confidence_level) / 2)}"
    upper_key = f"P{int(100 - (100 - confidence_level) / 2)}"

    st.write(f"Using confidence level: {confidence_level}%")
    st.write(f"Looking for keys: {lower_key}, {upper_key}")

    # Validate that we have the right keys and array sizes
    task_start_perc = mc_results.get("task_start_percentiles", {})
    task_finish_perc = mc_results.get("task_finish_percentiles", {})

    if lower_key not in task_start_perc or upper_key not in task_start_perc:
        st.error(f"❌ Required percentile keys not found. Available: {list(task_start_perc.keys())}")
        return

    # Check array sizes match
    start_lower_array = task_start_perc[lower_key]
    if len(start_lower_array) != num_tasks:
        st.error(f"❌ Array size mismatch: Expected {num_tasks} tasks, got {len(start_lower_array)} in percentiles")
        st.write("This suggests the Monte Carlo simulation stored results for a different number of tasks.")
        st.write("Try running the Monte Carlo simulation again.")
        return

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        available_levels = mc_results.get("confidence_levels", [90])
        confidence_level = st.selectbox(
            "Confidence Level for Bands",
            options=available_levels,
            index=0
        )
    with col2:
        show_criticality = st.checkbox("Show Critical Path Probability", value=True)

    # Recalculate keys based on selected confidence level
    lower_key = f"P{int((100 - confidence_level) / 2)}"
    upper_key = f"P{int(100 - (100 - confidence_level) / 2)}"

    # Validate selected keys exist
    if lower_key not in task_start_perc or upper_key not in task_start_perc:
        st.error(f"❌ Selected confidence level {confidence_level}% not available")
        return

    # Create Gantt chart
    fig, ax = plt.subplots(figsize=(12, max(6, num_tasks * 0.4)))

    # Color scheme
    colors = plt.cm.Set3(np.linspace(0, 1, num_tasks))

    # Get mean times (fallback if percentiles fail)
    mean_start_times = mc_results.get("mean_start_times")
    mean_finish_times = mc_results.get("mean_finish_times")

    if mean_start_times is None or len(mean_start_times) != num_tasks:
        st.error("❌ Mean start times not available or wrong size")
        return
    if mean_finish_times is None or len(mean_finish_times) != num_tasks:
        st.error("❌ Mean finish times not available or wrong size")
        return

    # Plot each task
    for i, row in task_df.iterrows():
        if i >= num_tasks:  # Safety check
            break

        color = colors[i]

        try:
            # Get confidence intervals - with bounds checking
            start_lower = task_start_perc[lower_key][i]
            start_upper = task_start_perc[upper_key][i]
            finish_lower = task_finish_perc[lower_key][i]
            finish_upper = task_finish_perc[upper_key][i]

            mean_start = mean_start_times[i]
            mean_finish = mean_finish_times[i]

            # Draw confidence band
            ax.barh(i, finish_upper - start_lower, left=start_lower, height=0.3,
                    alpha=0.3, color=color,
                    label=f"{confidence_level}% Confidence" if i == 0 else "")

            # Draw mean duration bar
            mean_duration = mean_finish - mean_start
            ax.barh(i, mean_duration, left=mean_start, height=0.6,
                    alpha=0.8, color=color, edgecolor='black', linewidth=1)

            # Add task name and criticality
            task_name = row['Task']

            if show_criticality and "task_criticality" in mc_results:
                criticality = mc_results["task_criticality"]
                if i < len(criticality):
                    crit_value = criticality[i]
                    if crit_value > 10:
                        label = f"{task_name} ({crit_value:.0f}% critical)"
                        color_text = 'red' if crit_value > 80 else 'orange' if crit_value > 50 else 'black'
                    else:
                        label = task_name
                        color_text = 'black'
                else:
                    label = task_name
                    color_text = 'black'
            else:
                label = task_name
                color_text = 'black'

            ax.text(mean_finish + 0.5, i, label, va='center', fontsize=9, color=color_text)

        except IndexError as e:
            st.error(f"❌ Error plotting task {i + 1}: {e}")
            st.write(f"Task index: {i}, Array sizes:")
            st.write(f"  start_lower: {len(task_start_perc[lower_key])}")
            st.write(f"  finish_lower: {len(task_finish_perc[lower_key])}")
            continue
        except Exception as e:
            st.error(f"❌ Unexpected error plotting task {i + 1}: {e}")
            continue

    # Format chart
    ax.set_yticks(range(num_tasks))
    ax.set_yticklabels([f"Task {i + 1}" for i in range(num_tasks)])
    ax.invert_yaxis()
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Tasks")
    ax.set_title(f"Monte Carlo Gantt Chart ({confidence_level}% Confidence Bands)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig, use_container_width=True)


# Add these functions to the end of view.py

def debug_monte_carlo_results(model):
    """Debug function to inspect Monte Carlo results structure"""

    if not model.simulation_data.get("monte_carlo_results"):
        return "No Monte Carlo results found"

    mc_results = model.simulation_data["monte_carlo_results"]
    task_df = model.task_df

    debug_info = {
        "num_tasks_in_df": len(task_df),
        "monte_carlo_keys": list(mc_results.keys()),
    }

    # Check key arrays
    for key in ["task_criticality", "mean_start_times", "mean_finish_times"]:
        if key in mc_results:
            value = mc_results[key]
            if hasattr(value, '__len__'):
                debug_info[f"{key}_length"] = len(value)
            else:
                debug_info[f"{key}_type"] = type(value)

    # Check percentiles structure
    if "task_start_percentiles" in mc_results:
        perc_data = mc_results["task_start_percentiles"]
        debug_info["percentile_keys"] = list(perc_data.keys())
        for key, values in perc_data.items():
            if hasattr(values, '__len__'):
                debug_info[f"percentile_{key}_length"] = len(values)

    return debug_info


def render_monte_carlo_debug_info(model):
    """Render debug information for Monte Carlo issues"""

    st.subheader("🔍 Monte Carlo Debug Information")

    debug_info = debug_monte_carlo_results(model)

    if isinstance(debug_info, str):
        st.write(debug_info)
        return

    st.write("**Task DataFrame Info:**")
    st.write(f"Number of tasks: {debug_info.get('num_tasks_in_df', 'Unknown')}")

    st.write("**Monte Carlo Results Structure:**")
    for key, value in debug_info.items():
        if key != 'num_tasks_in_df':
            st.write(f"  {key}: {value}")

    # Show first few tasks from DataFrame
    st.write("**Task DataFrame Preview:**")
    st.dataframe(model.task_df[["ID", "Task", "Duration (days)"]].head())



# -------------------------------
# Eigenvalue Tab
# -------------------------------
def render_eigenvalue_tab(model):
    if not model.simulation_data.get("tasks"):
        st.info("Please run the simulation to view eigenvalue data.")
        return

    st.subheader("Eigenvalue Analysis")
    eigenvalues, second_eigenvalue, error = model.compute_eigenvalues()
    if error:
        st.error(error)
        return
    num_tasks = len(model.simulation_data["tasks"])
    adjacency = model.simulation_data["adjacency"]

    st.write("### Eigenvalues of Adjacency Matrix")
    st.write("The eigenvalues represent the spectral properties of the task dependency structure.")
    df_eigen = pd.DataFrame({"Eigenvalue": np.real(eigenvalues), "Imaginary": np.imag(eigenvalues)})
    st.table(df_eigen)
    st.write(f"### Second Eigenvalue (Fiedler Value) for Centrality: {second_eigenvalue:.4f}")
    st.write("Note: The second smallest eigenvalue of the Laplacian indicates graph connectivity/centrality.")

    fig_bar, ax_bar = plt.subplots(figsize=(6, 4), facecolor='#fff')
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
    df_connectivity = pd.DataFrame(adjacency, index=[f"Task {i + 1}" for i in range(num_tasks)],
                                   columns=[f"Task {i + 1}" for i in range(num_tasks)])
    fig_heat, ax_heat = plt.subplots(figsize=(6, 4), facecolor='#fff')
    sns.heatmap(df_connectivity, annot=True, cmap="Oranges", cbar_kws={'label': 'Dependency Strength'}, ax=ax_heat)
    ax_heat.set_xlabel("Task ID", fontsize=12, fontfamily='Arial', color='#1a2a44')
    ax_heat.set_ylabel("Task ID", fontsize=12, fontfamily='Arial', color='#1a2a44')
    ax_heat.set_title("Adjacency Matrix Heatmap", fontsize=14, fontfamily='Arial', pad=10, color='#1a2a44')
    st.pyplot(fig_heat, use_container_width=True)

# -------------------------------
# PDE Gantt Tab with Critical Path
# -------------------------------
def render_pde_gantt(model):
    # DEFENSIVE VALIDATION - Add at the beginning
    print(f"PDE Gantt called with {len(model.task_df)} tasks")

    if not model.simulation_data.get("tasks"):
        st.info("Run the simulation to view the PDE Gantt chart.")
        print("No PDE simulation data found")
        return

        # Get arrays and validate they exist
    start_times = model.simulation_data.get("start_times_risk")
    finish_times = model.simulation_data.get("finish_times_risk")
    durations_risk = model.simulation_data.get("durations_risk")

    if start_times is None:
        st.error("PDE start times missing. Run simulation first.")
        print("No PDE start times found")
        return

    if finish_times is None:
        st.error("PDE finish times missing. Run simulation first.")
        print("No PDE finish times found")
        return

    if durations_risk is None:
        st.error("PDE durations missing. Run simulation first.")
        print("No PDE durations found")
        return

    print(f"PDE arrays: start_times={len(start_times)}, finish_times={len(finish_times)}, durations={len(durations_risk)}")

        # Validate array sizes match task count
    if len(start_times) != len(model.task_df):
            st.error(f"Data mismatch detected: {len(model.task_df)} tasks but {len(start_times)} PDE start times. Please re-run simulation.")
            print(f"MISMATCH: {len(model.task_df)} tasks vs {len(start_times)} PDE start times")
            return

    if len(finish_times) != len(model.task_df):
            st.error(f"Data mismatch detected: {len(model.task_df)} tasks but {len(finish_times)} PDE finish times. Please re-run simulation.")
            print(f"MISMATCH: {len(model.task_df)} tasks vs {len(finish_times)} PDE finish times")
            return

    print("PDE Gantt validation passed")

    task_df = model.task_df.copy()
    tasks = model.simulation_data["tasks"]
    num_tasks = model.simulation_data["num_tasks"]
    start_times = model.simulation_data["start_times_risk"]
    finish_times = model.simulation_data["finish_times_risk"]
    durations_risk = model.simulation_data["durations_risk"]

    pde_completion_time = np.max(finish_times)
    st.subheader("PDE Gantt Chart")
    st.write(f"**Time to Completion:** PDE: {pde_completion_time:.1f} days")

    colors = plt.cm.Oranges(np.linspace(0.3, 1, num_tasks))
    fig_gantt, ax_gantt = plt.subplots(figsize=(8,5), facecolor='#fff')

    for i, row in task_df.iterrows():
        color = colors[i]
        bar = ax_gantt.barh(i, durations_risk[i], left=start_times[i], height=0.4,
                            align="center", color=color, edgecolor="#1a2a44", alpha=0.9)
        # Task name + duration at end of bar
        ax_gantt.text(start_times[i] + durations_risk[i] + 0.5, i,
                      f"{row['Task']} ({durations_risk[i]:.0f}d)",
                      ha="left", va="center", fontsize=8, fontfamily='Arial')
        # Hover info
        mplcursors.cursor(bar, hover=True).connect("add", lambda sel: sel.annotation.set_text(
            f"Task: {tasks[sel.index]}\nDuration: {durations_risk[sel.index]:.1f} days\nRisk: {row['Risk (0-5)']:.1f}"
        ))

    # Draw arrows
    adjacency = model.simulation_data["adjacency"]
    for i in range(num_tasks):
        preds = np.where(adjacency[:, i] > 0)[0]
        for p in preds:
            ax_gantt.annotate("",
                              xy=(start_times[i], i),
                              xytext=(finish_times[p], p),
                              arrowprops=dict(arrowstyle="->", color="#1a2a44", lw=1.5))

    ax_gantt.set_yticks(range(num_tasks))
    ax_gantt.set_yticklabels([f"{i+1}" for i in range(num_tasks)], fontsize=10, fontfamily='Arial')
    ax_gantt.invert_yaxis()
    ax_gantt.set_xlabel("Time (days)", fontsize=12, fontfamily='Arial')
    ax_gantt.set_ylabel("Task ID", fontsize=12, fontfamily='Arial')
    ax_gantt.set_title("Gantt Chart (PDE)", fontsize=14, fontfamily='Arial', pad=10)
    ax_gantt.grid(True, axis="x", linestyle="--", alpha=0.7, color='#2a4066')
    ax_gantt.set_facecolor('#fff')
    st.pyplot(fig_gantt, use_container_width=True)


# -------------------------------
# Eigenvalue Tab
# -------------------------------
def render_eigenvalue_tab(model):
    if not model.simulation_data.get("tasks"):
        st.info("Please run the simulation to view eigenvalue data.")
        return

    st.subheader("Eigenvalue Analysis")
    eigenvalues, second_eigenvalue, error = model.compute_eigenvalues()
    if error:
        st.error(error)
        return
    num_tasks = len(model.simulation_data["tasks"])
    adjacency = model.simulation_data["adjacency"]

    st.write("### Eigenvalues of Adjacency Matrix")
    st.write("The eigenvalues represent the spectral properties of the task dependency structure.")
    df_eigen = pd.DataFrame({"Eigenvalue": np.real(eigenvalues), "Imaginary": np.imag(eigenvalues)})
    st.table(df_eigen)
    st.write(f"### Second Eigenvalue (Fiedler Value) for Centrality: {second_eigenvalue:.4f}")
    st.write("Note: The second smallest eigenvalue of the Laplacian indicates graph connectivity/centrality.")

    fig_bar, ax_bar = plt.subplots(figsize=(6, 4), facecolor='#fff')
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
    df_connectivity = pd.DataFrame(adjacency, index=[f"Task {i + 1}" for i in range(num_tasks)],
                                   columns=[f"Task {i + 1}" for i in range(num_tasks)])
    fig_heat, ax_heat = plt.subplots(figsize=(6, 4), facecolor='#fff')
    sns.heatmap(df_connectivity, annot=True, cmap="Oranges", cbar_kws={'label': 'Dependency Strength'}, ax=ax_heat)
    ax_heat.set_xlabel("Task ID", fontsize=12, fontfamily='Arial', color='#1a2a44')
    ax_heat.set_ylabel("Task ID", fontsize=12, fontfamily='Arial', color='#1a2a44')
    ax_heat.set_title("Adjacency Matrix Heatmap", fontsize=14, fontfamily='Arial', pad=10, color='#1a2a44')
    st.pyplot(fig_heat, use_container_width=True)

#---------------------------------
# SDE Calculations
#---------------------------------
def render_sde_gantt(model, controller):
    """Render SDE Gantt chart with multiple realizations and confidence bands"""

    st.subheader("🌊 SDE Stochastic Simulation")

    # Check if we have SDE results - look for the sde_results key directly
    sde_results = model.simulation_data.get("sde_results")

    # Also check for SDE-specific data keys
    has_sde_data = (
            model.simulation_data.get("sde_start_times") is not None and
            model.simulation_data.get("sde_finish_times") is not None
    )

    if sde_results is None or not has_sde_data:
        st.info("Run SDE simulation to view stochastic Gantt chart with confidence bands.")

        # SDE Parameter Controls
        with st.expander("⚙️ SDE Simulation Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                n_paths = st.slider("Number of Paths", 50, 1000, 500, 50,
                                    help="More paths = better statistics but slower computation")
                volatility = st.slider("Volatility", 0.05, 0.5, 0.15, 0.05,
                                       help="Base uncertainty level for all tasks")

            with col2:
                correlation_strength = st.slider("Dependency Correlation", 0.0, 1.0, 0.5, 0.1,
                                                 help="How much dependencies correlate uncertainty")
                dt = st.slider("Time Step", 0.01, 0.1, 0.01, 0.01,
                               help="Simulation time step (smaller = more accurate)")

            with col3:
                jump_intensity = st.slider("Jump Intensity", 0.0, 0.5, 0.1, 0.05,
                                           help="Rate of sudden disruptions")
                risk_amplification = st.slider("Risk Amplification", 0.5, 3.0, 1.5, 0.1,
                                               help="How much risk levels affect uncertainty")

        # Run SDE Simulation Button
        if st.button("🚀 Run SDE Simulation", type="primary", key="run_sde"):
            st.write("🔍 Debug: Creating SDE parameters...")

            # Create SDE parameters
            sde_params = SDEParameters(
                dt=dt,
                T=max(100, np.sum(model.task_df["Duration (days)"]) * 2),
                n_paths=n_paths,
                volatility=volatility,
                correlation_strength=correlation_strength,
                jump_intensity=jump_intensity,
                risk_amplification=risk_amplification
            )

            st.write(
                f"🔍 Debug: Parameters created - T={sde_params.T}, dt={sde_params.dt}, n_paths={sde_params.n_paths}")
            st.write(f"🔍 Debug: This will require {int(sde_params.T / sde_params.dt):,} time steps per path")

            st.write("🔍 Debug: Calling controller...")

            # Run simulation
            with st.spinner("Running SDE simulation..."):
                success, error = controller.run_sde_simulation(sde_params)

                st.write(f"🔍 Debug: Controller returned - success={success}, error={error}")

                if success:
                    st.success("SDE simulation completed successfully!")
                    # Debug: Show what data is actually stored
                    st.write("🔍 Debug: Data stored in simulation_data:")
                    sde_keys = [k for k in model.simulation_data.keys() if 'sde' in k.lower()]
                    st.write(f"SDE keys found: {sde_keys}")
                    st.rerun()
                else:
                    st.error(f"SDE simulation failed: {error}")
        return

    # Use SDE-specific data from the new storage format
    start_times = model.simulation_data.get("sde_start_times")
    finish_times = model.simulation_data.get("sde_finish_times")
    durations = model.simulation_data.get("sde_durations")

    # Display SDE Results
    st.success("✅ SDE Simulation Results Available")

    # Debug info to confirm data availability
    st.write(f"🔍 Debug: SDE data shapes - start_times: {start_times.shape if start_times is not None else 'None'}")
    st.write(f"🔍 Debug: SDE data shapes - finish_times: {finish_times.shape if finish_times is not None else 'None'}")

    # Risk Summary Metrics
    risk_summary = controller.get_sde_risk_summary()
    if risk_summary:
        metrics = risk_summary.get("sde_metrics", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            mean_duration = metrics.get('mean_project_duration', 0)
            st.metric("Mean Project Duration", f"{mean_duration:.1f} days")

        with col2:
            std_duration = metrics.get('std_project_duration', 0)
            cv = metrics.get('cv_project_duration', 0)
            st.metric("Standard Deviation", f"{std_duration:.1f} days",
                      delta=f"CV: {cv:.2f}")

        with col3:
            prob_on_time = metrics.get('prob_on_time', 0)
            st.metric("On-Time Probability", f"{prob_on_time:.1%}")

        with col4:
            var_95 = metrics.get('var_95', 0)
            st.metric("95% VaR", f"{var_95:.1f} days")

    # Visualization Options
    viz_option = st.selectbox(
        "Choose Visualization",
        ["Confidence Band Gantt", "Multiple Realizations", "Completion Time Distributions", "Risk Analysis"],
        key="sde_viz_option"
    )

    if viz_option == "Confidence Band Gantt":
        render_sde_confidence_gantt(sde_results, model.task_df)
    elif viz_option == "Multiple Realizations":
        render_sde_realizations(sde_results, model.task_df)
    elif viz_option == "Completion Time Distributions":
        render_sde_distributions(sde_results, model.task_df)
    elif viz_option == "Risk Analysis":
        render_sde_risk_analysis(sde_results, risk_summary)


# TEMPORARY DEBUG FUNCTION - Add this to your view.py temporarily
def debug_simulation_data_keys(model):
    """Temporary debug function to see all simulation data keys"""
    st.write("🔍 **DEBUG: All simulation_data keys:**")
    for key, value in model.simulation_data.items():
        if value is not None:
            if hasattr(value, 'shape'):
                st.write(f"✅ {key}: {type(value).__name__} shape {value.shape}")
            elif hasattr(value, '__len__'):
                st.write(f"✅ {key}: {type(value).__name__} length {len(value)}")
            else:
                st.write(f"✅ {key}: {type(value).__name__}")
        else:
            st.write(f"❌ {key}: None")

def render_sde_confidence_gantt(sde_results, task_df):
    """Render Gantt chart with confidence bands"""

    st.subheader("🎯 Confidence Band Gantt Chart")

    n_tasks = len(task_df)
    task_names = task_df["Task"].tolist()

    # Calculate percentiles for completion times
    completion_times = sde_results.completion_times
    percentiles = [5, 25, 50, 75, 95]
    completion_percentiles = np.percentile(completion_times, percentiles, axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(6, n_tasks * 0.6)))

    colors = ['#ffcccc', '#ff9999', '#ff6666', '#ff9999', '#ffcccc']
    labels = ['5-95%', '25-75%', '50% (Median)', '25-75%', '5-95%']

    for i in range(n_tasks):
        y_pos = n_tasks - i - 1

        # Draw confidence bands
        for j in range(len(percentiles) // 2):
            left_perc = completion_percentiles[j, i]
            right_perc = completion_percentiles[-(j + 1), i]
            width = right_perc - left_perc

            ax.barh(y_pos, width, left=left_perc, height=0.6,
                    color=colors[j], alpha=0.7,
                    label=labels[j] if i == 0 else "")

        # Median line
        median_time = completion_percentiles[2, i]
        ax.axvline(x=median_time, ymin=(y_pos - 0.3) / (n_tasks), ymax=(y_pos + 0.3) / (n_tasks),
                   color='red', linewidth=2)

        # Task label
        ax.text(median_time + 1, y_pos, f"{task_names[i]}",
                va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels([f"Task {i + 1}" for i in range(n_tasks)])
    ax.set_xlabel("Time (days)")
    ax.set_title("SDE Project Schedule with Confidence Bands")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    st.pyplot(fig, use_container_width=True)

    # Show percentile table
    st.subheader("📋 Completion Time Percentiles")
    perc_df = pd.DataFrame(
        completion_percentiles.T,
        columns=[f"{p}%" for p in percentiles],
        index=[f"{name}" for name in task_names]
    )
    perc_df = perc_df.round(1)
    st.dataframe(perc_df, use_container_width=True)


def render_sde_realizations(sde_results, task_df, n_show=10):
    """Show multiple individual simulation paths"""

    st.subheader("🎲 Individual Simulation Realizations")

    n_show = st.slider("Number of paths to show", 1, 50, 10, 1)

    n_tasks = len(task_df)
    task_names = task_df["Task"].tolist()
    completion_times = sde_results.completion_times

    # Select random paths to show
    selected_paths = np.random.choice(completion_times.shape[1], n_show, replace=False)

    fig, ax = plt.subplots(figsize=(12, max(6, n_tasks * 0.6)))

    colors = plt.cm.Set3(np.linspace(0, 1, n_show))

    for path_idx, path in enumerate(selected_paths):
        for i in range(n_tasks):
            y_pos = n_tasks - i - 1
            completion_time = completion_times[i, path]

            # Simple bar for this realization
            ax.barh(y_pos, completion_time, height=0.8 / n_show,
                    left=path_idx * 0.8 / n_show - 0.4,  # Changed from 'bottom' to 'left'
                    color=colors[path_idx], alpha=0.7,
                    label=f"Path {path + 1}" if i == 0 else "")

    # Add task names
    for i in range(n_tasks):
        y_pos = n_tasks - i - 1
        ax.text(ax.get_xlim()[1] * 0.02, y_pos, task_names[i],
                va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels([f"Task {i + 1}" for i in range(n_tasks)])
    ax.set_xlabel("Completion Time (days)")
    ax.set_title(f"Individual SDE Simulation Paths (showing {n_show} of {completion_times.shape[1]})")

    if n_show <= 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    st.pyplot(fig, use_container_width=True)


def render_sde_distributions(sde_results, task_df):
    """Show completion time distributions for each task"""

    st.subheader("📈 Completion Time Distributions")

    completion_times = sde_results.completion_times
    project_times = sde_results.project_completion_times
    task_names = task_df["Task"].tolist()

    # Choose what to plot
    plot_option = st.selectbox(
        "Select distribution to view:",
        ["Project Completion Time", "Individual Task Times", "All Tasks Overlay"]
    )

    if plot_option == "Project Completion Time":
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(project_times, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')

        # Add statistics
        mean_time = np.mean(project_times)
        median_time = np.median(project_times)

        ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.1f}')
        ax.axvline(median_time, color='green', linestyle='--', linewidth=2, label=f'Median: {median_time:.1f}')

        ax.set_xlabel("Project Completion Time (days)")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of Project Completion Times")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig, use_container_width=True)

    elif plot_option == "Individual Task Times":
        # Select task to view
        task_idx = st.selectbox("Select task:", range(len(task_names)),
                                format_func=lambda x: f"Task {x + 1}: {task_names[x]}")

        fig, ax = plt.subplots(figsize=(10, 6))

        task_times = completion_times[task_idx, :]
        ax.hist(task_times, bins=30, density=True, alpha=0.7, color='lightcoral', edgecolor='black')

        mean_time = np.mean(task_times)
        median_time = np.median(task_times)

        ax.axvline(mean_time, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_time:.1f}')
        ax.axvline(median_time, color='green', linestyle='--', linewidth=2, label=f'Median: {median_time:.1f}')

        ax.set_xlabel("Completion Time (days)")
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution: {task_names[task_idx]}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        st.pyplot(fig, use_container_width=True)

    else:  # All Tasks Overlay
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(task_names)))

        for i, (task_times, color) in enumerate(zip(completion_times, colors)):
            ax.hist(task_times, bins=30, density=True, alpha=0.6,
                    color=color, label=f"Task {i + 1}: {task_names[i]}")

        ax.set_xlabel("Completion Time (days)")
        ax.set_ylabel("Density")
        ax.set_title("Completion Time Distributions - All Tasks")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig, use_container_width=True)


def render_sde_risk_analysis(sde_results, risk_summary):
    """Show detailed risk analysis from SDE results"""

    st.subheader("⚠️ Risk Analysis Dashboard")

    if not risk_summary:
        st.warning("No risk summary available")
        return

    metrics = risk_summary.get("sde_metrics", {})
    confidence_intervals = risk_summary.get("confidence_intervals", {})

    # Risk Metrics Table
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Key Risk Metrics")

        risk_data = {
            "Metric": [
                "Mean Project Duration",
                "Standard Deviation",
                "Coefficient of Variation",
                "Schedule Risk Factor",
                "95% Value at Risk",
                "Expected Shortfall",
                "Probability On-Time"
            ],
            "Value": [
                f"{metrics.get('mean_project_duration', 0):.1f} days",
                f"{metrics.get('std_project_duration', 0):.1f} days",
                f"{metrics.get('cv_project_duration', 0):.3f}",
                f"{metrics.get('schedule_risk_factor', 1):.2f}x",
                f"{metrics.get('var_95', 0):.1f} days",
                f"{metrics.get('expected_shortfall', 0):.1f} days",
                f"{metrics.get('prob_on_time', 0):.1%}"
            ]
        }

        st.dataframe(pd.DataFrame(risk_data), use_container_width=True)

    with col2:
        st.subheader("📈 Confidence Intervals")

        # Show project completion confidence interval
        project_ci = confidence_intervals.get('project_completion', (0, 0))

        fig, ax = plt.subplots(figsize=(8, 4))

        # Simple visualization of confidence interval
        mean_val = metrics.get('mean_project_duration', 0)
        lower, upper = project_ci

        ax.errorbar([0], [mean_val], yerr=[[mean_val - lower], [upper - mean_val]],
                    fmt='o', capsize=10, capthick=2, markersize=8, color='red')

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylabel("Project Duration (days)")
        ax.set_title("90% Confidence Interval")
        ax.set_xticks([])
        ax.grid(True, alpha=0.3)

        # Add text annotations
        ax.text(0.1, mean_val, f"Mean: {mean_val:.1f}", va='center')
        ax.text(0.1, lower, f"5%: {lower:.1f}", va='center')
        ax.text(0.1, upper, f"95%: {upper:.1f}", va='center')

        st.pyplot(fig, use_container_width=True)

    # Simulation Parameters
    st.subheader("⚙️ Simulation Parameters Used")
    sim_params = risk_summary.get("simulation_params", {})

    param_cols = st.columns(3)
    with param_cols[0]:
        st.metric("Number of Paths", sim_params.get('n_paths', 'N/A'))
    with param_cols[1]:
        st.metric("Volatility", f"{sim_params.get('volatility', 0):.3f}")
    with param_cols[2]:
        st.metric("Correlation Strength", f"{sim_params.get('correlation_strength', 0):.2f}")
