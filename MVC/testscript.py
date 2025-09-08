import numpy as np
import matplotlib.pyplot as plt

from model import TaskModel  # replace with your actual model import


# Add this to your model.py or wherever you're debugging

def debug_simulation_data(model):
    """Debug function to check simulation data availability"""
    print("=== SIMULATION DATA DEBUG ===")

    # Check if simulation_data exists
    if not hasattr(model, 'simulation_data'):
        print("❌ No simulation_data attribute")
        return

    data = model.simulation_data

    # Check each key
    keys_to_check = [
        "tasks", "simulation_time", "risk_curve", "classical_risk",
        "start_times_risk", "finish_times_risk", "u_matrix"
    ]

    for key in keys_to_check:
        value = data.get(key)
        if value is None:
            print(f"❌ {key}: None")
        elif isinstance(value, np.ndarray):
            print(f"✅ {key}: array shape {value.shape}, type {value.dtype}")
        elif isinstance(value, list):
            print(f"✅ {key}: list length {len(value)}")
        else:
            print(f"✅ {key}: {type(value)} - {value}")

    # Check if arrays match in length
    sim_time = data.get("simulation_time")
    risk_curve = data.get("risk_curve")
    classical_risk = data.get("classical_risk")

    if sim_time is not None and risk_curve is not None:
        print(f"📊 simulation_time length: {len(sim_time)}")
        print(f"📊 risk_curve length: {len(risk_curve)}")

        if classical_risk is not None:
            print(f"📊 classical_risk length: {len(classical_risk)}")
            if len(sim_time) != len(classical_risk):
                print("⚠️  LENGTH MISMATCH: simulation_time vs classical_risk")
        else:
            print("❌ classical_risk is None - this is likely your problem!")

    print("=== END DEBUG ===")


# Fixed version of your run_classical_simulation method
def run_classical_simulation_fixed(self):
    """Fixed version that ensures proper data alignment"""

    # Check if we have start_times_classical from the simulation controller
    start_times = self.simulation_data.get("start_times_classical")
    if start_times is None:
        print("Error: No start_times_classical found. Run simulation first.")
        return False, "Run simulation first"

    durations = self.task_df["Duration (days)"].to_numpy(dtype=float)
    risks = self.task_df["Risk (0-5)"].to_numpy(dtype=float)
    durations_risk = durations * np.maximum(1.0, risks)

    # Use the SAME simulation_time as PDE simulation - this is crucial
    simulation_time = self.simulation_data.get("simulation_time")
    if simulation_time is None:
        print("Error: No simulation_time from PDE simulation")
        return False, "Run PDE simulation first"

    print(f"Using simulation_time with {len(simulation_time)} points")

    # Calculate classical completion curve
    num_tasks = len(durations)
    u = np.zeros((num_tasks, len(simulation_time)))

    finish_times = start_times + durations_risk

    for i in range(num_tasks):
        for j, t in enumerate(simulation_time):
            if t < start_times[i]:
                u[i, j] = 0
            elif t < finish_times[i]:
                # Linear progress during task execution
                u[i, j] = (t - start_times[i]) / durations_risk[i]
            else:
                u[i, j] = 1

    classical_risk = u.mean(axis=0)

    print(f"Generated classical_risk with {len(classical_risk)} points")

    # Store the result - don't overwrite simulation_time!
    self.simulation_data["classical_risk"] = classical_risk

    return True, None


# Add this to your render_simulation_results function at the start
def render_simulation_results_with_debug(model):
    """Enhanced version with debugging"""

    # Debug first
    debug_simulation_data(model)

    # Check if we have the required data
    if not model.simulation_data.get("tasks"):
        st.info("Please run the simulation to view results.")
        return

    simulation_time = model.simulation_data.get("simulation_time")
    risk_curve = model.simulation_data.get("risk_curve")
    classical_risk = model.simulation_data.get("classical_risk")

    # More detailed error checking
    if simulation_time is None:
        st.error("simulation_time is None")
        return
    if risk_curve is None:
        st.error("risk_curve is None")
        return
    if classical_risk is None:
        st.error("classical_risk is None - classical simulation didn't run properly")
        return

    # Check lengths match
    if len(simulation_time) != len(risk_curve):
        st.error(f"Length mismatch: simulation_time({len(simulation_time)}) vs risk_curve({len(risk_curve)})")
        return
    if len(simulation_time) != len(classical_risk):
        st.error(f"Length mismatch: simulation_time({len(simulation_time)}) vs classical_risk({len(classical_risk)})")
        return

    st.subheader("Simulation Results")
    col1, col2 = st.columns(2)

    # --- 2D Plot ---
    with col1:
        st.subheader("2D Completion Plot")
        fig, ax = plt.subplots(figsize=(6, 4), facecolor='#fff')

        # Plot both curves
        ax.plot(simulation_time, classical_risk, color="#1976d2", lw=2, label="Classical Risk")
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
        fig_3d = plt.figure(figsize=(6, 4), facecolor='#fff')
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        ax_3d.plot(simulation_time, [0] * len(simulation_time), classical_risk,
                   color="#1976d2", lw=2, label="Classical Risk")
        ax_3d.plot(simulation_time, [1] * len(simulation_time), risk_curve,
                   color="#d32f2f", lw=2, linestyle='--', label="Diffusion Risk")

        ax_3d.set_xlabel("Time (days)")
        ax_3d.set_ylabel("Model (0=Classical,1=Diffusion)")
        ax_3d.set_zlabel("Average Completion (0–1)")
        ax_3d.set_title("3D Completion: Classical vs Diffusion")
        ax_3d.legend()
        st.pyplot(fig_3d, use_container_width=True)