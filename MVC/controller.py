# controller.py - Clean working version
import streamlit as st
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List
from dataclasses import dataclass

from matplotlib import pyplot as plt

from project_templates import ProjectTemplates


# ====== BASE CONTROLLER ======
class BaseController(ABC):
    """Base controller with common functionality"""

    def __init__(self, model):
        self.model = model
        self._observers = []

    def add_observer(self, observer):
        """Add observer for MVC updates"""
        self._observers.append(observer)

    def notify_observers(self, event_type: str, data: Dict[str, Any] = None):
        """Notify all observers of model changes"""
        for observer in self._observers:
            observer.on_model_change(event_type, data or {})

    def handle_error(self, error: Exception) -> Tuple[bool, str]:
        """Standard error handling"""
        return False, f"Error: {str(error)}"


# ====== TASK CONTROLLER ======
class TaskController(BaseController):
    """Handles task-related operations"""

    def add_task(self, name: str, duration: float,
                 dependencies: List[int] = None,
                 risk_level: float = 0.0,
                 parent_id: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """Add a new task"""
        try:
            # For compatibility with existing model
            new_id = len(self.model.task_df) + 1
            new_row = {
                "Select": False,
                "ID": new_id,
                "Task": name,
                "Duration (days)": duration,
                "Dependencies (IDs)": ",".join(map(str, dependencies or [])),
                "Risk (0-5)": risk_level,
                "Parent ID": parent_id or ""
            }
            self.model.task_df = pd.concat([self.model.task_df, pd.DataFrame([new_row])], ignore_index=True)
            self.model.save_tasks()
            self.notify_observers("task_added", {"task_id": new_id})
            return True, None
        except Exception as e:
            return self.handle_error(e)

    def update_task_from_dataframe(self, task_df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """Update tasks from DataFrame (for UI compatibility)"""
        try:
            self.model.task_df = task_df
            self.model.save_tasks()
            self.notify_observers("tasks_updated", {})
            return True, None
        except Exception as e:
            return self.handle_error(e)

    def delete_selected_tasks(self) -> Tuple[bool, Optional[str]]:
        """Delete tasks marked as selected"""
        try:
            selected_indices = self.model.task_df[self.model.task_df["Select"] == True].index
            if not selected_indices.empty:
                deleted_ids = self.model.task_df.loc[selected_indices, "ID"].astype(str).tolist()
                self.model.task_df = self.model.task_df.drop(selected_indices).reset_index(drop=True)
                self.model.task_df["ID"] = pd.Series(range(1, len(self.model.task_df) + 1))

                # Update dependencies
                for i in range(len(self.model.task_df)):
                    deps = self.model.task_df.loc[i, "Dependencies (IDs)"]
                    if isinstance(deps, str) and deps.strip() != "":
                        dep_list = deps.split(",")
                        new_deps = []
                        for dep in dep_list:
                            try:
                                dep_id = int(dep.strip())
                                if str(dep_id) not in deleted_ids:
                                    new_id = self.model.task_df[self.model.task_df["ID"] == dep_id].index
                                    if len(new_id) > 0:
                                        new_deps.append(str(new_id[0] + 1))
                            except ValueError:
                                continue
                        self.model.task_df.loc[i, "Dependencies (IDs)"] = ",".join(new_deps)

                self.model.save_tasks()
                self.notify_observers("tasks_deleted", {"count": len(selected_indices)})
                return True, None
            else:
                return False, "No tasks selected for deletion"
        except Exception as e:
            return self.handle_error(e)

    def validate_tasks(self) -> Tuple[bool, List[str]]:
        """Validate all tasks"""
        return self.model.validate_tasks()

    def load_template(self, template_name: str) -> Tuple[bool, Optional[str]]:
        """Load a project template"""
        try:
            new_df = ProjectTemplates.get_template(template_name)
            self.model.task_df = new_df
            self.model.save_tasks()
            self.notify_observers("tasks_updated", {})
            return True, None
        except Exception as e:
            return self.handle_error(e)


# ====== SIMULATION CONTROLLER ======
class SimulationController(BaseController):
    """Handles simulation operations"""

    def run_simulation(self, diffusion=0.02, reaction_multiplier=2.0, max_delay=0.05, maxed=False) -> Tuple[bool, Optional[str]]:
        """Run complete simulation - SIMPLIFIED VERSION"""
        try:
            valid, errors = self.model.validate_tasks()
            if not valid:
                return False, "; ".join(errors)

            # Run classical schedule
            num_tasks = len(self.model.task_df)
            start_times_classical = np.zeros(num_tasks)
            finish_times_classical = np.zeros(num_tasks)

            for i, row in self.model.task_df.iterrows():
                deps = str(row['Dependencies (IDs)']).split(",") if row['Dependencies (IDs)'] else []
                deps = [int(d.strip()) - 1 for d in deps if d.strip().isdigit()]
                if deps:
                    start_times_classical[i] = max(finish_times_classical[d] for d in deps)
                else:
                    start_times_classical[i] = 0
                finish_times_classical[i] = start_times_classical[i] + row['Duration (days)']

            self.model.simulation_data["start_times_classical"] = start_times_classical
            self.model.simulation_data["finish_times_classical"] = finish_times_classical

            # Run PDE simulation
            if maxed:
                success, error = self.model.run_pde_simulation_maxed()
            else:
                success, error = self.model.run_pde_simulation(diffusion, reaction_multiplier, max_delay)

            if not success:
                return False, error

            # Run classical risk calculation
            success_classical, error_classical = self.model.run_classical_simulation()
            if not success_classical:
                return False, error_classical

            self.notify_observers("simulation_completed", {
                "diffusion": diffusion,
                "reaction_multiplier": reaction_multiplier,
                "max_delay": max_delay,
                "maxed": maxed
            })
            return True, None

        except Exception as e:
            return self.handle_error(e)

    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        return {
            "has_results": self.model.simulation_data.get("tasks") is not None,
            "has_classical": self.model.simulation_data.get("classical_risk") is not None,
            "task_count": len(self.model.task_df)
        }

# ====== Monte Carlo CONTROLLER ======#

    def run_monte_carlo(self, num_simulations=1000, confidence_levels=[90]) -> Tuple[bool, Optional[str]]:
        """Run Monte Carlo simulation"""
        try:
            # Validate tasks first - call on model, not controller
            valid, errors = self.model.validate_tasks()
            if not valid:
                return False, "; ".join(errors)

            # Run the Monte Carlo simulation
            success, error = self.model.run_monte_carlo_simulation(
                num_simulations=num_simulations,
                confidence_levels=confidence_levels
            )

            if not success:
                return False, error

            self.notify_observers("monte_carlo_completed", {
                "num_simulations": num_simulations,
                "confidence_levels": confidence_levels
            })

            return True, None

        except Exception as e:
            return self.handle_error(e)

    def _calculate_monte_carlo_schedule(self, task_df, durations):
            """Calculate schedule for one Monte Carlo sample"""
            num_tasks = len(task_df)
            start_times = np.zeros(num_tasks)
            finish_times = np.zeros(num_tasks)

            # Forward pass scheduling (same logic as classical)
            for i, row in task_df.iterrows():
                deps = str(row['Dependencies (IDs)']).split(",") if row['Dependencies (IDs)'] else []
                deps = [int(d.strip()) - 1 for d in deps if d.strip().isdigit()]
                if deps:
                    start_times[i] = max(finish_times[d] for d in deps)
                else:
                    start_times[i] = 0
                finish_times[i] = start_times[i] + durations[i]

            # Find critical path (tasks with zero float)
            total_float = self._calculate_float(start_times, finish_times, task_df)
            critical_path = np.where(np.abs(total_float) < 0.001)[0]

            return start_times, finish_times, critical_path

    def _calculate_float(self, start_times, finish_times, task_df):
            """Calculate total float for tasks using proper backward pass"""
            num_tasks = len(task_df)
            durations = finish_times - start_times
            project_duration = np.max(finish_times)

            # Initialize late finish times
            late_finish = np.full(num_tasks, project_duration)

            # Backward pass to calculate late finish times
            # Process tasks in reverse dependency order
            for i in reversed(range(num_tasks)):
                # Find successors of task i
                successors = []
                for j, row in task_df.iterrows():
                    deps = str(row['Dependencies (IDs)']).split(",") if row['Dependencies (IDs)'] else []
                    deps = [int(d.strip()) - 1 for d in deps if d.strip().isdigit()]
                    if i in deps:
                        successors.append(j)

                if successors:
                    # Late finish = minimum of successors' late start times
                    successor_late_starts = [late_finish[s] - durations[s] for s in successors]
                    late_finish[i] = min(successor_late_starts)

            # Calculate late start times
            late_start = late_finish - durations

            # Total float = late start - early start
            total_float = late_start - start_times

            return total_float

    # Add to view.py - Enhanced Monte Carlo Gantt Chart

    def render_monte_carlo_gantt_chart(model):
        """Render Monte Carlo Gantt chart with confidence bands"""

        if not model.simulation_data.get("monte_carlo_results"):
            st.info("🎲 Run Monte Carlo analysis to view probabilistic Gantt chart")
            return

        st.subheader("📊 Monte Carlo Gantt Chart")

        # Get Monte Carlo results
        mc_results = model.simulation_data["monte_carlo_results"]
        task_df = model.task_df
        num_tasks = len(task_df)

        # Controls
        col1, col2 = st.columns(2)
        with col1:
            confidence_level = st.selectbox(
                "Confidence Level for Bands",
                options=mc_results["confidence_levels"],
                index=0
            )
        with col2:
            show_criticality = st.checkbox("Show Critical Path Probability", value=True)

        # Create Gantt chart
        fig, ax = plt.subplots(figsize=(12, max(6, num_tasks * 0.4)))

        # Color scheme
        colors = plt.cm.Set3(np.linspace(0, 1, num_tasks))

        # Get percentile keys for selected confidence level
        lower_key = f"P{int((100 - confidence_level) / 2)}"
        upper_key = f"P{int(100 - (100 - confidence_level) / 2)}"

        for i, row in task_df.iterrows():
            color = colors[i]

            # Get confidence intervals
            start_lower = mc_results["task_start_percentiles"][lower_key][i]
            start_upper = mc_results["task_start_percentiles"][upper_key][i]
            finish_lower = mc_results["task_finish_percentiles"][lower_key][i]
            finish_upper = mc_results["task_finish_percentiles"][upper_key][i]

            mean_start = mc_results["mean_start_times"][i]
            mean_finish = mc_results["mean_finish_times"][i]

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
            criticality = mc_results["task_criticality"][i]

            if show_criticality and criticality > 10:
                label = f"{task_name} ({criticality:.0f}% critical)"
                color_text = 'red' if criticality > 80 else 'orange' if criticality > 50 else 'black'
            else:
                label = task_name
                color_text = 'black'

            ax.text(mean_finish + 0.5, i, label, va='center', fontsize=9, color=color_text)

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

# ====== IMPORT/EXPORT CONTROLLER ======
class ImportExportController(BaseController):
    """Handles import/export operations"""

    def import_mpp(self, file) -> Tuple[bool, Optional[str]]:
        """Import MPP file"""
        try:
            return self.model.import_mpp(file)
        except Exception as e:
            return self.handle_error(e)

    def export_mpp(self) -> Tuple[Optional[str], Optional[str]]:
        """Export to MPP format"""
        try:
            return self.model.export_mpp()
        except Exception as e:
            return None, str(e)


# ====== MAIN CONTROLLER ======
class MainController:
    """Main controller coordinating all operations"""

    def __init__(self, model):
        self.model = model
        self.task_controller = TaskController(model)
        self.simulation_controller = SimulationController(model)
        self.import_export_controller = ImportExportController(model)

        # ADD THIS LINE - SDE Integration
        from sde_solver import SDEModelIntegration
        self.sde_integration = SDEModelIntegration(model)

        # Set up notifications
        self.task_controller.add_observer(self)
        self.simulation_controller.add_observer(self)

    # Replace your on_model_change method in controller.py with this:

    def on_model_change(self, event_type: str, data: Dict[str, Any]):
        """Handle model change notifications - BULLETPROOF VERSION with diagnostics"""
        print(f"🔍 Model change triggered: {event_type} with data: {data}")
        print(f"🔍 Current task count: {len(self.model.task_df)}")

        # Check current simulation data state before clearing
        if self.model.simulation_data.get("start_times_classical") is not None:
            current_classical_size = len(self.model.simulation_data["start_times_classical"])
            print(f"🔍 Current classical arrays size: {current_classical_size}")

        if event_type in ["task_added", "task_updated", "tasks_deleted", "tasks_imported", "template_loaded"]:
            # Clear ALL simulation results when tasks change
            self.model.simulation_data = {
                # Basic simulation data
                "tasks": None,
                "adjacency": None,
                "num_tasks": None,

                # PDE simulation results
                "u_matrix": None,
                "start_times_risk": None,
                "finish_times_risk": None,
                "durations_risk": None,
                "simulation_time": None,
                "risk_curve": None,

                # Classical simulation results
                "classical_risk": None,
                "start_times_classical": None,
                "finish_times_classical": None,

                # Monte Carlo simulation results
                "monte_carlo_results": None,

                # SDE simulation results
                "sde_results": None,

                # Any other simulation data that might exist
                "eigenvalues": None,
                "critical_path": None,
                "task_statistics": None,
                "project_metrics": None
            }

            # Also clear any cached properties in the model itself
            if hasattr(self.model, 'simulation_results'):
                self.model.simulation_results = None

            print(f"🧹 Cleared all simulation data due to: {event_type}")
            print(f"🧹 All arrays reset to None")
        else:
            print(f"🔍 Event {event_type} did not trigger clearing")


    def run_sde_simulation(self, sde_params=None):
        """Run SDE simulation with optional parameters"""
        return self.sde_integration.run_sde_simulation(sde_params)

    def get_sde_risk_summary(self):
        """Get SDE risk analysis summary"""
        return self.sde_integration.get_risk_summary()


    # Delegate methods
    def add_task(self, **kwargs):
        return self.task_controller.add_task(**kwargs)

    def update_tasks(self, task_df):
        return self.task_controller.update_task_from_dataframe(task_df)

    def delete_selected_tasks(self):
        return self.task_controller.delete_selected_tasks()

    def validate_tasks(self):
        return self.task_controller.validate_tasks()

    def run_simulation(self, **kwargs):
        return self.simulation_controller.run_simulation(**kwargs)

    def get_simulation_status(self):
        return self.simulation_controller.get_simulation_status()

    def import_mpp(self, file):
        return self.import_export_controller.import_mpp(file)

    def export_mpp(self):
        return self.import_export_controller.export_mpp()

    def load_template(self, template_name):
        return self.task_controller.load_template(template_name)

    def run_monte_carlo(self, **kwargs):
        """Delegate Monte Carlo to simulation controller"""
        return self.simulation_controller.run_monte_carlo(**kwargs)