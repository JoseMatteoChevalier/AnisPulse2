# models/task_model.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import json
import os


@dataclass
class Task:
    """Individual task data structure"""
    id: int
    name: str
    duration: float
    dependencies: List[int]
    risk_level: float
    parent_id: Optional[int] = None

    def __post_init__(self):
        if self.duration <= 0:
            raise ValueError(f"Task {self.id}: Duration must be positive")
        if not (0 <= self.risk_level <= 5):
            raise ValueError(f"Task {self.id}: Risk must be between 0-5")


@dataclass
class SimulationResults:
    """Container for simulation results"""
    tasks: List[Task]
    adjacency_matrix: np.ndarray
    u_matrix: np.ndarray
    start_times: np.ndarray
    finish_times: np.ndarray
    simulation_time: np.ndarray
    risk_curve: np.ndarray
    classical_risk: Optional[np.ndarray] = None


class TaskRepository:
    """Handles task data persistence"""

    def __init__(self):
        self._tasks: Dict[int, Task] = {}
        self._next_id = 1
        self._load_default_tasks()

    def _load_default_tasks(self):
        """Load default tasks if no saved tasks exist"""
        # Try to load from file first
        loaded_tasks = self._load_from_file()
        if loaded_tasks:
            for task_data in loaded_tasks:
                task = Task(
                    id=task_data['id'],
                    name=task_data['name'],
                    duration=task_data['duration'],
                    dependencies=task_data.get('dependencies', []),
                    risk_level=task_data.get('risk_level', 0.0),
                    parent_id=task_data.get('parent_id')
                )
                self._tasks[task.id] = task
                self._next_id = max(self._next_id, task.id + 1)
        else:
            # Load default project tasks
            default_tasks = [
                ("Requirements", 14, [], 0),
                ("Database Design", 21, [1], 0),
                ("Backend API", 28, [2], 1),
                ("Third-party Integration", 21, [3], 0),
                ("Frontend UI", 35, [2], 2),
                ("Testing & Deployment", 14, [4, 5], 0)
            ]

            for name, duration, deps, risk in default_tasks:
                self.add_task(name, duration, deps, risk)

    def _load_from_file(self) -> Optional[List[Dict]]:
        """Load tasks from JSON file - handles both old and new formats"""
        try:
            if os.path.exists("task_list.json"):
                with open("task_list.json", "r") as f:
                    data = json.load(f)

                # Handle old DataFrame format (list of dicts with DataFrame columns)
                if isinstance(data, list) and len(data) > 0:
                    first_item = data[0]
                    if 'ID' in first_item:  # Old format
                        converted_data = []
                        for item in data:
                            # Convert dependencies string to list
                            deps = []
                            if item.get('Dependencies (IDs)') and str(item['Dependencies (IDs)']).strip():
                                dep_str = str(item['Dependencies (IDs)'])
                                deps = [int(d.strip()) for d in dep_str.split(',') if d.strip().isdigit()]

                            converted_data.append({
                                'id': int(item['ID']),
                                'name': str(item['Task']),
                                'duration': float(item['Duration (days)']),
                                'dependencies': deps,
                                'risk_level': float(item.get('Risk (0-5)', 0.0)),
                                'parent_id': int(item['Parent ID']) if item.get('Parent ID') and str(
                                    item['Parent ID']).strip() else None
                            })
                        return converted_data
                    elif 'id' in first_item:  # New format
                        return data

                return data
        except Exception as e:
            print(f"Warning: Could not load tasks from file: {e}")
            pass
        return None

    def _save_to_file(self):
        """Save tasks to JSON file"""
        try:
            tasks_data = []
            for task in self._tasks.values():
                tasks_data.append({
                    'id': task.id,
                    'name': task.name,
                    'duration': task.duration,
                    'dependencies': task.dependencies,
                    'risk_level': task.risk_level,
                    'parent_id': task.parent_id
                })

            with open("task_list.json", "w") as f:
                json.dump(tasks_data, f, indent=2)
            return True
        except Exception:
            return False

    def add_task(self, name: str, duration: float, dependencies: List[int] = None,
                 risk_level: float = 0.0, parent_id: Optional[int] = None) -> Task:
        if dependencies is None:
            dependencies = []

        task = Task(
            id=self._next_id,
            name=name,
            duration=duration,
            dependencies=dependencies,
            risk_level=risk_level,
            parent_id=parent_id
        )
        self._tasks[self._next_id] = task
        self._next_id += 1
        self._save_to_file()
        return task

    def get_task(self, task_id: int) -> Optional[Task]:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        return list(self._tasks.values())

    def update_task(self, task_id: int, **kwargs) -> bool:
        if task_id not in self._tasks:
            return False

        task = self._tasks[task_id]
        for key, value in kwargs.items():
            if hasattr(task, key):
                setattr(task, key, value)

        self._save_to_file()
        return True

    def delete_task(self, task_id: int) -> bool:
        if task_id in self._tasks:
            del self._tasks[task_id]
            # Update dependencies that reference deleted task
            for task in self._tasks.values():
                task.dependencies = [dep for dep in task.dependencies if dep != task_id]
            self._save_to_file()
            return True
        return False

    def clear_all_tasks(self):
        """Clear all tasks"""
        self._tasks.clear()
        self._next_id = 1

    def update_from_dataframe(self, task_df: pd.DataFrame):
        """Update tasks from DataFrame (for UI compatibility)"""
        self.clear_all_tasks()

        for _, row in task_df.iterrows():
            dependencies = []
            if row.get('Dependencies (IDs)') and str(row['Dependencies (IDs)']).strip():
                dep_str = str(row['Dependencies (IDs)'])
                dependencies = [int(d.strip()) for d in dep_str.split(',') if d.strip().isdigit()]

            parent_id = None
            if row.get('Parent ID') and str(row['Parent ID']).strip():
                try:
                    parent_id = int(row['Parent ID'])
                except (ValueError, TypeError):
                    parent_id = None

            task = Task(
                id=int(row['ID']),
                name=str(row['Task']),
                duration=float(row['Duration (days)']),
                dependencies=dependencies,
                risk_level=float(row.get('Risk (0-5)', 0.0)),
                parent_id=parent_id
            )
            self._tasks[task.id] = task
            self._next_id = max(self._next_id, task.id + 1)

        self._save_to_file()

    def validate_dependencies(self) -> Tuple[bool, List[str]]:
        """Validate all task dependencies"""
        errors = []
        task_ids = set(self._tasks.keys())

        for task in self._tasks.values():
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    errors.append(f"Task {task.id}: Invalid dependency {dep_id}")
                if dep_id == task.id:
                    errors.append(f"Task {task.id}: Cannot depend on itself")

        # Check for circular dependencies
        if self._has_circular_dependencies():
            errors.append("Circular dependencies detected")

        return len(errors) == 0, errors

    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies using DFS"""

        def dfs(task_id, visited, rec_stack):
            visited.add(task_id)
            rec_stack.add(task_id)

            task = self._tasks.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    if dep_id not in visited:
                        if dfs(dep_id, visited, rec_stack):
                            return True
                    elif dep_id in rec_stack:
                        return True

            rec_stack.remove(task_id)
            return False

        visited = set()
        for task_id in self._tasks.keys():
            if task_id not in visited:
                if dfs(task_id, visited, set()):
                    return True
        return False


class PDESimulator:
    """Handles PDE simulation logic"""

    def __init__(self, dt: float = 0.01):
        self.dt = dt

    def simulate(self, tasks: List[Task], T: float,
                 diffusion: float = 0.02,
                 reaction_multiplier: float = 2.0,
                 max_delay: float = 0.05) -> SimulationResults:
        """Run PDE simulation"""

        # Build adjacency matrix
        adjacency = self._build_adjacency_matrix(tasks)

        # Calculate schedule
        start_times, finish_times, durations_risk = self._calculate_schedule(tasks)

        # Set up simulation
        T = max(T, np.max(finish_times) + 10)
        steps = max(1, int(T / self.dt))
        simulation_time = np.linspace(0, T, steps + 1)

        # Run PDE
        u_matrix = self._run_pde(tasks, adjacency, start_times, durations_risk,
                                 steps, diffusion, reaction_multiplier, max_delay)

        risk_curve = u_matrix.mean(axis=0)

        return SimulationResults(
            tasks=tasks,
            adjacency_matrix=adjacency,
            u_matrix=u_matrix,
            start_times=start_times,
            finish_times=finish_times,
            simulation_time=simulation_time,
            risk_curve=risk_curve
        )

    def simulate_maxed(self, tasks: List[Task], T: float) -> SimulationResults:
        """Run PDE simulation with maxed parameters"""

        # Build adjacency matrix
        adjacency = self._build_adjacency_matrix(tasks)

        # Calculate schedule
        start_times, finish_times, durations_risk = self._calculate_schedule(tasks)

        # Set up simulation with longer time horizon
        T = max(T, np.max(finish_times) + 20)
        steps = max(1, int(T / self.dt))
        simulation_time = np.linspace(0, T, steps + 1)

        # Run maxed PDE
        u_matrix = self._run_maxed_pde(tasks, adjacency, start_times, durations_risk, steps)

        risk_curve = u_matrix.mean(axis=0)

        return SimulationResults(
            tasks=tasks,
            adjacency_matrix=adjacency,
            u_matrix=u_matrix,
            start_times=start_times,
            finish_times=finish_times,
            simulation_time=simulation_time,
            risk_curve=risk_curve
        )

    def _build_adjacency_matrix(self, tasks: List[Task]) -> np.ndarray:
        """Build adjacency matrix from task dependencies"""
        num_tasks = len(tasks)
        adjacency = np.zeros((num_tasks, num_tasks))

        # Create ID to index mapping
        id_to_idx = {task.id: idx for idx, task in enumerate(tasks)}

        for i, task in enumerate(tasks):
            for dep_id in task.dependencies:
                if dep_id in id_to_idx:
                    j = id_to_idx[dep_id]
                    adjacency[j, i] = 1

        return adjacency

    def _calculate_schedule(self, tasks: List[Task]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate start/finish times using forward pass"""
        num_tasks = len(tasks)
        start_times = np.zeros(num_tasks)
        finish_times = np.zeros(num_tasks)
        durations_risk = np.array([task.duration * max(1.0, task.risk_level) for task in tasks])

        # Create ID to index mapping
        id_to_idx = {task.id: idx for idx, task in enumerate(tasks)}

        # Topological sort for proper scheduling order
        in_degree = np.zeros(num_tasks)
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in id_to_idx:
                    in_degree[id_to_idx[task.id]] += 1

        queue = [i for i in range(num_tasks) if in_degree[i] == 0]
        processed = []

        while queue:
            current = queue.pop(0)
            processed.append(current)
            task = tasks[current]

            # Calculate start time
            dep_finish_times = []
            for dep_id in task.dependencies:
                if dep_id in id_to_idx:
                    dep_idx = id_to_idx[dep_id]
                    dep_finish_times.append(finish_times[dep_idx])

            start_times[current] = max(dep_finish_times) if dep_finish_times else 0
            finish_times[current] = start_times[current] + durations_risk[current]

            # Update in-degrees
            for i, other_task in enumerate(tasks):
                if task.id in other_task.dependencies:
                    in_degree[i] -= 1
                    if in_degree[i] == 0:
                        queue.append(i)

        return start_times, finish_times, durations_risk

    def _run_pde(self, tasks: List[Task], adjacency: np.ndarray,
                 start_times: np.ndarray, durations_risk: np.ndarray,
                 steps: int, diffusion: float, reaction_multiplier: float,
                 max_delay: float) -> np.ndarray:
        """Core PDE simulation logic - ensures PDE is slower than classical due to risk propagation"""
        num_tasks = len(tasks)
        u = np.zeros((num_tasks, steps + 1))

        for t in range(steps):
            du = np.zeros(num_tasks)

            for i, task in enumerate(tasks):
                elapsed_time = t * self.dt - start_times[i]
                if elapsed_time < 0:
                    continue

                # Base progress rate (slower if task has high risk)
                if u[i, t] < 1.0:
                    # Reduce base progress rate based on task's own risk
                    risk_factor = 1 + task.risk_level * 0.5  # Higher risk = slower intrinsic progress
                    base_progress = self._reaction_rate(
                        u[i, t], durations_risk[i] * risk_factor, reaction_multiplier
                    )
                else:
                    base_progress = 0

                # Delay from predecessors (this is what makes PDE slower than classical)
                delay = 0
                preds = np.where(adjacency[:, i] > 0)[0]
                if len(preds) > 0:
                    pred_risks = np.array([tasks[p].risk_level for p in preds])
                    pred_remaining = 1.0 - u[preds, t]

                    if np.sum(pred_risks) > 0:
                        # Stronger delay effect - this ensures PDE is meaningfully slower
                        weighted_remaining = np.sum(pred_remaining * pred_risks) / np.sum(pred_risks)
                        delay = min(diffusion * weighted_remaining * 1.2, max_delay)

                        # Additional cascading delay for high-risk dependencies
                        if np.any(pred_risks >= 2.0):  # High risk predecessors
                            cascade_delay = min(0.02 * np.max(pred_risks), max_delay * 0.5)
                            delay += cascade_delay

                # Net progress (can be negative if delay is very high)
                net_progress = max(0, base_progress - delay)
                du[i] = net_progress

            # Update with smaller time steps for numerical stability
            u[:, t + 1] = np.clip(u[:, t] + du * self.dt, 0, 1.0)

            # Ensure no task completes before its classical completion time
            for i, task in enumerate(tasks):
                classical_completion_time = start_times[i] + task.duration * max(1.0, task.risk_level)
                current_time = (t + 1) * self.dt
                if current_time < classical_completion_time:
                    expected_progress = max(0, (current_time - start_times[i]) / (
                                task.duration * max(1.0, task.risk_level)))
                    u[i, t + 1] = min(u[i, t + 1], expected_progress)

        return u

    def _run_maxed_pde(self, tasks: List[Task], adjacency: np.ndarray,
                       start_times: np.ndarray, durations_risk: np.ndarray,
                       steps: int) -> np.ndarray:
        """Maxed-out PDE simulation"""
        num_tasks = len(tasks)
        u = np.zeros((num_tasks, steps + 1))

        for t in range(steps):
            du = np.zeros(num_tasks)

            for i, task in enumerate(tasks):
                elapsed_time = t * self.dt - start_times[i]
                if elapsed_time < 0:
                    continue

                # Base progress scaled aggressively by own risk
                if u[i, t] < 1.0:
                    base_progress = self._reaction_rate(
                        u[i, t], durations_risk[i] * (1 + task.risk_level), 1.0
                    )
                else:
                    base_progress = 0

                # Maxed-out downstream influence
                delay = 0
                preds = np.where(adjacency[:, i] > 0)[0]
                if len(preds) > 0:
                    pred_risks = np.array([tasks[p].risk_level for p in preds])
                    pred_remaining = 1.0 - u[preds, t]
                    if np.sum(pred_risks) > 0:
                        weighted_avg = np.sum(pred_remaining * pred_risks) / np.sum(pred_risks)
                        delay = min(weighted_avg * 1.5, 0.2)  # very strong influence

                du[i] = max(0, base_progress - delay)

            u[:, t + 1] = np.clip(u[:, t] + du * self.dt, 0, 1.0)

        return u

    def _reaction_rate(self, u_i: float, duration: float, multiplier: float) -> float:
        """Calculate reaction rate for task progress"""
        return multiplier / duration if duration > 0 else 0


class TaskModel:
    """Main model coordinating repository and simulator"""

    def __init__(self):
        self.repository = TaskRepository()
        self.simulator = PDESimulator()

        # Initialize simulation data structure for compatibility
        self.simulation_data = {
            "tasks": None,
            "adjacency": None,
            "u_matrix": None,
            "num_tasks": None,
            "start_times_risk": None,
            "finish_times_risk": None,
            "durations_risk": None,
            "simulation_time": None,
            "risk_curve": None,
            "classical_risk": None,
            "start_times_classical": None,
            "finish_times_classical": None
        }

        self.simulation_results: Optional[SimulationResults] = None
        self.dt = 0.01
        self.T = 150

    @property
    def task_df(self) -> pd.DataFrame:
        """Convert tasks to DataFrame for UI compatibility"""
        tasks = self.repository.get_all_tasks()
        data = []
        for task in tasks:
            data.append({
                "Select": False,
                "ID": task.id,
                "Task": task.name,
                "Duration (days)": task.duration,
                "Dependencies (IDs)": ",".join(map(str, task.dependencies)) if task.dependencies else "",
                "Risk (0-5)": task.risk_level,
                "Parent ID": task.parent_id or ""
            })
        return pd.DataFrame(data)

    @task_df.setter
    def task_df(self, value: pd.DataFrame):
        """Update tasks from DataFrame"""
        self.repository.update_from_dataframe(value)

    def save_tasks(self):
        """Save tasks - compatibility wrapper"""
        return self.repository._save_to_file()

    def load_tasks(self) -> pd.DataFrame:
        """Load tasks - compatibility wrapper"""
        return self.task_df

    def validate_tasks(self) -> Tuple[bool, List[str]]:
        """Validate all tasks"""
        return self.repository.validate_dependencies()

    def build_adjacency(self) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Build adjacency matrix - compatibility wrapper"""
        tasks = self.repository.get_all_tasks()
        if not tasks:
            return None, "No tasks available"

        try:
            adjacency = self.simulator._build_adjacency_matrix(tasks)
            return adjacency, None
        except Exception as e:
            return None, str(e)

    def run_pde_simulation(self, diffusion=0.02, reaction_multiplier=2.0, max_delay=0.05) -> Tuple[bool, Optional[str]]:
        """Run PDE simulation - compatibility wrapper"""
        tasks = self.repository.get_all_tasks()
        if not tasks:
            return False, "No tasks available"

        try:
            results = self.simulator.simulate(tasks, self.T, diffusion, reaction_multiplier, max_delay)

            # Update simulation_data for compatibility
            self.simulation_data.update({
                "tasks": [task.name for task in results.tasks],
                "adjacency": results.adjacency_matrix,
                "u_matrix": results.u_matrix,
                "num_tasks": len(results.tasks),
                "start_times_risk": results.start_times,
                "finish_times_risk": results.finish_times,
                "durations_risk": np.array([task.duration * max(1.0, task.risk_level) for task in results.tasks]),
                "simulation_time": results.simulation_time,
                "risk_curve": results.risk_curve
            })

            self.simulation_results = results
            return True, None
        except Exception as e:
            return False, str(e)

    def run_pde_simulation_maxed(self) -> Tuple[bool, Optional[str]]:
        """Run maxed PDE simulation - compatibility wrapper"""
        tasks = self.repository.get_all_tasks()
        if not tasks:
            return False, "No tasks available"

        try:
            results = self.simulator.simulate_maxed(tasks, self.T)

            # Update simulation_data for compatibility
            self.simulation_data.update({
                "tasks": [task.name for task in results.tasks],
                "adjacency": results.adjacency_matrix,
                "u_matrix": results.u_matrix,
                "num_tasks": len(results.tasks),
                "start_times_risk": results.start_times,
                "finish_times_risk": results.finish_times,
                "durations_risk": np.array([task.duration * max(1.0, task.risk_level) for task in results.tasks]),
                "simulation_time": results.simulation_time,
                "risk_curve": results.risk_curve
            })

            self.simulation_results = results
            return True, None
        except Exception as e:
            return False, str(e)

    def run_classical_simulation(self) -> Tuple[bool, Optional[str]]:
        """Run classical simulation - compatibility wrapper"""
        start_times = self.simulation_data.get("start_times_classical")
        if start_times is None:
            return False, "Run Classical schedule first"

        try:
            tasks = self.repository.get_all_tasks()
            durations = np.array([task.duration for task in tasks])
            risks = np.array([task.risk_level for task in tasks])
            durations_risk = durations * np.maximum(1.0, risks)

            # Use same simulation_time as PDE if available
            simulation_time = self.simulation_data.get("simulation_time")
            if simulation_time is None:
                max_time = np.max(start_times + durations_risk) * 1.1
                steps = max(1, int(max_time / self.dt))
                simulation_time = np.linspace(0, max_time, steps + 1)

            u = np.zeros((len(durations), len(simulation_time)))
            for i in range(len(durations)):
                for j, t in enumerate(simulation_time):
                    if t < start_times[i]:
                        u[i, j] = 0
                    elif t < start_times[i] + durations_risk[i]:
                        u[i, j] = (t - start_times[i]) / durations_risk[i]
                    else:
                        u[i, j] = 1

            classical_risk = u.mean(axis=0)
            self.simulation_data["classical_risk"] = classical_risk
            return True, None
        except Exception as e:
            return False, str(e)

    def import_mpp(self, file) -> Tuple[bool, Optional[str]]:
        """Import MPP file - compatibility wrapper"""
        try:
            import mppx
            project = mppx.read(file)

            # Clear existing tasks
            self.repository.clear_all_tasks()

            # Add tasks from MPP
            for task in project.tasks:
                name = task.name or f"Task {task.id}"
                duration = task.duration or 7.0
                deps = [dep.id for dep in task.predecessors] if task.predecessors else []
                self.repository.add_task(name, duration, deps, 0.0)

            return True, None
        except ImportError:
            return False, "MPP import requires 'python-mppx' library. Please install it or use manual entry."
        except Exception as e:
            return False, f"Error importing .mpp file: {str(e)}"

    def export_mpp(self) -> Tuple[Optional[str], Optional[str]]:
        """Export to MPP format - compatibility wrapper"""
        if self.simulation_data["start_times_risk"] is None:
            return None, "Run simulation first to export MPP data"

        try:
            tasks = self.repository.get_all_tasks()
            mpp_data = {
                "Tasks": [task.name for task in tasks],
                "Duration": [task.duration for task in tasks],
                "Start": self.simulation_data["start_times_risk"].tolist(),
                "Finish": self.simulation_data["finish_times_risk"].tolist(),
                "Predecessors": [",".join(map(str, task.dependencies)) for task in tasks],
                "Risk": [task.risk_level for task in tasks]
            }
            mpp_df = pd.DataFrame(mpp_data)
            return mpp_df.to_csv(index=False), None
        except Exception as e:
            return None, str(e)

    def compute_eigenvalues(self) -> Tuple[Optional[np.ndarray], Optional[float], Optional[str]]:
        """Compute eigenvalues - compatibility wrapper"""
        adjacency = self.simulation_data.get("adjacency")
        if adjacency is None:
            return None, None, "No adjacency matrix available"

        try:
            eigenvalues, _ = np.linalg.eig(adjacency.astype(float))
            degree = np.sum(adjacency, axis=1)
            D = np.diag(degree)
            laplacian = D - adjacency
            eigvals_lap, _ = np.linalg.eig(laplacian.astype(float))
            second_eigenvalue = np.sort(eigvals_lap)[1] if len(eigvals_lap) > 1 else 0
            return eigenvalues, second_eigenvalue, None
        except Exception as e:
            return None, None, str(e)

    def save_project(self, u_matrix=None, risk_curve=None, classical_risk=None, filename="project.json") -> Tuple[
        bool, Optional[str]]:
        """Save the entire project to JSON"""
        try:
            project_data = {
                "tasks": self.task_df.to_dict(orient="list"),
                "simulation_data": {
                    "u_matrix": u_matrix.tolist() if u_matrix is not None else (
                        self.simulation_data["u_matrix"].tolist() if self.simulation_data.get(
                            "u_matrix") is not None else None
                    ),
                    "risk_curve": risk_curve.tolist() if risk_curve is not None else (
                        self.simulation_data["risk_curve"].tolist() if self.simulation_data.get(
                            "risk_curve") is not None else None
                    ),
                    "classical_risk": classical_risk.tolist() if classical_risk is not None else (
                        self.simulation_data["classical_risk"].tolist() if self.simulation_data.get(
                            "classical_risk") is not None else None
                    )
                }
            }
            with open(filename, "w") as f:
                json.dump(project_data, f, indent=2)
            return True, None
        except Exception as e:
            return False, str(e)

    def run_monte_carlo_simulation(self, num_simulations=1000, confidence_levels=[80, 90, 95]) -> Tuple[
        bool, Optional[str]]:
        """Monte Carlo simulation using triangular distributions"""
        try:
            valid, errors = self.validate_tasks()
            if not valid:
                return False, "; ".join(errors)

            task_df = self.task_df
            num_tasks = len(task_df)

            # Create triangular distributions based on risk levels
            base_durations = task_df["Duration (days)"].values
            risks = task_df["Risk (0-5)"].values

            # Risk-adjusted uncertainty ranges
            # Risk 0: ±10%, Risk 1-2: ±15-25%, Risk 3-4: ±30-50%, Risk 5: ±60%
            uncertainty_factors = np.where(risks == 0, 0.1,
                                           np.where(risks <= 2, 0.15 + risks * 0.05,
                                                    np.where(risks <= 4, 0.25 + (risks - 2) * 0.125,
                                                             0.6)))  # Risk 5

            # Calculate optimistic and pessimistic estimates
            optimistic = base_durations * (1 - uncertainty_factors)
            pessimistic = base_durations * (1 + uncertainty_factors * 1.5)  # Asymmetric toward delays

            completion_times = []
            task_criticality_count = np.zeros(num_tasks)
            all_start_times = []
            all_finish_times = []

            # Run simulations
            for sim in range(num_simulations):
                # Sample durations from triangular distribution
                sampled_durations = np.random.triangular(optimistic, base_durations, pessimistic)

                # Calculate critical path for this sample
                start_times, finish_times, critical_path = self._calculate_monte_carlo_schedule(
                    task_df, sampled_durations
                )

                # Record results
                completion_times.append(np.max(finish_times))
                task_criticality_count[critical_path] += 1
                all_start_times.append(start_times.copy())
                all_finish_times.append(finish_times.copy())

            # Calculate statistics
            completion_times = np.array(completion_times)
            all_start_times = np.array(all_start_times)
            all_finish_times = np.array(all_finish_times)

            # Calculate percentiles
            percentiles = {}
            for level in confidence_levels:
                lower = (100 - level) / 2
                upper = 100 - lower
                percentiles[f"P{int(lower)}"] = np.percentile(completion_times, lower)
                percentiles[f"P{int(upper)}"] = np.percentile(completion_times, upper)

            # Add median and mean
            percentiles["P50 (Median)"] = np.percentile(completion_times, 50)
            percentiles["Mean"] = np.mean(completion_times)

            # Calculate task-level statistics
            task_start_percentiles = {}
            task_finish_percentiles = {}
            for level in confidence_levels:
                lower = (100 - level) / 2
                upper = 100 - lower
                task_start_percentiles[f"P{int(lower)}"] = np.percentile(all_start_times, lower, axis=0)
                task_start_percentiles[f"P{int(upper)}"] = np.percentile(all_start_times, upper, axis=0)
                task_finish_percentiles[f"P{int(lower)}"] = np.percentile(all_finish_times, lower, axis=0)
                task_finish_percentiles[f"P{int(upper)}"] = np.percentile(all_finish_times, upper, axis=0)

            # Store results
            self.simulation_data["monte_carlo_results"] = {
                "completion_times": completion_times,
                "percentiles": percentiles,
                "task_criticality": task_criticality_count / num_simulations * 100,
                "num_simulations": num_simulations,
                "mean_completion": np.mean(completion_times),
                "std_completion": np.std(completion_times),
                "task_start_percentiles": task_start_percentiles,
                "task_finish_percentiles": task_finish_percentiles,
                "mean_start_times": np.mean(all_start_times, axis=0),
                "mean_finish_times": np.mean(all_finish_times, axis=0),
                "confidence_levels": confidence_levels
            }

            return True, None

        except Exception as e:
            return False, str(e)

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