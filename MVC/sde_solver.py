# sde_solver.py - Stochastic Differential Equation Solver for Project Risk Analysis
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings


@dataclass
class SDEParameters:
    """Parameters for SDE simulation"""
    dt: float = 0.01  # Time step
    T: float = 200.0  # Total simulation time
    n_paths: int = 1000  # Number of Monte Carlo paths
    volatility: float = 0.2  # Base volatility coefficient
    correlation_strength: float = 0.5  # Cross-task correlation strength
    risk_amplification: float = 1.5  # Risk amplification factor
    drift_coefficient: float = 0.1  # Mean reversion/drift term
    jump_intensity: float = 0.1  # Poisson jump intensity
    jump_magnitude: float = 0.15  # Jump size standard deviation


@dataclass
class SDEResults:
    """Container for SDE simulation results"""
    time_grid: np.ndarray
    task_paths: np.ndarray  # Shape: (n_tasks, n_paths, n_time_steps)
    completion_times: np.ndarray  # Shape: (n_tasks, n_paths)
    project_completion_times: np.ndarray  # Shape: (n_paths,)
    risk_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]


class NoiseGenerator(ABC):
    """Abstract base class for noise generation"""

    @abstractmethod
    def generate(self, shape: Tuple[int, ...], dt: float) -> np.ndarray:
        """Generate noise for given shape and time step"""
        pass


class BrownianMotion(NoiseGenerator):
    """Standard Brownian motion noise generator"""

    def __init__(self, correlation_matrix: Optional[np.ndarray] = None):
        self.correlation_matrix = correlation_matrix

    def generate(self, shape: Tuple[int, ...], dt: float) -> np.ndarray:
        """Generate correlated Brownian increments"""
        if len(shape) == 2:  # (n_tasks, n_time_steps)
            n_tasks, n_steps = shape

            if self.correlation_matrix is not None:
                # Generate correlated noise
                independent_noise = np.random.randn(n_tasks, n_steps)
                L = np.linalg.cholesky(self.correlation_matrix)
                correlated_noise = L @ independent_noise
                return correlated_noise * np.sqrt(dt)
            else:
                return np.random.randn(*shape) * np.sqrt(dt)
        else:
            return np.random.randn(*shape) * np.sqrt(dt)


class JumpDiffusion(NoiseGenerator):
    """Jump diffusion noise generator (Merton model)"""

    def __init__(self, jump_intensity: float = 0.1, jump_magnitude: float = 0.15):
        self.jump_intensity = jump_intensity
        self.jump_magnitude = jump_magnitude

    def generate(self, shape: Tuple[int, ...], dt: float) -> np.ndarray:
        """Generate jump diffusion increments"""
        # Brownian component
        brownian = np.random.randn(*shape) * np.sqrt(dt)

        # Jump component
        jump_times = np.random.poisson(self.jump_intensity * dt, shape)
        jumps = np.where(jump_times > 0,
                         np.random.normal(0, self.jump_magnitude, shape),
                         0)

        return brownian + jumps


class SDESolver:
    """
    Stochastic Differential Equation solver for project management with risk propagation.

    Models task completion as SDEs with:
    - Risk-dependent volatility
    - Cross-task correlations
    - Jump processes for major disruptions
    - Dependency-based drift terms
    """

    def __init__(self, parameters: SDEParameters = None):
        self.params = parameters or SDEParameters()
        self.noise_generator = None
        self._setup_default_noise()

    def _setup_default_noise(self):
        """Setup default Brownian motion noise generator"""
        self.noise_generator = BrownianMotion()

    def set_noise_generator(self, generator: NoiseGenerator):
        """Set custom noise generator"""
        self.noise_generator = generator

    def _build_correlation_matrix(self, adjacency: np.ndarray,
                                  correlation_strength: float) -> np.ndarray:
        """
        Build correlation matrix based on task dependencies.

        Args:
            adjacency: Task dependency adjacency matrix
            correlation_strength: Strength of correlation between dependent tasks

        Returns:
            Correlation matrix for noise generation
        """
        n_tasks = adjacency.shape[0]
        correlation_matrix = np.eye(n_tasks)

        # Add correlations based on dependencies
        for i in range(n_tasks):
            for j in range(n_tasks):
                if adjacency[i, j] > 0:  # j depends on i
                    correlation_matrix[i, j] = correlation_strength
                    correlation_matrix[j, i] = correlation_strength

        # Ensure positive definiteness
        eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
        eigenvals = np.maximum(eigenvals, 0.01)  # Floor eigenvalues
        correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        return correlation_matrix

    def _compute_dynamic_start_times(self, progress: np.ndarray, adjacency: np.ndarray,
                                     completion_times: np.ndarray, t: float) -> np.ndarray:
        """
        Compute when tasks can actually start based on predecessor completion.

        Args:
            progress: Current progress state [0,1] for each task
            adjacency: Task dependency matrix
            completion_times: When tasks actually completed (inf if not completed)
            t: Current time

        Returns:
            Array indicating whether each task can start (True/False)
        """
        n_tasks = len(progress)
        can_start = np.zeros(n_tasks, dtype=bool)

        for i in range(n_tasks):
            predecessors = np.where(adjacency[:, i] > 0)[0]

            if len(predecessors) == 0:
                # No dependencies, can start immediately
                can_start[i] = True
            else:
                # Task can start if all predecessors are complete (progress >= 1.0)
                can_start[i] = np.all(progress[predecessors] >= 1.0)

        return can_start

    def _compute_drift(self, progress: np.ndarray, adjacency: np.ndarray,
                       durations: np.ndarray, risks: np.ndarray,
                       t: float, can_start: np.ndarray) -> np.ndarray:
        """
        Compute drift term for SDE based on:
        - Natural task progression
        - Dependency constraints (FIXED)
        - Risk-based delays (FIXED)

        Args:
            progress: Current progress state [0,1] for each task
            adjacency: Task dependency matrix
            durations: Task durations
            risks: Risk levels [0,5] for each task
            t: Current time
            can_start: Boolean array indicating which tasks can start

        Returns:
            Drift vector for each task
        """
        n_tasks = len(progress)
        drift = np.zeros(n_tasks)

        for i in range(n_tasks):
            # Task cannot progress if dependencies not met
            if not can_start[i]:
                drift[i] = 0.0
                continue

            # Task cannot progress if already complete
            if progress[i] >= 1.0:
                # Add mean reversion to prevent over-completion
                drift[i] = -self.params.drift_coefficient * (progress[i] - 1.0)
                continue

            # Base progression rate (inverse of risk-adjusted duration)
            # FIXED: Risk should slow down tasks, not speed them up
            risk_factor = 1.0 + risks[i] * 0.3  # Higher risk = slower progress
            effective_duration = durations[i] * risk_factor
            base_rate = 1.0 / effective_duration if effective_duration > 0 else 0

            # Apply drift
            drift[i] = base_rate

        return drift

    def _compute_volatility(self, progress: np.ndarray, risks: np.ndarray,
                            adjacency: np.ndarray, can_start: np.ndarray) -> np.ndarray:
        """
        Compute volatility (diffusion coefficient) based on:
        - Task risk levels
        - Current progress state
        - Dependency propagation

        Args:
            progress: Current progress [0,1] for each task
            risks: Risk levels [0,5] for each task
            adjacency: Task dependency matrix
            can_start: Boolean array indicating which tasks can start

        Returns:
            Volatility vector for each task
        """
        n_tasks = len(progress)
        volatility = np.zeros(n_tasks)

        for i in range(n_tasks):
            # No volatility if task cannot start or is complete
            if not can_start[i] or progress[i] >= 1.0:
                volatility[i] = 0.0
                continue

            # Base volatility scaled by risk
            base_vol = self.params.volatility * (1.0 + risks[i] * 0.3)

            # Reduce volatility as task approaches completion
            completion_factor = max(0.1, 1.0 - progress[i])

            # Amplify volatility from risky predecessors that recently completed
            risk_propagation = 0.0
            predecessors = np.where(adjacency[:, i] > 0)[0]
            if len(predecessors) > 0:
                pred_risks = risks[predecessors]
                # Volatility increases if predecessors were risky
                risk_propagation = np.mean(pred_risks) * 0.05

            volatility[i] = base_vol * completion_factor + risk_propagation

        return volatility

    def _calculate_critical_path_duration(self, durations: np.ndarray, risks: np.ndarray,
                                          adjacency: np.ndarray) -> float:
        """Calculate critical path duration with risk adjustments"""
        n_tasks = len(durations)
        risk_adjusted_durations = durations * (1.0 + risks * 0.3)

        # Forward pass to calculate earliest finish times
        earliest_start = np.zeros(n_tasks)
        earliest_finish = np.zeros(n_tasks)

        # Topological sort to process tasks in dependency order
        in_degree = np.sum(adjacency, axis=0)
        queue = [i for i in range(n_tasks) if in_degree[i] == 0]
        processed = []

        while queue:
            current = queue.pop(0)
            processed.append(current)

            # Calculate earliest start time
            predecessors = np.where(adjacency[:, current] > 0)[0]
            if len(predecessors) > 0:
                earliest_start[current] = np.max(earliest_finish[predecessors])

            earliest_finish[current] = earliest_start[current] + risk_adjusted_durations[current]

            # Update successors
            for i in range(n_tasks):
                if adjacency[current, i] > 0:
                    in_degree[i] -= 1
                    if in_degree[i] == 0:
                        queue.append(i)

        return np.max(earliest_finish)

    def solve(self, adjacency: np.ndarray, durations: np.ndarray,
              risks: np.ndarray, start_times: np.ndarray = None) -> SDEResults:
        """
        Solve the SDE system for project completion with risk propagation.

        Args:
            adjacency: Task dependency adjacency matrix (n_tasks x n_tasks)
            durations: Task durations in days (n_tasks,)
            risks: Risk levels [0,5] for each task (n_tasks,)
            start_times: Optional earliest start times for tasks (n_tasks,)

        Returns:
            SDEResults containing simulation results and risk metrics
        """
        n_tasks = len(durations)
        if start_times is None:
            start_times = np.zeros(n_tasks)

        # Setup correlation matrix and noise generator
        correlation_matrix = self._build_correlation_matrix(
            adjacency, self.params.correlation_strength)
        self.set_noise_generator(BrownianMotion(correlation_matrix))

        # Time grid
        n_steps = int(self.params.T / self.params.dt) + 1
        time_grid = np.linspace(0, self.params.T, n_steps)

        # Storage for all paths
        task_paths = np.zeros((n_tasks, self.params.n_paths, n_steps))
        completion_times = np.full((n_tasks, self.params.n_paths), np.inf)

        # Monte Carlo simulation
        for path in range(self.params.n_paths):
            # Initialize progress for this path
            progress = np.zeros(n_tasks)
            path_completion_times = np.full(n_tasks, np.inf)

            for step in range(1, n_steps):
                t = time_grid[step]
                dt = self.params.dt

                # Determine which tasks can start based on dependencies
                can_start = self._compute_dynamic_start_times(
                    progress, adjacency, path_completion_times, t)

                # Also respect earliest start times
                can_start = can_start & (t >= start_times)

                # Compute drift and volatility for eligible tasks
                drift = self._compute_drift(progress, adjacency, durations, risks, t, can_start)
                volatility = self._compute_volatility(progress, risks, adjacency, can_start)

                # Generate noise
                noise = self.noise_generator.generate((n_tasks, 1), dt).flatten()

                # SDE step: dX = drift*dt + volatility*dW
                for i in range(n_tasks):
                    if can_start[i] and progress[i] < 1.0:
                        dp = drift[i] * dt + volatility[i] * noise[i]
                        progress[i] = max(0, progress[i] + dp)

                        # Record completion time when task reaches 100%
                        if progress[i] >= 1.0 and path_completion_times[i] == np.inf:
                            path_completion_times[i] = t
                            progress[i] = 1.0  # Cap at 1.0

                # Store progress
                task_paths[:, path, step] = progress.copy()

            # Store completion times for this path
            completion_times[:, path] = path_completion_times

        # Set infinite completion times to max time for analysis
        completion_times = np.where(completion_times == np.inf,
                                    self.params.T, completion_times)

        # Compute project completion times (max of all tasks per path)
        project_completion_times = np.max(completion_times, axis=0)

        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            completion_times, project_completion_times, durations, risks, adjacency)

        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            completion_times, project_completion_times)

        return SDEResults(
            time_grid=time_grid,
            task_paths=task_paths,
            completion_times=completion_times,
            project_completion_times=project_completion_times,
            risk_metrics=risk_metrics,
            confidence_intervals=confidence_intervals
        )

    def _calculate_risk_metrics(self, completion_times: np.ndarray,
                                project_completion_times: np.ndarray,
                                durations: np.ndarray, risks: np.ndarray,
                                adjacency: np.ndarray) -> Dict[str, float]:
        """Calculate various risk metrics from simulation results"""
        metrics = {}

        # Project-level metrics
        metrics['mean_project_duration'] = np.mean(project_completion_times)
        metrics['std_project_duration'] = np.std(project_completion_times)

        # Avoid division by zero
        if metrics['mean_project_duration'] > 0:
            metrics['cv_project_duration'] = metrics['std_project_duration'] / metrics['mean_project_duration']
        else:
            metrics['cv_project_duration'] = 0.0

        # Calculate proper critical path baseline
        deterministic_duration = self._calculate_critical_path_duration(durations, risks, adjacency)

        # Calculate schedule risk factor
        if deterministic_duration > 0:
            metrics['schedule_risk_factor'] = metrics['mean_project_duration'] / deterministic_duration
        else:
            metrics['schedule_risk_factor'] = 1.0

        # Probability metrics (using critical path baseline as target)
        target_duration = deterministic_duration
        metrics['prob_on_time'] = np.mean(project_completion_times <= target_duration)

        # Value at Risk (95% confidence)
        metrics['var_95'] = np.percentile(project_completion_times, 95)
        metrics['expected_shortfall'] = np.mean(
            project_completion_times[project_completion_times >= metrics['var_95']])

        # Task-level aggregates
        mean_task_completion = np.mean(completion_times, axis=1)
        risk_adjusted_durations = durations * (1.0 + risks * 0.3)

        task_delays = mean_task_completion - risk_adjusted_durations
        metrics['max_expected_task_delay'] = np.max(task_delays)
        metrics['total_expected_delay'] = np.sum(np.maximum(0, task_delays))

        return metrics

    def _calculate_confidence_intervals(self, completion_times: np.ndarray,
                                        project_completion_times: np.ndarray,
                                        confidence_levels: List[float] = [0.05, 0.95]) -> Dict[
        str, Tuple[float, float]]:
        """Calculate confidence intervals for completion times"""
        intervals = {}

        # Project completion time intervals
        lower = np.percentile(project_completion_times, confidence_levels[0] * 100)
        upper = np.percentile(project_completion_times, confidence_levels[1] * 100)
        intervals['project_completion'] = (lower, upper)

        # Task completion time intervals
        n_tasks = completion_times.shape[0]
        for i in range(n_tasks):
            task_times = completion_times[i, :]
            lower = np.percentile(task_times, confidence_levels[0] * 100)
            upper = np.percentile(task_times, confidence_levels[1] * 100)
            intervals[f'task_{i + 1}_completion'] = (lower, upper)

        return intervals

    def analyze_sensitivity(self, adjacency: np.ndarray, durations: np.ndarray,
                            risks: np.ndarray, parameter_name: str,
                            parameter_range: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform sensitivity analysis on SDE parameters.

        Args:
            adjacency: Task dependency matrix
            durations: Task durations
            risks: Risk levels
            parameter_name: Name of parameter to vary
            parameter_range: Array of parameter values to test

        Returns:
            Dictionary with parameter values and corresponding metrics
        """
        results = {
            'parameter_values': parameter_range,
            'mean_durations': [],
            'std_durations': [],
            'prob_on_time': [],
            'var_95': []
        }

        original_value = getattr(self.params, parameter_name)

        for param_value in parameter_range:
            # Set parameter value
            setattr(self.params, parameter_name, param_value)

            # Run simulation
            sde_results = self.solve(adjacency, durations, risks)

            # Store metrics
            results['mean_durations'].append(sde_results.risk_metrics['mean_project_duration'])
            results['std_durations'].append(sde_results.risk_metrics['std_project_duration'])
            results['prob_on_time'].append(sde_results.risk_metrics['prob_on_time'])
            results['var_95'].append(sde_results.risk_metrics['var_95'])

        # Restore original parameter value
        setattr(self.params, parameter_name, original_value)

        # Convert lists to arrays
        for key in ['mean_durations', 'std_durations', 'prob_on_time', 'var_95']:
            results[key] = np.array(results[key])

        return results

    def calibrate_to_historical_data(self, historical_durations: np.ndarray,
                                     adjacency: np.ndarray, planned_durations: np.ndarray,
                                     risks: np.ndarray) -> Dict[str, float]:
        """
        Calibrate SDE parameters to match historical project data.

        Args:
            historical_durations: Observed project completion times
            adjacency: Task dependency matrix
            planned_durations: Originally planned task durations
            risks: Task risk levels

        Returns:
            Dictionary of calibrated parameters
        """

        def objective(params_vector):
            # Unpack parameters
            volatility, correlation_strength, risk_amplification = params_vector

            # Update SDE parameters
            self.params.volatility = volatility
            self.params.correlation_strength = correlation_strength
            self.params.risk_amplification = risk_amplification

            # Run simulation
            try:
                results = self.solve(adjacency, planned_durations, risks)
                simulated_durations = results.project_completion_times

                # Calculate fit metrics
                historical_mean = np.mean(historical_durations)
                historical_std = np.std(historical_durations)
                simulated_mean = np.mean(simulated_durations)
                simulated_std = np.std(simulated_durations)

                # Minimize difference in moments
                mean_error = (simulated_mean - historical_mean) ** 2
                std_error = (simulated_std - historical_std) ** 2

                return mean_error + std_error

            except Exception:
                return 1e6  # Large penalty for invalid parameters

        # Initial guess and bounds
        initial_params = [self.params.volatility,
                          self.params.correlation_strength,
                          self.params.risk_amplification]
        bounds = [(0.01, 1.0), (0.0, 1.0), (0.5, 3.0)]

        # Optimize
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')

        if result.success:
            # Update parameters with calibrated values
            self.params.volatility = result.x[0]
            self.params.correlation_strength = result.x[1]
            self.params.risk_amplification = result.x[2]

            return {
                'volatility': result.x[0],
                'correlation_strength': result.x[1],
                'risk_amplification': result.x[2],
                'calibration_error': result.fun
            }
        else:
            warnings.warn("Calibration failed to converge")
            return {}


# Integration with existing model structure
class SDEModelIntegration:
    """Integration layer for SDE solver with existing project model"""

    def __init__(self, model):
        self.model = model
        self.sde_solver = SDESolver()
        self.sde_results = None

    def run_sde_simulation(self, sde_params: SDEParameters = None) -> Tuple[bool, Optional[str]]:
        """
        Run SDE simulation using current model data.

        Args:
            sde_params: Optional SDE parameters, uses defaults if None

        Returns:
            (success, error_message) tuple
        """
        try:
            # Validate model data
            valid, errors = self.model.validate_tasks()
            if not valid:
                return False, "; ".join(errors)

            # Extract data from model
            adjacency, error = self.model.build_adjacency()
            if error:
                return False, error

            durations = self.model.task_df["Duration (days)"].to_numpy(dtype=float)
            risks = self.model.task_df["Risk (0-5)"].to_numpy(dtype=float)

            # Setup SDE solver
            if sde_params:
                self.sde_solver.params = sde_params

            # Run simulation
            self.sde_results = self.sde_solver.solve(adjacency, durations, risks)

            # Store results in model for compatibility
            self._update_model_simulation_data()

            return True, None

        except Exception as e:
            return False, str(e)

    def _update_model_simulation_data(self):
        """Update model's simulation_data with SDE results for UI compatibility"""
        if self.sde_results is None:
            return

        # Store SDE results in their own namespace - DON'T overwrite PDE data
        self.model.simulation_data["sde_results"] = self.sde_results

        # Use mean path for compatibility with existing UI
        mean_progress = np.mean(self.sde_results.task_paths, axis=1)
        n_tasks, n_steps = mean_progress.shape

        # Calculate proper start times based on dependencies
        adjacency, _ = self.model.build_adjacency()
        start_times = np.zeros(n_tasks)
        mean_completion_times = np.mean(self.sde_results.completion_times, axis=1)  # Fixed variable name

        # Forward pass to calculate start times
        for i in range(n_tasks):
            predecessors = np.where(adjacency[:, i] > 0)[0]
            if len(predecessors) > 0:
                start_times[i] = np.max(mean_completion_times[predecessors])

        # Store SDE results in SEPARATE keys to preserve original PDE data
        sde_specific_data = {
            "sde_u_matrix": mean_progress,
            "sde_start_times": start_times,
            "sde_finish_times": mean_completion_times,
            "sde_durations": mean_completion_times - start_times,
            "sde_simulation_time": self.sde_results.time_grid,
            "sde_risk_curve": np.mean(mean_progress, axis=0),
            "sde_results": self.sde_results  # Store full SDE results
        }

        # Add SDE data without overwriting PDE/Classical data
        self.model.simulation_data.update(sde_specific_data)

        # Only update core fields if they don't exist (preserve PDE data)
        if self.model.simulation_data.get("tasks") is None:
            self.model.simulation_data["tasks"] = self.model.task_df["Task"].tolist()
        if self.model.simulation_data.get("adjacency") is None:
            self.model.simulation_data["adjacency"] = adjacency
        if self.model.simulation_data.get("num_tasks") is None:
            self.model.simulation_data["num_tasks"] = n_tasks

        print("✅ SDE results stored without overwriting PDE/Classical data")


    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary from SDE results"""
        if self.sde_results is None:
            return {}

        summary = {
            "sde_metrics": self.sde_results.risk_metrics,
            "confidence_intervals": self.sde_results.confidence_intervals,
            "simulation_params": {
                "n_paths": self.sde_solver.params.n_paths,
                "volatility": self.sde_solver.params.volatility,
                "correlation_strength": self.sde_solver.params.correlation_strength
            }
        }

        return summary


# Example usage and testing
if __name__ == "__main__":
    # Test the SDE solver with sample data
    print("Testing SDE Solver for Project Management...")

    # Sample project data
    n_tasks = 5
    adjacency = np.array([
        [0, 1, 0, 0, 0],  # Task 1 -> Task 2
        [0, 0, 1, 1, 0],  # Task 2 -> Tasks 3,4
        [0, 0, 0, 0, 1],  # Task 3 -> Task 5
        [0, 0, 0, 0, 1],  # Task 4 -> Task 5
        [0, 0, 0, 0, 0]  # Task 5 (final)
    ])

    durations = np.array([10, 15, 12, 8, 6])  # days
    risks = np.array([1, 3, 2, 1, 2])  # risk levels 0-5

    # Setup SDE parameters
    sde_params = SDEParameters(
        dt=0.1,
        T=100.0,
        n_paths=500,
        volatility=0.15,
        correlation_strength=0.3
    )

    # Create and run solver
    solver = SDESolver(sde_params)
    results = solver.solve(adjacency, durations, risks)

    # Print results
    print(f"\nSDE Simulation Results:")
    print(f"Mean project duration: {results.risk_metrics['mean_project_duration']:.2f} days")
    print(f"Standard deviation: {results.risk_metrics['std_project_duration']:.2f} days")
    print(f"Probability of on-time completion: {results.risk_metrics['prob_on_time']:.3f}")
    print(f"95% Value at Risk: {results.risk_metrics['var_95']:.2f} days")

    # Show confidence intervals
    ci = results.confidence_intervals['project_completion']
    print(f"90% Confidence interval: [{ci[0]:.2f}, {ci[1]:.2f}] days")

    print("\nSDE Solver test completed successfully!")