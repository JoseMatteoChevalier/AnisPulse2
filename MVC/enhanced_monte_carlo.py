# enhanced_monte_carlo.py - Professional Monte Carlo with 3-Point Estimation
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DistributionType(Enum):
    """Supported probability distributions for task durations"""
    TRIANGULAR = "triangular"
    BETA_PERT = "beta_pert"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"


@dataclass
class TaskEstimate:
    """3-Point estimation data for a task"""
    optimistic: float
    most_likely: float
    pessimistic: float
    distribution_type: DistributionType = DistributionType.TRIANGULAR

    def __post_init__(self):
        """Validate 3-point estimate"""
        if not (self.optimistic <= self.most_likely <= self.pessimistic):
            raise ValueError(
                f"Invalid 3-point estimate: O({self.optimistic}) <= M({self.most_likely}) <= P({self.pessimistic})")

    @property
    def expected_duration(self) -> float:
        """Calculate expected duration using PERT formula"""
        return (self.optimistic + 4 * self.most_likely + self.pessimistic) / 6

    @property
    def variance(self) -> float:
        """Calculate variance using PERT formula"""
        return ((self.pessimistic - self.optimistic) / 6) ** 2

    @property
    def standard_deviation(self) -> float:
        """Calculate standard deviation"""
        return np.sqrt(self.variance)


class EnhancedMonteCarlo:
    """Professional Monte Carlo simulation with industry-standard features"""

    def __init__(self):
        self.task_estimates: Dict[int, TaskEstimate] = {}
        self.correlation_matrix: Optional[np.ndarray] = None

    def add_task_estimate(self, task_id: int, optimistic: float,
                          most_likely: float, pessimistic: float,
                          distribution_type: DistributionType = DistributionType.TRIANGULAR):
        """Add 3-point estimate for a task"""
        self.task_estimates[task_id] = TaskEstimate(
            optimistic=optimistic,
            most_likely=most_likely,
            pessimistic=pessimistic,
            distribution_type=distribution_type
        )

    def auto_generate_estimates_from_risk(self, task_df: pd.DataFrame) -> None:
        """Auto-generate 3-point estimates from risk levels"""
        for i, row in task_df.iterrows():
            base_duration = row["Duration (days)"]
            risk_level = row["Risk (0-5)"]

            # Risk-based uncertainty ranges
            if risk_level <= 1:
                # Low risk: ±10% optimistic, +20% pessimistic
                optimistic = base_duration * 0.9
                pessimistic = base_duration * 1.2
            elif risk_level <= 2:
                # Medium-low risk: ±15% optimistic, +30% pessimistic
                optimistic = base_duration * 0.85
                pessimistic = base_duration * 1.3
            elif risk_level <= 3:
                # Medium risk: ±20% optimistic, +40% pessimistic
                optimistic = base_duration * 0.8
                pessimistic = base_duration * 1.4
            elif risk_level <= 4:
                # High risk: ±25% optimistic, +60% pessimistic
                optimistic = base_duration * 0.75
                pessimistic = base_duration * 1.6
            else:
                # Very high risk: ±30% optimistic, +80% pessimistic
                optimistic = base_duration * 0.7
                pessimistic = base_duration * 1.8

            # Choose distribution based on risk level
            if risk_level >= 3:
                distribution_type = DistributionType.BETA_PERT  # More sophisticated for high risk
            else:
                distribution_type = DistributionType.TRIANGULAR  # Simple for low risk

            self.add_task_estimate(
                task_id=row["ID"],
                optimistic=optimistic,
                most_likely=base_duration,
                pessimistic=pessimistic,
                distribution_type=distribution_type
            )

    def sample_task_duration(self, task_id: int) -> float:
        """Sample duration for a single task"""
        if task_id not in self.task_estimates:
            raise ValueError(f"No estimate found for task {task_id}")

        estimate = self.task_estimates[task_id]

        if estimate.distribution_type == DistributionType.TRIANGULAR:
            return np.random.triangular(
                estimate.optimistic,
                estimate.most_likely,
                estimate.pessimistic
            )

        elif estimate.distribution_type == DistributionType.BETA_PERT:
            return self._sample_beta_pert(estimate)

        elif estimate.distribution_type == DistributionType.NORMAL:
            # Use PERT mean and std
            return np.random.normal(
                estimate.expected_duration,
                estimate.standard_deviation
            )

        elif estimate.distribution_type == DistributionType.LOGNORMAL:
            # Convert to lognormal parameters
            mu, sigma = self._normal_to_lognormal_params(
                estimate.expected_duration,
                estimate.standard_deviation
            )
            return np.random.lognormal(mu, sigma)

        elif estimate.distribution_type == DistributionType.UNIFORM:
            return np.random.uniform(
                estimate.optimistic,
                estimate.pessimistic
            )

        else:
            # Fallback to triangular
            return np.random.triangular(
                estimate.optimistic,
                estimate.most_likely,
                estimate.pessimistic
            )

    def _sample_beta_pert(self, estimate: TaskEstimate) -> float:
        """Sample from Beta-PERT distribution"""
        # Beta-PERT parameters
        a = estimate.optimistic
        b = estimate.pessimistic
        c = estimate.most_likely

        # Shape parameters for Beta distribution
        if b == a:  # Degenerate case
            return a

        # Lambda parameter (controls peakedness, typically 4)
        lambda_param = 4.0

        # Mean and variance
        mu = (a + lambda_param * c + b) / (lambda_param + 2)

        if b - a == 0:
            return a

        # Beta distribution parameters
        alpha = ((mu - a) * (2 * c - a - b)) / ((c - mu) * (b - a))
        beta = (alpha * (b - mu)) / (mu - a)

        # Ensure valid parameters
        alpha = max(alpha, 0.1)
        beta = max(beta, 0.1)

        # Sample from Beta(0,1) and scale to [a,b]
        beta_sample = stats.beta.rvs(alpha, beta)
        return a + beta_sample * (b - a)

    def _normal_to_lognormal_params(self, mean: float, std: float) -> Tuple[float, float]:
        """Convert normal distribution parameters to lognormal"""
        variance = std ** 2
        mu = np.log(mean ** 2 / np.sqrt(variance + mean ** 2))
        sigma = np.sqrt(np.log(variance / mean ** 2 + 1))
        return mu, sigma

    def sample_all_tasks(self, num_simulations: int) -> np.ndarray:
        """Sample durations for all tasks across all simulations"""
        task_ids = sorted(self.task_estimates.keys())
        num_tasks = len(task_ids)

        # Storage: [task_index, simulation]
        durations = np.zeros((num_tasks, num_simulations))

        for sim in range(num_simulations):
            for task_idx, task_id in enumerate(task_ids):
                durations[task_idx, sim] = self.sample_task_duration(task_id)

        return durations

    def calculate_schedule_statistics(self, all_start_times: np.ndarray,
                                      all_finish_times: np.ndarray,
                                      confidence_levels: List[int] = [80, 90, 95]) -> Dict:
        """Calculate comprehensive statistics from simulation results"""
        num_tasks, num_simulations = all_finish_times.shape

        # Project completion times (max finish time per simulation)
        project_completion_times = np.max(all_finish_times, axis=0)

        # Basic statistics
        stats_dict = {
            "num_simulations": num_simulations,
            "mean_project_duration": np.mean(project_completion_times),
            "median_project_duration": np.median(project_completion_times),
            "std_project_duration": np.std(project_completion_times),
            "min_project_duration": np.min(project_completion_times),
            "max_project_duration": np.max(project_completion_times),
        }

        # Confidence intervals
        for level in confidence_levels:
            alpha = (100 - level) / 2
            lower_percentile = alpha
            upper_percentile = 100 - alpha

            stats_dict[f"project_duration_p{int(lower_percentile)}"] = np.percentile(project_completion_times,
                                                                                     lower_percentile)
            stats_dict[f"project_duration_p{int(upper_percentile)}"] = np.percentile(project_completion_times,
                                                                                     upper_percentile)

        # Task-level statistics
        task_stats = {}
        for task_idx in range(num_tasks):
            task_starts = all_start_times[task_idx, :]
            task_finishes = all_finish_times[task_idx, :]

            task_stats[f"task_{task_idx + 1}"] = {
                "mean_start": np.mean(task_starts),
                "mean_finish": np.mean(task_finishes),
                "std_start": np.std(task_starts),
                "std_finish": np.std(task_finishes),
            }

            # Task confidence intervals
            for level in confidence_levels:
                alpha = (100 - level) / 2
                lower_percentile = alpha
                upper_percentile = 100 - alpha

                task_stats[f"task_{task_idx + 1}"][f"start_p{int(lower_percentile)}"] = np.percentile(task_starts,
                                                                                                      lower_percentile)
                task_stats[f"task_{task_idx + 1}"][f"start_p{int(upper_percentile)}"] = np.percentile(task_starts,
                                                                                                      upper_percentile)
                task_stats[f"task_{task_idx + 1}"][f"finish_p{int(lower_percentile)}"] = np.percentile(task_finishes,
                                                                                                       lower_percentile)
                task_stats[f"task_{task_idx + 1}"][f"finish_p{int(upper_percentile)}"] = np.percentile(task_finishes,
                                                                                                       upper_percentile)

        stats_dict["task_statistics"] = task_stats
        stats_dict["all_project_completion_times"] = project_completion_times

        return stats_dict

    def get_estimate_summary(self) -> pd.DataFrame:
        """Get summary of all task estimates"""
        data = []
        for task_id, estimate in self.task_estimates.items():
            data.append({
                "Task ID": task_id,
                "Optimistic": estimate.optimistic,
                "Most Likely": estimate.most_likely,
                "Pessimistic": estimate.pessimistic,
                "Expected (PERT)": estimate.expected_duration,
                "Std Deviation": estimate.standard_deviation,
                "Distribution": estimate.distribution_type.value
            })

        return pd.DataFrame(data)


# Integration functions for existing codebase
def create_enhanced_monte_carlo_from_dataframe(task_df: pd.DataFrame,
                                               use_manual_estimates: bool = False) -> EnhancedMonteCarlo:
    """Create enhanced Monte Carlo from existing task DataFrame"""
    mc = EnhancedMonteCarlo()

    if use_manual_estimates:
        # Check if DataFrame has 3-point estimate columns
        required_cols = ["Optimistic (O)", "Most Likely (M)", "Pessimistic (P)"]
        if all(col in task_df.columns for col in required_cols):
            for i, row in task_df.iterrows():
                distribution_type = DistributionType.TRIANGULAR
                if "Distribution Type" in task_df.columns:
                    dist_str = row["Distribution Type"].lower()
                    if "beta" in dist_str or "pert" in dist_str:
                        distribution_type = DistributionType.BETA_PERT

                mc.add_task_estimate(
                    task_id=row["ID"],
                    optimistic=row["Optimistic (O)"],
                    most_likely=row["Most Likely (M)"],
                    pessimistic=row["Pessimistic (P)"],
                    distribution_type=distribution_type
                )
        else:
            # Fall back to auto-generation
            mc.auto_generate_estimates_from_risk(task_df)
    else:
        # Auto-generate from risk levels
        mc.auto_generate_estimates_from_risk(task_df)

    return mc


def run_enhanced_monte_carlo_simulation(task_df: pd.DataFrame,
                                        adjacency_matrix: np.ndarray,
                                        num_simulations: int = 1000,
                                        confidence_levels: List[int] = [80, 90, 95],
                                        use_manual_estimates: bool = False) -> Dict:
    """
    Run enhanced Monte Carlo simulation with 3-point estimation

    Args:
        task_df: DataFrame with task data
        adjacency_matrix: Task dependency matrix
        num_simulations: Number of Monte Carlo runs
        confidence_levels: Confidence levels for statistics
        use_manual_estimates: Whether to use manual 3-point estimates

    Returns:
        Dictionary with comprehensive simulation results
    """

    # Create enhanced Monte Carlo instance
    mc = create_enhanced_monte_carlo_from_dataframe(task_df, use_manual_estimates)

    # Sample all task durations
    all_durations = mc.sample_all_tasks(num_simulations)
    num_tasks = len(task_df)

    # Storage for results
    all_start_times = np.zeros((num_tasks, num_simulations))
    all_finish_times = np.zeros((num_tasks, num_simulations))
    critical_path_count = np.zeros(num_tasks)

    # Run simulations
    for sim in range(num_simulations):
        sim_durations = all_durations[:, sim]

        # Calculate schedule for this simulation
        start_times = np.zeros(num_tasks)
        finish_times = np.zeros(num_tasks)

        # Forward pass scheduling
        for i in range(num_tasks):
            # Find dependencies (predecessors)
            predecessors = np.where(adjacency_matrix[:, i] > 0)[0]

            if len(predecessors) > 0:
                start_times[i] = np.max(finish_times[predecessors])
            else:
                start_times[i] = 0

            finish_times[i] = start_times[i] + sim_durations[i]

        # Store results
        all_start_times[:, sim] = start_times
        all_finish_times[:, sim] = finish_times

        # Calculate critical path for this simulation
        project_duration = np.max(finish_times)
        critical_tasks = np.where(np.abs(finish_times - project_duration) < 0.001)[0]
        critical_path_count[critical_tasks] += 1

    # Calculate comprehensive statistics
    results = mc.calculate_schedule_statistics(
        all_start_times, all_finish_times, confidence_levels
    )

    # Add critical path analysis
    results["task_criticality"] = (critical_path_count / num_simulations) * 100
    results["critical_path_analysis"] = {
        f"task_{i + 1}": float(critical_path_count[i] / num_simulations * 100)
        for i in range(num_tasks)
    }

    # Add estimation summary
    results["estimation_summary"] = mc.get_estimate_summary().to_dict(orient='records')

    # Add confidence levels used
    results["confidence_levels"] = confidence_levels

    return results


def add_estimation_columns_to_dataframe(task_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 3-point estimation columns to existing task DataFrame

    Returns new DataFrame with additional columns for manual estimation
    """
    df_enhanced = task_df.copy()

    # Add 3-point estimation columns if they don't exist
    if "Optimistic (O)" not in df_enhanced.columns:
        df_enhanced["Optimistic (O)"] = df_enhanced["Duration (days)"] * 0.8

    if "Most Likely (M)" not in df_enhanced.columns:
        df_enhanced["Most Likely (M)"] = df_enhanced["Duration (days)"]

    if "Pessimistic (P)" not in df_enhanced.columns:
        df_enhanced["Pessimistic (P)"] = df_enhanced["Duration (days)"] * 1.4

    if "Distribution Type" not in df_enhanced.columns:
        df_enhanced["Distribution Type"] = "Triangular"

    if "Expected (PERT)" not in df_enhanced.columns:
        df_enhanced["Expected (PERT)"] = (
                                                 df_enhanced["Optimistic (O)"] +
                                                 4 * df_enhanced["Most Likely (M)"] +
                                                 df_enhanced["Pessimistic (P)"]
                                         ) / 6

    return df_enhanced


def validate_3point_estimates(task_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate 3-point estimates in DataFrame

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    required_cols = ["Optimistic (O)", "Most Likely (M)", "Pessimistic (P)"]
    missing_cols = [col for col in required_cols if col not in task_df.columns]

    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors

    for i, row in task_df.iterrows():
        task_id = row.get("ID", i + 1)
        optimistic = row["Optimistic (O)"]
        most_likely = row["Most Likely (M)"]
        pessimistic = row["Pessimistic (P)"]

        # Check ordering
        if not (optimistic <= most_likely <= pessimistic):
            errors.append(
                f"Task {task_id}: Invalid ordering O({optimistic}) <= M({most_likely}) <= P({pessimistic})"
            )

        # Check for negative values
        if optimistic < 0 or most_likely < 0 or pessimistic < 0:
            errors.append(f"Task {task_id}: Negative duration values not allowed")

        # Check for reasonable ranges (pessimistic shouldn't be > 5x optimistic)
        if pessimistic > optimistic * 5:
            errors.append(
                f"Task {task_id}: Very large range - P({pessimistic}) > 5 * O({optimistic}). "
                f"Consider reviewing estimates."
            )

    return len(errors) == 0, errors


# Example usage and testing
if __name__ == "__main__":
    # Example usage with test data
    import pandas as pd

    # Create test DataFrame
    test_df = pd.DataFrame({
        "ID": [1, 2, 3, 4],
        "Task": ["Requirements", "Design", "Development", "Testing"],
        "Duration (days)": [10, 15, 25, 8],
        "Dependencies (IDs)": ["", "1", "2", "3"],
        "Risk (0-5)": [1, 2, 3, 1]
    })

    # Add estimation columns
    enhanced_df = add_estimation_columns_to_dataframe(test_df)
    print("Enhanced DataFrame with 3-point estimates:")
    print(enhanced_df[["Task", "Optimistic (O)", "Most Likely (M)", "Pessimistic (P)", "Expected (PERT)"]])

    # Create adjacency matrix
    adjacency = np.array([
        [0, 1, 0, 0],  # Task 1 -> Task 2
        [0, 0, 1, 0],  # Task 2 -> Task 3
        [0, 0, 0, 1],  # Task 3 -> Task 4
        [0, 0, 0, 0]  # Task 4 (final)
    ])

    # Run enhanced simulation
    results = run_enhanced_monte_carlo_simulation(
        enhanced_df, adjacency, num_simulations=1000
    )

    print(f"\nSimulation Results:")
    print(f"Mean project duration: {results['mean_project_duration']:.2f} days")
    print(f"Standard deviation: {results['std_project_duration']:.2f} days")
    print(
        f"90% confidence interval: [{results['project_duration_p5']:.1f}, {results['project_duration_p95']:.1f}] days")

    print(f"\nTask Criticality (% of time on critical path):")
    for i, criticality in enumerate(results['task_criticality']):
        print(f"Task {i + 1}: {criticality:.1f}%")