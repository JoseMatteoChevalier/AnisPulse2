# run_all_simulations.py - Comprehensive test script for all simulation methods
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import sys
import os

# Add project root to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import TaskModel
from controller import MainController
from project_templates import ProjectTemplates


class SimulationTester:
    """Comprehensive testing suite for all simulation methods"""

    def __init__(self):
        self.model = TaskModel()
        self.controller = MainController(self.model)
        self.test_results = {}

    def get_standard_test_project(self) -> pd.DataFrame:
        """Standard test project with known characteristics for validation"""
        return pd.DataFrame({
            "Select": [False] * 6,
            "ID": [1, 2, 3, 4, 5, 6],
            "Task": [
                "Requirements Analysis",
                "Database Design",
                "Backend API Development",
                "Third-party Integration",
                "Frontend UI Development",
                "Testing & Deployment"
            ],
            "Duration (days)": [10, 15, 20, 12, 18, 8],
            "Dependencies (IDs)": ["", "1", "2", "3", "2", "4,5"],
            "Risk (0-5)": [0.5, 1.0, 2.5, 1.5, 2.0, 1.0],
            "Parent ID": [""] * 6
        })

    def setup_test_model(self) -> bool:
        """Setup model with standard test data"""
        try:
            test_df = self.get_standard_test_project()
            self.model.task_df = test_df
            self.model.save_tasks()
            return True
        except Exception as e:
            print(f"❌ Error setting up test model: {e}")
            return False

    def run_basic_schedule_test(self) -> Dict[str, Any]:
        """Test basic CPM scheduling (no risk)"""
        print("🔄 Testing Basic Schedule...")
        start_time = time.time()

        try:
            # Calculate basic schedule from task data
            task_df = self.model.task_df
            num_tasks = len(task_df)

            # Simple forward pass scheduling
            start_times = np.zeros(num_tasks)
            finish_times = np.zeros(num_tasks)
            durations = task_df["Duration (days)"].values

            for i, row in task_df.iterrows():
                deps = str(row['Dependencies (IDs)']).split(",") if row['Dependencies (IDs)'] else []
                deps = [int(d.strip()) - 1 for d in deps if d.strip().isdigit() and 0 <= int(d.strip()) - 1 < num_tasks]

                if deps:
                    start_times[i] = max(finish_times[d] for d in deps) if deps else 0
                else:
                    start_times[i] = 0
                finish_times[i] = start_times[i] + durations[i]

            execution_time = time.time() - start_time
            project_duration = np.max(finish_times)

            return {
                "method": "Basic Schedule",
                "project_duration": project_duration,
                "execution_time": execution_time,
                "status": "✅ Success",
                "start_times": start_times,
                "finish_times": finish_times
            }

        except Exception as e:
            return {
                "method": "Basic Schedule",
                "status": f"❌ Error: {e}",
                "execution_time": time.time() - start_time
            }

    def run_classical_with_risk_test(self) -> Dict[str, Any]:
        """Test classical scheduling with risk factors"""
        print("🔄 Testing Classical with Risk...")
        start_time = time.time()

        try:
            # Run basic simulation to get classical risk calculation
            success, error = self.controller.run_simulation(
                diffusion=0.02,
                reaction_multiplier=2.0,
                max_delay=0.05
            )

            execution_time = time.time() - start_time

            if success:
                classical_times = self.model.simulation_data.get("finish_times_classical")
                if classical_times is not None:
                    project_duration = np.max(classical_times)

                    return {
                        "method": "Classical + Risk",
                        "project_duration": project_duration,
                        "execution_time": execution_time,
                        "status": "✅ Success",
                        "finish_times": classical_times
                    }

            return {
                "method": "Classical + Risk",
                "status": f"❌ Error: {error}",
                "execution_time": execution_time
            }

        except Exception as e:
            return {
                "method": "Classical + Risk",
                "status": f"❌ Exception: {e}",
                "execution_time": time.time() - start_time
            }

    def run_pde_test(self) -> Dict[str, Any]:
        """Test PDE simulation"""
        print("🔄 Testing PDE Simulation...")
        start_time = time.time()

        try:
            # PDE should already be run from classical test
            # Check if PDE results exist
            pde_times = self.model.simulation_data.get("finish_times_risk")
            execution_time = time.time() - start_time

            if pde_times is not None:
                project_duration = np.max(pde_times)

                return {
                    "method": "PDE",
                    "project_duration": project_duration,
                    "execution_time": execution_time,
                    "status": "✅ Success",
                    "finish_times": pde_times
                }
            else:
                return {
                    "method": "PDE",
                    "status": "❌ No PDE results available",
                    "execution_time": execution_time
                }

        except Exception as e:
            return {
                "method": "PDE",
                "status": f"❌ Exception: {e}",
                "execution_time": time.time() - start_time
            }

    def run_monte_carlo_test(self, fast_mode=True) -> Dict[str, Any]:
        """Test Monte Carlo simulation"""
        print("🔄 Testing Monte Carlo...")
        start_time = time.time()

        try:
            # Use fast parameters for testing
            num_sims = 100 if fast_mode else 1000
            confidence_levels = [90] if fast_mode else [80, 90, 95]

            success, error = self.controller.run_monte_carlo(
                num_simulations=num_sims,
                confidence_levels=confidence_levels
            )

            execution_time = time.time() - start_time

            if success:
                mc_results = self.model.simulation_data.get("monte_carlo_results")
                if mc_results:
                    return {
                        "method": "Monte Carlo",
                        "project_duration": mc_results["mean_completion"],
                        "std_deviation": mc_results["std_completion"],
                        "execution_time": execution_time,
                        "status": "✅ Success",
                        "num_simulations": num_sims,
                        "confidence_levels": confidence_levels
                    }

            return {
                "method": "Monte Carlo",
                "status": f"❌ Error: {error}",
                "execution_time": execution_time
            }

        except Exception as e:
            return {
                "method": "Monte Carlo",
                "status": f"❌ Exception: {e}",
                "execution_time": time.time() - start_time
            }

    # Add this to your run_all_simulations.py file

    def run_enhanced_monte_carlo_test(self, fast_mode=True) -> Dict[str, Any]:
        """Test Enhanced Monte Carlo simulation with 3-point estimation"""
        print("🔄 Testing Enhanced Monte Carlo...")
        start_time = time.time()

        try:
            # Import the enhanced Monte Carlo
            from enhanced_monte_carlo import run_enhanced_monte_carlo_simulation, add_estimation_columns_to_dataframe

            # Get adjacency matrix
            adjacency, error = self.model.build_adjacency()
            if error:
                return {
                    "method": "Enhanced Monte Carlo",
                    "status": f"❌ Adjacency Error: {error}",
                    "execution_time": time.time() - start_time
                }

            # Use fast parameters for testing
            num_sims = 100 if fast_mode else 1000
            confidence_levels = [90] if fast_mode else [80, 90, 95]

            # Add 3-point estimation columns to task DataFrame
            enhanced_df = add_estimation_columns_to_dataframe(self.model.task_df)

            # Run enhanced Monte Carlo simulation
            results = run_enhanced_monte_carlo_simulation(
                task_df=enhanced_df,
                adjacency_matrix=adjacency,
                num_simulations=num_sims,
                confidence_levels=confidence_levels,
                use_manual_estimates=False  # Use auto-generated estimates
            )

            execution_time = time.time() - start_time

            return {
                "method": "Enhanced Monte Carlo",
                "project_duration": results["mean_project_duration"],
                "std_deviation": results["std_project_duration"],
                "confidence_interval_90": [
                    results.get("project_duration_p5", 0),
                    results.get("project_duration_p95", 0)
                ],
                "execution_time": execution_time,
                "status": "✅ Success",
                "num_simulations": num_sims,
                "confidence_levels": confidence_levels,
                "task_criticality": results["task_criticality"],
                "distribution_types": "Auto-generated (Risk-based)"
            }

        except ImportError as e:
            return {
                "method": "Enhanced Monte Carlo",
                "status": f"❌ Import Error: {e}",
                "execution_time": time.time() - start_time
            }
        except Exception as e:
            return {
                "method": "Enhanced Monte Carlo",
                "status": f"❌ Exception: {e}",
                "execution_time": time.time() - start_time
            }

    # Also update the comprehensive test to include enhanced Monte Carlo
    def run_comprehensive_test(self, fast_mode=True) -> Dict[str, Any]:
        """Run all simulation methods and compare results"""
        print("=" * 60)
        print("🚀 COMPREHENSIVE SIMULATION TEST SUITE")
        print("=" * 60)

        # Setup test project
        if not self.setup_test_model():
            return {"error": "Failed to setup test model"}

        # Run all tests
        results = {}

        # 1. Basic Schedule (naive)
        results["basic"] = self.run_basic_schedule_test()

        # 2. Classical + Risk (includes PDE)
        results["classical_risk"] = self.run_classical_with_risk_test()
        results["pde"] = self.run_pde_test()

        # 3. Original Monte Carlo
        results["monte_carlo"] = self.run_monte_carlo_test(fast_mode)

        # 4. Enhanced Monte Carlo (NEW)
        results["enhanced_monte_carlo"] = self.run_enhanced_monte_carlo_test(fast_mode)

        # 5. SDE
        results["sde"] = self.run_sde_test(fast_mode)

        # Store results
        self.test_results = results

        return results

    # Enhanced print function to show more details
    def print_enhanced_comparison_table(self, results: Dict[str, Any]):
        """Print enhanced comparison table with confidence intervals"""
        print("\n" + "=" * 100)
        print("📊 ENHANCED SIMULATION METHOD COMPARISON")
        print("=" * 100)

        # Header
        print(f"{'Method':<18} | {'Duration':<12} | {'Std Dev':<10} | {'90% CI':<16} | {'Status':<20} | {'Time':<8}")
        print("-" * 100)

        # Results
        for method_key, result in results.items():
            if isinstance(result, dict):
                method = result.get("method", method_key)

                # Duration
                duration = result.get("project_duration", "N/A")
                if isinstance(duration, (int, float)):
                    duration_str = f"{duration:.1f} days"
                else:
                    duration_str = str(duration)

                # Standard deviation
                std_dev = result.get("std_deviation", "")
                if isinstance(std_dev, (int, float)) and std_dev > 0:
                    std_str = f"±{std_dev:.1f}"
                else:
                    std_str = ""

                # Confidence interval
                ci = result.get("confidence_interval_90", "")
                if isinstance(ci, list) and len(ci) == 2:
                    ci_str = f"[{ci[0]:.1f}, {ci[1]:.1f}]"
                else:
                    ci_str = ""

                # Status
                status = result.get("status", "Unknown")

                # Execution time
                exec_time = result.get("execution_time", 0)
                time_str = f"{exec_time:.3f}s"

                print(
                    f"{method:<18} | {duration_str:<12} | {std_str:<10} | {ci_str:<16} | {status:<20} | {time_str:<8}")

        print("=" * 100)

        # Show critical path analysis for enhanced methods
        print("\n🎯 CRITICAL PATH ANALYSIS")
        print("-" * 40)

        for method_key, result in results.items():
            if isinstance(result, dict) and "task_criticality" in result:
                method = result.get("method", method_key)
                criticality = result["task_criticality"]

                print(f"\n{method}:")
                for i, crit_pct in enumerate(criticality):
                    if crit_pct > 5:  # Only show tasks with >5% criticality
                        print(f"  Task {i + 1}: {crit_pct:.1f}% critical")

    def run_sde_test(self, fast_mode=True) -> Dict[str, Any]:
        """Test SDE simulation with lightweight parameters"""
        print("🔄 Testing SDE Simulation...")
        start_time = time.time()

        try:
            # Check if SDE integration exists
            if hasattr(self.controller, 'sde_integration'):
                from sde_solver import SDEParameters

                # ULTRA-FAST parameters for testing
                if fast_mode:
                    sde_params = SDEParameters(
                        dt=0.5,  # Less aggressive (was 1.0)
                        T=80.0,  # Longer simulation (was 20.0)
                        n_paths=25,  # More paths (was 10)
                        volatility=0.15
                    )
                else:
                    sde_params = SDEParameters()

                success, error = self.controller.run_sde_simulation(sde_params)
                execution_time = time.time() - start_time

                if success:
                    sde_results = self.model.simulation_data.get("sde_results")
                    if sde_results:
                        mean_duration = sde_results.risk_metrics.get('mean_project_duration', 0)
                        std_duration = sde_results.risk_metrics.get('std_project_duration', 0)

                        return {
                            "method": "SDE",
                            "project_duration": mean_duration,
                            "std_deviation": std_duration,
                            "execution_time": execution_time,
                            "status": "✅ Success",
                            "n_paths": sde_params.n_paths
                        }

                return {
                    "method": "SDE",
                    "status": f"❌ Error: {error}",
                    "execution_time": execution_time
                }
            else:
                return {
                    "method": "SDE",
                    "status": "⚠️ SDE integration not available",
                    "execution_time": time.time() - start_time
                }

        except Exception as e:  # ← ADD THIS EXCEPT BLOCK
            return {
                "method": "SDE",
                "status": f"❌ Exception: {e}",
                "execution_time": time.time() - start_time
            }
    def print_enhanced_comparison_table(self, results: Dict[str, Any]):
        """Print enhanced comparison table with confidence intervals"""
        print("\n" + "=" * 100)
        print("📊 ENHANCED SIMULATION METHOD COMPARISON")
        print("=" * 100)

        # Header
        print(f"{'Method':<18} | {'Duration':<12} | {'Std Dev':<10} | {'90% CI':<16} | {'Status':<20} | {'Time':<8}")
        print("-" * 100)

        # Results
        for method_key, result in results.items():
            if isinstance(result, dict):
                method = result.get("method", method_key)

                # Duration
                duration = result.get("project_duration", "N/A")
                if isinstance(duration, (int, float)):
                    duration_str = f"{duration:.1f} days"
                else:
                    duration_str = str(duration)

                # Standard deviation
                std_dev = result.get("std_deviation", "")
                if isinstance(std_dev, (int, float)) and std_dev > 0:
                    std_str = f"±{std_dev:.1f}"
                else:
                    std_str = ""

                # Confidence interval
                ci = result.get("confidence_interval_90", "")
                if isinstance(ci, list) and len(ci) == 2:
                    ci_str = f"[{ci[0]:.1f}, {ci[1]:.1f}]"
                else:
                    ci_str = ""

                # Status
                status = result.get("status", "Unknown")

                # Execution time
                exec_time = result.get("execution_time", 0)
                time_str = f"{exec_time:.3f}s"

                print(
                    f"{method:<18} | {duration_str:<12} | {std_str:<10} | {ci_str:<16} | {status:<20} | {time_str:<8}")

        print("=" * 100)

        # Show critical path analysis for enhanced methods
        print("\n🎯 CRITICAL PATH ANALYSIS")
        print("-" * 40)

        for method_key, result in results.items():
            if isinstance(result, dict) and "task_criticality" in result:
                method = result.get("method", method_key)
                criticality = result["task_criticality"]

                print(f"\n{method}:")
                for i, crit_pct in enumerate(criticality):
                    if crit_pct > 5:  # Only show tasks with >5% criticality
                        print(f"  Task {i + 1}: {crit_pct:.1f}% critical")

    def validate_consistency(self, results: Dict[str, Any]):
        """Validate that results are consistent and reasonable"""
        print("\n🔍 VALIDATION CHECKS")
        print("-" * 40)

        # Extract durations where available
        durations = {}
        for method_key, result in results.items():
            if isinstance(result, dict) and "project_duration" in result:
                duration = result["project_duration"]
                if isinstance(duration, (int, float)):
                    durations[method_key] = duration

        if len(durations) >= 2:
            # Check ordering: Basic ≤ Classical ≤ PDE/MC/SDE
            basic_dur = durations.get("basic", 0)
            classical_dur = durations.get("classical_risk", 0)

            if basic_dur > 0 and classical_dur > 0:
                if basic_dur <= classical_dur:
                    print("✅ Basic ≤ Classical+Risk (expected)")
                else:
                    print("⚠️  Basic > Classical+Risk (unexpected)")

            # Check that stochastic methods show reasonable spread
            stochastic_methods = ["monte_carlo", "sde"]
            for method in stochastic_methods:
                if method in results and "std_deviation" in results[method]:
                    std = results[method]["std_deviation"]
                    mean = results[method]["project_duration"]
                    if isinstance(std, (int, float)) and isinstance(mean, (int, float)) and mean > 0:
                        cv = std / mean
                        if 0.1 <= cv <= 0.5:
                            print(f"✅ {method} coefficient of variation: {cv:.2f} (reasonable)")
                        else:
                            print(f"⚠️  {method} coefficient of variation: {cv:.2f} (check parameters)")

        # Check execution times
        total_time = sum(r.get("execution_time", 0) for r in results.values() if isinstance(r, dict))
        print(f"⏱️  Total execution time: {total_time:.3f}s")

        if total_time < 2.0:
            print("✅ Fast execution suitable for development")
        elif total_time < 10.0:
            print("⚠️  Moderate execution time")
        else:
            print("🐌 Slow execution - consider optimization")


def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive simulation testing")
    parser.add_argument("--fast", action="store_true", help="Use fast parameters for quick testing")
    parser.add_argument("--method", type=str, help="Test specific method only")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    tester = SimulationTester()

    if args.method:
        # Test specific method
        print(f"Testing {args.method} only...")
        # Could implement specific method testing here
    else:
        # Run comprehensive test
        results = tester.run_comprehensive_test(fast_mode=args.fast)
        tester.print_enhanced_comparison_table(results)
        tester.validate_consistency(results)

    print("\n🎯 Test completed!")


if __name__ == "__main__":
    main()