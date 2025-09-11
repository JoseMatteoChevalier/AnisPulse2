# app.py - Main Streamlit application
# Minor test change for Git push
#Adding password check
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


from view import render_monte_carlo_gantt_chart
from model import TaskModel
from controller import MainController
from view import (
    render_sidebar,
    render_editor_tab,
    render_basic_schedule_tab,  # NEW IMPORT
    render_simulation_results_plotly_test,
    render_dependency_tab,
    render_classical_gantt,
    render_pde_gantt,
    render_eigenvalue_tab
)
from project_templates import ProjectTemplates


# Add this after your imports, before main()
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "ProjectRisk2024":  # Change this password
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "🔒 Enter Password to Access Project Uncertainty Toolkit",
        type="password",
        on_change=password_entered,
        key="password",
        help="This app contains proprietary project risk algorithms"
    )

    if "password_correct" in st.session_state:
        st.error("😞 Password incorrect")

    return False


# Add this at the very start of your main() function
def main():
    # Password protection - add this as the FIRST line in main()
    if not check_password():
        st.stop()

    # Your existing code continues here...
    st.set_page_config(
        page_title="Project Management PDE Simulator",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # ... rest of your existing main() function

def main():

    if not check_password():
        st.stop()

    st.set_page_config(
        page_title="Project Management PDE Simulator",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🚀 Project Management PDE Simulator")
    st.markdown("*Advanced project scheduling with reaction-diffusion modeling*")

    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = TaskModel()
        st.session_state.controller = MainController(st.session_state.model)

    model = st.session_state.model
    controller = st.session_state.controller

    # Sidebar with simulation parameters
    diffusion, reaction_multiplier, max_delay = render_sidebar(model)

    # Update session state with current parameters
    st.session_state.diffusion = diffusion
    st.session_state.reaction_multiplier = reaction_multiplier
    st.session_state.max_delay = max_delay

    # Main tabs - Basic Schedule is now SECOND tab (most important baseline)
    tab_names = [
        "📝 Editor",
        "📅 Basic Schedule",
        "📊 Classical Gantt",
        "🌊 PDE Gantt",
        "🎲 Monte Carlo Gantt",
        "📊 SDE Gantt",
        "📈 Simulation",
        "🔗 Dependencies",
        "🧮 Eigenvalues"
    ]
    tabs = st.tabs(tab_names)

    # Tab 1: Editor
    with tabs[0]:

        st.markdown("---")
        st.subheader("📋 Project Templates")
        templates = ProjectTemplates.get_available_templates()
        selected_template = st.selectbox("Choose a template:", ["Custom Project"] + templates)

        if st.button("🔄 Load Template") and selected_template != "Custom Project":
            success, error = controller.load_template(selected_template)
            if success:
                st.success(f"Loaded template: {selected_template}")
                st.rerun()
            else:
                st.error(error)

        task_df, mpp_file = render_editor_tab(model)

        # Add template selector


        # Handle editor interactions
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.session_state.get("add_row", False):
                success, error = controller.add_task(
                    name=f"New Task {len(model.task_df) + 1}",
                    duration=7.0,
                    dependencies=[],
                    risk_level=0.0
                )
                if success:
                    st.success("Task added successfully!")
                else:
                    st.error(error)
                st.rerun()

        with col2:
            if st.session_state.get("delete_rows", False):
                success, error = controller.delete_selected_tasks()
                if success:
                    st.success("Selected tasks deleted!")
                else:
                    st.error(error)
                st.rerun()

        with col3:
            # In the Editor tab, replace the recalculate button section with:
            if st.session_state.get("recalculate", False):
                # Update model with edited DataFrame
                success, error = controller.update_tasks(task_df)
                if not success:
                    st.error(error)
                    st.rerun()

                # Run simulation
                success, error = controller.run_simulation(
                    diffusion=diffusion,
                    reaction_multiplier=reaction_multiplier,
                    max_delay=max_delay
                )
                if success:
                    st.success("Tasks updated and simulation completed!")
                else:
                    st.error(f"Simulation failed: {error}")
                st.rerun()
        # Handle MPP file import
        if mpp_file is not None:
            success, error = controller.import_mpp(mpp_file)
            if success:
                st.success("MPP file imported successfully!")
                st.rerun()
            else:
                st.error(error)

    # Tab 2: Basic Schedule (NEW - MOST IMPORTANT)
    with tabs[1]:
        with st.expander("ℹ️ About Basic Schedule Analysis", expanded=False):
            st.markdown("""
               **What Basic Schedule Does:**
               - Creates a pure dependency-based project schedule using task durations and precedence relationships
               - Calculates the critical path - the sequence of tasks that determines minimum project duration
               - Shows task float (slack time) - how much tasks can be delayed without affecting the project
               - Provides the baseline foundation for all other advanced analyses

               **This is Your Project's Foundation:**
               - **No risk modeling** - uses task durations exactly as entered
               - **No uncertainty** - assumes everything goes according to plan
               - **Pure logic** - only considers task dependencies and durations
               - **Industry standard** - follows traditional Critical Path Method (CPM)

               **Key Concepts Explained:**
               - **Critical Path (Red tasks):** Chain of tasks with zero float - any delay here delays the entire project
               - **Float/Slack (Gray bars):** Extra time available before a task becomes critical
               - **Early Start:** Earliest possible start time based on predecessor completion
               - **Late Start:** Latest start time without delaying the project
               - **Total Duration:** Sum of critical path task durations

               **How to Read the Results:**
               - **Critical Path Tasks:** Focus management attention here - zero tolerance for delays
               - **Float Time:** Tasks with float can be delayed without project impact
               - **Project Duration:** Minimum time needed if everything goes perfectly
               - **Task Dependencies:** Arrows show which tasks must finish before others can start

               **When to Use Basic Schedule:**
               - **Project planning baseline** - start here before adding risk analysis
               - **Stakeholder communication** - simple, clear timeline without complexity
               - **Resource planning** - understand task sequencing and timing
               - **Comparison baseline** - see how risk affects your "perfect world" schedule

               **💡 Pro Tips:**
               - Critical path tasks need the most attention and resources
               - Tasks with high float can be resource-leveled or delayed if needed
               - This schedule assumes no risks - compare with advanced methods to see uncertainty impact
               - Use this as your "best case scenario" for stakeholder discussions
               """)

        render_basic_schedule_tab(model)

        # Add export functionality
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Export Basic Schedule", help="Export the basic schedule as CSV"):
                try:
                    # This would need to be implemented in the view function
                    st.info("Export functionality coming soon!")
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")

        with col2:
            if st.button("🔄 Refresh Schedule", help="Recalculate the basic schedule"):
                st.rerun()

    # Tab 3: Classical Gantt
    with tabs[2]:
        render_classical_gantt(model)

        st.markdown("---")
        st.info(
            "💡 **Note**: This shows the classical schedule with risk factors applied to durations, but no diffusion effects.")

    # Tab 4: PDE Gantt
    with tabs[3]:
        render_pde_gantt(model)

        st.markdown("---")
        st.info("💡 **Note**: This shows the PDE simulation results with risk diffusion between dependent tasks.")


    # Tab 5: Monte Carlo Gantt
    with tabs[4]:
        st.subheader("🎲 Monte Carlo Gantt Chart")

        # ADD THIS DOCUMENTATION SECTION
        with st.expander("ℹ️ About Monte Carlo Analysis", expanded=False):
            st.markdown("""
                **What Monte Carlo Analysis Does:**
                - Runs thousands of simulations with varying task durations based on risk levels
                - Shows the range of possible project outcomes and completion times
                - Identifies which tasks are most likely to cause project delays (criticality analysis)
                - Provides confidence intervals for project planning and risk management

                **When to Use Monte Carlo:**
                - When you need probabilistic project estimates instead of single-point forecasts
                - For risk-aware project planning and stakeholder communication
                - To identify the most critical tasks that require extra attention
                - When project uncertainty is high and you need confidence bounds

                **Parameter Guide:**
                - **Number of Simulations:** 
                  - 1,000: Fast, good for initial analysis
                  - 5,000: Better statistics, recommended for most projects
                  - 10,000: High precision, use for critical/high-stakes projects
                - **Confidence Levels:** 
                  - 80%: Basic planning scenarios
                  - 90%: Standard project management (recommended)
                  - 95%: High-stakes or critical projects

                **Understanding Your Results:**
                - **Criticality %:** Percentage of simulations where each task was on the critical path
                  - >80%: High risk tasks - monitor closely and plan contingencies
                  - 50-80%: Medium risk tasks - watch for delays
                  - <50%: Lower risk tasks - standard monitoring
                - **Mean Completion:** Best estimate for project completion time
                - **Confidence Bands:** Range of likely completion times (gray area in charts)
                - **Percentiles:** P90 means 90% of simulations finished by that time

                **💡 Pro Tips:**
                - Focus management attention on tasks with >50% criticality
                - Use P90 completion time for stakeholder commitments
                - Compare with Classical duration to see impact of uncertainty
                - Higher risk levels (0-5) create wider completion time distributions
                """)

        if not model.simulation_data.get("monte_carlo_results"):
            # Parameters section
            col1, col2 = st.columns(2)
            with col1:
                num_sims = st.selectbox("Number of Simulations", [1000, 5000, 10000], index=0)
            with col2:
                confidence = st.multiselect("Confidence Levels", [80, 90, 95], default=[90])

            if st.button("🎲 Run Monte Carlo Analysis", type="primary"):
                with st.spinner("Running Monte Carlo simulation..."):
                    success, error = controller.run_monte_carlo(
                        num_simulations=num_sims,
                        confidence_levels=confidence
                    )
                    if success:
                        st.success("Monte Carlo analysis completed!")
                        st.rerun()
                    else:
                        st.error(f"Monte Carlo failed: {error}")
        else:
            # ADD DEBUG INFO HERE (temporarily for testing)
            from view import render_monte_carlo_debug_info
            render_monte_carlo_debug_info(model)

            # Display results
            results = model.simulation_data["monte_carlo_results"]

            # Add the enhanced Gantt chart
            render_monte_carlo_gantt_chart(model)

            st.divider()  # Visual separator

            # Task criticality analysis
            st.subheader("Task Criticality Index")
            st.write("Percentage of simulations where each task was on the critical path:")

            criticality_df = pd.DataFrame({
                'Task ID': range(1, len(model.task_df) + 1),
                'Task Name': model.task_df['Task'].values,
                'Critical Path %': results['task_criticality'].round(1),
                'Risk Level': model.task_df['Risk (0-5)'].values
            }).sort_values('Critical Path %', ascending=False)

            # Color code by criticality with visible text
            def highlight_criticality(row):
                if row['Critical Path %'] > 80:
                    return ['background-color: #ffebee; color: #000000'] * len(row)  # Light red with black text
                elif row['Critical Path %'] > 50:
                    return ['background-color: #fff3e0; color: #000000'] * len(row)  # Light orange with black text
                else:
                    return ['color: #000000'] * len(row)  # Just ensure black text

            styled_df = criticality_df.style.apply(highlight_criticality, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

            # Critical path insights
            high_criticality = criticality_df[criticality_df['Critical Path %'] > 80]
            medium_criticality = criticality_df[
                (criticality_df['Critical Path %'] > 50) & (criticality_df['Critical Path %'] <= 80)
                ]

            if not high_criticality.empty:
                st.error(f"🔴 High criticality tasks (>80%): {', '.join(high_criticality['Task Name'].tolist())}")
            if not medium_criticality.empty:
                st.warning(
                    f"🟡 Medium criticality tasks (50-80%): {', '.join(medium_criticality['Task Name'].tolist())}")

            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Completion", f"{results['mean_completion']:.1f} days")
            with col2:
                st.metric("Std Deviation", f"{results['std_completion']:.1f} days")
            with col3:
                finish_times_classical = model.simulation_data.get("finish_times_classical")
                if finish_times_classical is not None and len(finish_times_classical) > 0:
                    classical_duration = np.max(finish_times_classical)
                    delay_risk = ((results[
                                       'mean_completion'] - classical_duration) / classical_duration * 100) if classical_duration > 0 else 0
                    st.metric("Delay Risk", f"+{delay_risk:.1f}%")
                else:
                    st.metric("Delay Risk", "N/A")
            with col4:
                st.metric("Simulations", f"{results['num_simulations']:,}")

            # Completion time histogram
            st.subheader("Project Completion Time Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(results['completion_times'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(results['mean_completion'], color='red', linestyle='--',
                       label=f"Mean: {results['mean_completion']:.1f}")

            # Add classical completion time for comparison
            finish_times_classical = model.simulation_data.get("finish_times_classical")
            if finish_times_classical is not None and len(finish_times_classical) > 0:
                classical_duration = np.max(finish_times_classical)
                ax.axvline(classical_duration, color='blue', linestyle='-',
                           label=f"Classical: {classical_duration:.1f}")

            ax.set_xlabel('Completion Time (days)')
            ax.set_ylabel('Frequency')
            ax.set_title('Monte Carlo Project Completion Distribution')
            ax.legend()
            st.pyplot(fig, use_container_width=True)

            # Percentiles table
            st.subheader("Completion Time Percentiles")
            percentiles_df = pd.DataFrame([results['percentiles']])
            st.dataframe(percentiles_df, use_container_width=True)
    # Tab 6: SDE Gantt
    with tabs[5]:
        st.subheader("📊 SDE Gantt Chart")
        # ADD THIS DOCUMENTATION SECTION
        with st.expander("ℹ️ About Stochastic Differential Equations (SDE)", expanded=False):
            st.markdown("""
                **What SDE Analysis Does:**
                - Models project progress as continuous stochastic processes over time
                - Captures risk propagation and dependencies between connected tasks
                - Shows multiple possible project paths with uncertainty bands
                - Provides advanced risk quantification beyond simple Monte Carlo

                **When to Use SDE:**
                - For complex projects with significant task interdependencies
                - When risks can cascade through connected tasks (contagion effects)
                - For sophisticated uncertainty modeling and risk analysis
                - When continuous-time modeling is more appropriate than discrete events

                **Key Differences from Monte Carlo:**
                - **Continuous:** Models progress smoothly over time vs. discrete outcomes
                - **Path-dependent:** Each simulation path influences future progress
                - **Risk propagation:** Delays in one task affect connected downstream tasks
                - **Advanced statistics:** Provides sophisticated risk metrics and analysis

                **Parameter Guide:**
                - **Number of Paths:** 
                  - 500: Fast computation, good for testing
                  - 1000: Recommended for most analyses
                  - 2000: High precision for critical projects
                - **Volatility (0.05-0.5):** 
                  - 0.1: Low uncertainty environment
                  - 0.15: Typical uncertainty (recommended start)
                  - 0.3+: High uncertainty/volatile environment
                - **Dependency Correlation (0-1):** 
                  - 0.3: Weak correlation between connected tasks
                  - 0.5: Moderate correlation (recommended)
                  - 0.8: Strong correlation - delays spread easily
                - **Jump Intensity:** Rate of sudden disruptions (0.1 = 10% chance per time unit)
                - **Risk Amplification:** How much task risk levels affect uncertainty

                **Understanding SDE Results:**
                - **Confidence Bands:** Show range of possible completion times
                - **Multiple Realizations:** Individual simulation paths showing different scenarios
                - **Distribution Plots:** Statistical analysis of completion time uncertainty
                - **Risk Metrics:** 
                  - VaR (Value at Risk): Worst-case scenarios at confidence levels
                  - Expected Shortfall: Average of worst-case outcomes
                  - Probability on-time: Chance of meeting original schedule

                **Advanced Features:**
                - **Path Exploration:** See individual simulation trajectories
                - **Risk Surface Analysis:** Understand parameter sensitivity
                - **Confidence Band Visualization:** Interactive exploration of uncertainty

                **💡 Pro Tips:**
                - Start with default parameters, then adjust based on your project characteristics
                - Higher volatility = wider uncertainty bands
                - Use correlation to model how tightly connected your tasks really are
                - Compare results with Monte Carlo to understand the difference
                - Focus on VaR metrics for stakeholder risk communication
                """)
        from view import render_sde_gantt
        render_sde_gantt(model, controller)



    # Tab 7: Simulation Results (update index)
    with tabs[6]:
        with st.expander("ℹ️ Understanding Simulation Comparisons", expanded=False):
            st.markdown("""
            **What This Comparison Shows:**
            - **Classical Risk (Blue line):** Project completion over time with risk factors applied to task durations
            - **Diffusion Risk (Red dashed line):** Advanced PDE modeling showing how risks propagate through task dependencies
            - **Side-by-side analysis:** See how different modeling approaches affect project predictions

            **The Two Methods Explained:**

            **Classical Risk Modeling:**
            - Applies risk multipliers directly to task durations (Risk Level 0-5 increases duration)
            - Tasks progress independently - no risk propagation between connected tasks
            - Deterministic approach - same inputs always give same results
            - Industry standard method used in most project management tools

            **PDE Diffusion Risk Modeling:**
            - Models risk as a diffusion process that spreads through task dependencies
            - Delays in high-risk tasks can cascade to connected downstream tasks
            - Accounts for risk propagation and amplification effects
            - Advanced mathematical modeling using partial differential equations

            **Reading the Charts:**

            **2D Plot (Left):**
            - **X-axis:** Time progression (days)
            - **Y-axis:** Average project completion (0 = start, 1 = complete)
            - **Blue solid line:** Classical risk progression
            - **Red dashed line:** PDE diffusion progression
            - **Steeper curve:** Faster overall completion

            **3D Plot (Right):**
            - **Same data** shown in 3D perspective for visual clarity
            - **Y-axis separation:** Classical (Y=0) vs PDE (Y=1) methods
            - **Z-axis:** Completion percentage over time

            **Key Insights to Look For:**
            - **Gap between curves:** Shows impact of risk propagation effects
            - **PDE curve typically slower:** Risk diffusion usually extends project duration
            - **Similar curves:** Low-risk projects or weak task interdependencies
            - **Large differences:** High-risk projects with strong dependencies benefit most from PDE modeling

            **When Each Method is Most Appropriate:**

            **Use Classical Risk When:**
            - Simple projects with independent tasks
            - Quick estimates needed for planning
            - Stakeholders prefer traditional approaches
            - Limited task interdependencies

            **Use PDE Diffusion When:**
            - Complex projects with significant task dependencies
            - High-risk environment where delays cascade
            - Advanced risk management needed
            - Accuracy more important than simplicity

            **💡 Interpretation Tips:**
            - **Curves close together:** Risk propagation effects are minimal
            - **PDE curve significantly delayed:** Strong interdependency effects present
            - **Use PDE results** for complex project risk management
            - **Use Classical results** for stakeholder communication and simple planning
            - **Compare both** to understand the range of possible outcomes
            """)
        render_simulation_results_plotly_test(model)

    # Tab 8: Dependencies (update index)
    with tabs[7]:
        render_dependency_tab(model)
        # ... rest of dependencies tab code

    # Tab 9: Eigenvalues (update index)
    with tabs[8]:
        render_eigenvalue_tab(model)
        # ... rest of eigenvalues tab code

    # Status footer in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Project Status")

    # Task count and validation
    task_count = len(model.task_df)
    st.sidebar.metric("Total Tasks", task_count)

    # Validation status
    if task_count > 0:
        valid, errors = controller.validate_tasks()
        if valid:
            st.sidebar.success("✅ All tasks valid")
        else:
            st.sidebar.error(f"❌ {len(errors)} validation errors")
            with st.sidebar.expander("View Errors"):
                for error in errors:
                    st.write(f"• {error}")

    # Simulation status
    simulation_status = controller.get_simulation_status()
    if simulation_status["has_results"]:
        st.sidebar.success("✅ Simulation complete")
        if simulation_status["has_classical"]:
            st.sidebar.info("📊 Classical comparison available")
    else:
        st.sidebar.info("⏳ Ready to simulate")

    # Quick help
    st.sidebar.markdown("---")
    st.sidebar.subheader("🆘 Quick Help")
    st.sidebar.write("1. **Start** with Basic Schedule tab")
    st.sidebar.write("2. **Validate** your task dependencies")
    st.sidebar.write("3. **Run** simulations to see risk effects")
    st.sidebar.write("4. **Compare** results across tabs")

    # Version info
    st.sidebar.markdown("---")
    st.sidebar.caption("PDE Project Simulator v1.0")
    st.sidebar.caption("Built with Streamlit")


if __name__ == "__main__":
    main()