# app.py - Main Streamlit application
# Minor test change for Git push

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
    render_simulation_results,
    render_dependency_tab,
    render_classical_gantt,
    render_pde_gantt,
    render_eigenvalue_tab
)
from project_templates import ProjectTemplates

def main():
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

        from view import render_sde_gantt
        render_sde_gantt(model, controller)



    # Tab 7: Simulation Results (update index)
    with tabs[6]:
        render_simulation_results(model)

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