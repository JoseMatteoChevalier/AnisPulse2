# project_templates.py
import pandas as pd
import numpy as np


class ProjectTemplates:
    """Collection of predefined project templates"""

    @staticmethod
    def get_available_templates():
        """Return list of available template names"""
        return [
            "Software Development (Default)",
            "Construction Project",
            "Marketing Campaign",
            "Product Launch",
            "Research & Development",
            "Event Planning",
            "Manufacturing Setup",
            "Website Redesign"
        ]

    @staticmethod
    def get_template(template_name):
        """Return DataFrame for specified template"""
        templates = {
            "Software Development (Default)": ProjectTemplates._software_development(),
            "Construction Project": ProjectTemplates._construction_project(),
            "Marketing Campaign": ProjectTemplates._marketing_campaign(),
            "Product Launch": ProjectTemplates._product_launch(),
            "Research & Development": ProjectTemplates._research_development(),
            "Event Planning": ProjectTemplates._event_planning(),
            "Manufacturing Setup": ProjectTemplates._manufacturing_setup(),
            "Website Redesign": ProjectTemplates._website_redesign()
        }
        return templates.get(template_name, ProjectTemplates._software_development())

    @staticmethod
    def _software_development():
        """Software development project template"""
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
            "Duration (days)": [14, 21, 28, 21, 35, 14],
            "Dependencies (IDs)": ["", "1", "2", "3", "2", "4,5"],
            "Risk (0-5)": [0.5, 1.0, 2.5, 1.5, 2.0, 1.0],
            "Parent ID": [""] * 6
        })

    @staticmethod
    def _construction_project():
        """Construction project template"""
        return pd.DataFrame({
            "Select": [False] * 8,
            "ID": [1, 2, 3, 4, 5, 6, 7, 8],
            "Task": [
                "Site Survey & Planning",
                "Foundation Work",
                "Structural Framework",
                "Electrical Installation",
                "Plumbing Installation",
                "Interior Finishing",
                "Exterior Work",
                "Final Inspection"
            ],
            "Duration (days)": [7, 14, 21, 10, 8, 15, 12, 3],
            "Dependencies (IDs)": ["", "1", "2", "3", "3", "4,5", "3", "6,7"],
            "Risk (0-5)": [1.0, 3.0, 2.5, 1.5, 1.5, 1.0, 2.0, 0.5],
            "Parent ID": [""] * 8
        })

    @staticmethod
    def _marketing_campaign():
        """Marketing campaign template"""
        return pd.DataFrame({
            "Select": [False] * 7,
            "ID": [1, 2, 3, 4, 5, 6, 7],
            "Task": [
                "Market Research",
                "Strategy Development",
                "Creative Design",
                "Content Creation",
                "Media Planning",
                "Campaign Launch",
                "Performance Analysis"
            ],
            "Duration (days)": [10, 7, 14, 12, 5, 3, 7],
            "Dependencies (IDs)": ["", "1", "2", "2", "2", "3,4,5", "6"],
            "Risk (0-5)": [1.0, 0.5, 2.0, 1.5, 1.0, 3.0, 0.5],
            "Parent ID": [""] * 7
        })

    @staticmethod
    def _product_launch():
        """Product launch template"""
        return pd.DataFrame({
            "Select": [False] * 9,
            "ID": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "Task": [
                "Product Development",
                "Quality Testing",
                "Manufacturing Setup",
                "Marketing Materials",
                "Sales Training",
                "Distribution Setup",
                "Pre-launch Marketing",
                "Product Launch",
                "Post-launch Support"
            ],
            "Duration (days)": [45, 14, 21, 10, 7, 14, 10, 3, 30],
            "Dependencies (IDs)": ["", "1", "1", "1", "1", "2", "4", "3,5,6,7", "8"],
            "Risk (0-5)": [3.0, 2.0, 2.5, 1.0, 0.5, 1.5, 1.0, 2.5, 1.0],
            "Parent ID": [""] * 9
        })

    @staticmethod
    def _research_development():
        """R&D project template"""
        return pd.DataFrame({
            "Select": [False] * 6,
            "ID": [1, 2, 3, 4, 5, 6],
            "Task": [
                "Literature Review",
                "Hypothesis Formation",
                "Experimental Design",
                "Data Collection",
                "Data Analysis",
                "Report Writing"
            ],
            "Duration (days)": [21, 7, 14, 35, 21, 14],
            "Dependencies (IDs)": ["", "1", "2", "3", "4", "5"],
            "Risk (0-5)": [0.5, 1.0, 1.5, 4.0, 3.0, 1.0],
            "Parent ID": [""] * 6
        })

    @staticmethod
    def _event_planning():
        """Event planning template"""
        return pd.DataFrame({
            "Select": [False] * 8,
            "ID": [1, 2, 3, 4, 5, 6, 7, 8],
            "Task": [
                "Venue Booking",
                "Catering Arrangements",
                "Speaker Coordination",
                "Marketing & Invitations",
                "Equipment Setup",
                "Registration System",
                "Event Execution",
                "Post-event Cleanup"
            ],
            "Duration (days)": [3, 5, 10, 14, 2, 7, 1, 1],
            "Dependencies (IDs)": ["", "1", "1", "1", "1", "1", "2,3,4,5,6", "7"],
            "Risk (0-5)": [2.0, 1.5, 3.0, 1.0, 1.0, 0.5, 2.5, 0.5],
            "Parent ID": [""] * 8
        })

    @staticmethod
    def _manufacturing_setup():
        """Manufacturing setup template"""
        return pd.DataFrame({
            "Select": [False] * 7,
            "ID": [1, 2, 3, 4, 5, 6, 7],
            "Task": [
                "Equipment Procurement",
                "Facility Preparation",
                "Equipment Installation",
                "Safety Systems Setup",
                "Staff Training",
                "Quality Control Setup",
                "Production Testing"
            ],
            "Duration (days)": [30, 14, 10, 7, 14, 7, 5],
            "Dependencies (IDs)": ["", "", "1,2", "3", "3", "3", "4,5,6"],
            "Risk (0-5)": [2.5, 1.5, 2.0, 1.0, 1.0, 1.5, 2.0],
            "Parent ID": [""] * 7
        })

    @staticmethod
    def _website_redesign():
        """Website redesign template"""
        return pd.DataFrame({
            "Select": [False] * 8,
            "ID": [1, 2, 3, 4, 5, 6, 7, 8],
            "Task": [
                "Requirements Gathering",
                "UX/UI Design",
                "Content Strategy",
                "Frontend Development",
                "Backend Development",
                "Content Migration",
                "Testing & QA",
                "Launch & Deployment"
            ],
            "Duration (days)": [7, 14, 10, 21, 18, 5, 7, 3],
            "Dependencies (IDs)": ["", "1", "1", "2", "1", "3", "4,5,6", "7"],
            "Risk (0-5)": [0.5, 1.5, 1.0, 2.0, 2.5, 1.0, 1.5, 2.0],
            "Parent ID": [""] * 8
        })


# Updated view.py functions to include template selector
def render_template_selector(model):
    """Render template selection UI"""
    st.subheader("📋 Project Templates")

    # Template selector
    templates = ProjectTemplates.get_available_templates()
    selected_template = st.selectbox(
        "Choose a project template:",
        templates,
        help="Select a predefined project template to get started quickly"
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🔄 Load Template", type="primary"):
            return selected_template, True

    with col2:
        if st.button("👁️ Preview Template"):
            return selected_template, "preview"

    with col3:
        st.info("💡 Templates include realistic task durations, dependencies, and risk levels")

    return selected_template, False


def render_template_preview(template_name):
    """Show preview of selected template"""
    template_df = ProjectTemplates.get_template(template_name)

    st.subheader(f"📊 Preview: {template_name}")

    # Calculate some basic metrics
    total_duration = template_df["Duration (days)"].sum()
    avg_risk = template_df["Risk (0-5)"].mean()
    num_tasks = len(template_df)

    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tasks", num_tasks)
    with col2:
        st.metric("Total Duration", f"{total_duration} days")
    with col3:
        st.metric("Average Risk", f"{avg_risk:.1f}/5")

    # Show the data
    st.dataframe(
        template_df[["ID", "Task", "Duration (days)", "Dependencies (IDs)", "Risk (0-5)"]],
        use_container_width=True,
        hide_index=True
    )

    return template_df


def render_editor_tab_with_templates(model):
    """Enhanced editor with template support"""
    st.subheader("Project Editor")

    # Template section
    template_name, load_action = render_template_selector(model)

    if load_action == "preview":
        render_template_preview(template_name)
        st.divider()
    elif load_action is True:
        # Load the selected template
        new_df = ProjectTemplates.get_template(template_name)
        model.task_df = new_df
        model.save_tasks()
        st.success(f"✅ Loaded template: {template_name}")
        st.rerun()

    # Rest of the editor (existing functionality)
    st.info("Edit tasks below or import an .mpp file. Use **Task IDs** in Dependencies (e.g., `1,2`).")

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