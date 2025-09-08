import pandas as pd
import json
import os


def save_task_list(task_df):
    """Save task DataFrame to JSON file"""
    save_path = "task_list.json"
    task_dict = task_df.to_dict(orient="records")
    with open(save_path, "w") as f:
        json.dump(task_dict, f, indent=2)


def load_task_list():
    """Load task DataFrame from JSON file"""
    save_path = "task_list.json"
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            task_dict = json.load(f)
            df = pd.DataFrame(task_dict)
            # Ensure Parent ID column exists
            if "Parent ID" not in df.columns:
                df["Parent ID"] = [""] * len(df)
            return df
    return None


def validate_project_structure(task_df):
    """Validate the project structure and dependencies"""
    errors = []

    # Check for required columns
    required_columns = ["ID", "Task", "Duration (days)", "Dependencies (IDs)", "Risk (0-5)"]
    for col in required_columns:
        if col not in task_df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return False, errors

    # Check for valid task IDs (sequential starting from 1)
    expected_ids = list(range(1, len(task_df) + 1))
    actual_ids = task_df["ID"].tolist()
    if actual_ids != expected_ids:
        errors.append("Task IDs must be sequential starting from 1")

    # Check for valid dependencies
    valid_ids = set(task_df["ID"])
    for idx, row in task_df.iterrows():
        deps = row["Dependencies (IDs)"]
        if deps and str(deps).strip():
            dep_list = str(deps).split(",")
            for dep in dep_list:
                dep = dep.strip()
                if dep.isdigit():
                    dep_id = int(dep)
                    if dep_id not in valid_ids:
                        errors.append(f"Task {row['ID']}: Invalid dependency {dep_id}")
                    if dep_id == row["ID"]:
                        errors.append(f"Task {row['ID']}: Cannot depend on itself")
                else:
                    errors.append(f"Task {row['ID']}: Invalid dependency format '{dep}'")

    # Check for valid durations
    for idx, row in task_df.iterrows():
        if row["Duration (days)"] <= 0:
            errors.append(f"Task {row['ID']}: Duration must be positive")

    # Check for valid risk levels
    for idx, row in task_df.iterrows():
        risk = row["Risk (0-5)"]
        if not (0 <= risk <= 5):
            errors.append(f"Task {row['ID']}: Risk must be between 0-5")

    return len(errors) == 0, errors


def export_to_csv(task_df, filename="project_tasks.csv"):
    """Export task DataFrame to CSV"""
    try:
        task_df.to_csv(filename, index=False)
        return True, f"Successfully exported to {filename}"
    except Exception as e:
        return False, f"Error exporting to CSV: {str(e)}"


def import_from_csv(filename):
    """Import task DataFrame from CSV"""
    try:
        if not os.path.exists(filename):
            return None, f"File {filename} not found"

        df = pd.read_csv(filename)

        # Ensure required columns exist
        required_columns = ["ID", "Task", "Duration (days)", "Dependencies (IDs)", "Risk (0-5)"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return None, f"Missing required columns: {missing_columns}"

        # Ensure Parent ID column exists
        if "Parent ID" not in df.columns:
            df["Parent ID"] = [""] * len(df)

        # Ensure Select column exists
        if "Select" not in df.columns:
            df["Select"] = [False] * len(df)

        # Validate the imported data
        valid, errors = validate_project_structure(df)
        if not valid:
            return None, f"Invalid project structure: {'; '.join(errors)}"

        return df, None
    except Exception as e:
        return None, f"Error importing from CSV: {str(e)}"


def backup_project(task_df, backup_filename=None):
    """Create a backup of the current project"""
    if backup_filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"project_backup_{timestamp}.json"

    try:
        backup_data = {
            "timestamp": datetime.now().isoformat(),
            "tasks": task_df.to_dict(orient="records")
        }

        with open(backup_filename, "w") as f:
            json.dump(backup_data, f, indent=2)

        return True, f"Backup created: {backup_filename}"
    except Exception as e:
        return False, f"Error creating backup: {str(e)}"


def restore_from_backup(backup_filename):
    """Restore project from backup file"""
    try:
        if not os.path.exists(backup_filename):
            return None, f"Backup file {backup_filename} not found"

        with open(backup_filename, "r") as f:
            backup_data = json.load(f)

        if "tasks" not in backup_data:
            return None, "Invalid backup file format"

        df = pd.DataFrame(backup_data["tasks"])

        # Ensure all required columns exist
        if "Parent ID" not in df.columns:
            df["Parent ID"] = [""] * len(df)
        if "Select" not in df.columns:
            df["Select"] = [False] * len(df)

        return df, None
    except Exception as e:
        return None, f"Error restoring from backup: {str(e)}"


def get_project_statistics(task_df):
    """Get basic project statistics"""
    try:
        stats = {
            "total_tasks": len(task_df),
            "total_duration": task_df["Duration (days)"].sum(),
            "average_duration": task_df["Duration (days)"].mean(),
            "max_duration": task_df["Duration (days)"].max(),
            "min_duration": task_df["Duration (days)"].min(),
            "average_risk": task_df["Risk (0-5)"].mean(),
            "max_risk": task_df["Risk (0-5)"].max(),
            "high_risk_tasks": len(task_df[task_df["Risk (0-5)"] >= 3]),
            "tasks_with_dependencies": len(task_df[task_df["Dependencies (IDs)"].str.len() > 0])
        }
        return stats
    except Exception as e:
        return {"error": str(e)}


def clean_project_data(task_df):
    """Clean and standardize project data"""
    df_clean = task_df.copy()

    # Clean dependencies column - remove extra spaces
    df_clean["Dependencies (IDs)"] = df_clean["Dependencies (IDs)"].astype(str)
    df_clean["Dependencies (IDs)"] = df_clean["Dependencies (IDs)"].str.replace(" ", "")
    df_clean["Dependencies (IDs)"] = df_clean["Dependencies (IDs)"].str.replace("nan", "")

    # Clean Parent ID column
    df_clean["Parent ID"] = df_clean["Parent ID"].astype(str)
    df_clean["Parent ID"] = df_clean["Parent ID"].str.replace("nan", "")

    # Ensure numeric columns are proper numeric types
    df_clean["Duration (days)"] = pd.to_numeric(df_clean["Duration (days)"], errors='coerce')
    df_clean["Risk (0-5)"] = pd.to_numeric(df_clean["Risk (0-5)"], errors='coerce')

    # Fill NaN values with defaults
    df_clean["Duration (days)"].fillna(7.0, inplace=True)
    df_clean["Risk (0-5)"].fillna(0.0, inplace=True)
    df_clean["Dependencies (IDs)"].fillna("", inplace=True)
    df_clean["Parent ID"].fillna("", inplace=True)

    return df_clean