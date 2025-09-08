# /Users/jose/PDE_Project/MVC/tests/test_model.py
# Replace entire file
import pytest
import numpy as np
import pandas as pd
from model import TaskModel
from unittest.mock import Mock

@pytest.fixture
def task_model():
    model = TaskModel()
    return model

def test_load_tasks(task_model):
    df = task_model.task_df
    assert len(df) == 6
    assert list(df.columns) == ["Select", "ID", "Task", "Duration (days)", "Dependencies (IDs)", "Risk (0-5)", "Parent ID"]
    assert df["Task"].tolist() == [
        "Requirements", "Database Design", "Backend API",
        "Third-party Integration", "Frontend UI", "Testing & Deployment"
    ]
    assert np.array_equal(df["Duration (days)"].to_numpy(), np.array([14, 21, 28, 21, 35, 14], dtype=float))
    assert np.array_equal(df["Risk (0-5)"].to_numpy(), np.array([0, 0, 1, 0, 2, 0], dtype=float))
    assert df["Dependencies (IDs)"].tolist() == ["", "1", "2", "3", "2", "4,5"]
    assert df["Parent ID"].tolist() == [""] * 6

def test_validate_tasks_valid(task_model):
    valid, errors = task_model.validate_tasks()
    assert valid
    assert len(errors) == 0

def test_validate_tasks_invalid_duration(task_model):
    task_model.task_df.loc[0, "Duration (days)"] = 0
    valid, errors = task_model.validate_tasks()
    assert not valid
    assert any("Invalid duration for task 1" in e for e in errors)

def test_validate_tasks_invalid_dependency(task_model):
    task_model.task_df.loc[0, "Dependencies (IDs)"] = "7"
    valid, errors = task_model.validate_tasks()
    assert not valid
    assert any("Invalid dependency for task 1: '7'" in e for e in errors)

    task_model.task_df.loc[0, "Dependencies (IDs)"] = "1"
    valid, errors = task_model.validate_tasks()
    assert not valid
    assert any("Invalid dependency for task 1: '1'" in e for e in errors)

def test_build_adjacency(task_model):
    adjacency, error = task_model.build_adjacency()
    expected = np.zeros((6, 6))
    expected[0, 1] = 1
    expected[1, 2] = 1
    expected[2, 3] = 1
    expected[1, 4] = 1
    expected[3, 5] = 1
    expected[4, 5] = 1
    assert np.array_equal(adjacency, expected)
    assert error is None

def test_run_pde_simulation(task_model):
    success, error = task_model.run_pde_simulation(diffusion=0.001, reaction_multiplier=2.0, max_delay=0.05)
    assert success
    assert error is None
    assert task_model.simulation_data["u_matrix"].shape[0] == 6
    assert task_model.simulation_data["risk_curve"].shape[0] > 0
    assert len(task_model.simulation_data["simulation_time"]) == len(task_model.simulation_data["risk_curve"])

def test_run_classical_simulation(task_model):
    task_model.run_pde_simulation(diffusion=0.001, reaction_multiplier=2.0, max_delay=0.05)
    success, error = task_model.run_classical_simulation()
    assert success
    assert error is None
    assert task_model.simulation_data["classical_risk"].shape[0] > 0
    assert len(task_model.simulation_data["simulation_time"]) == len(task_model.simulation_data["classical_risk"])

def test_import_mpp_no_mppx(task_model, monkeypatch):
    def mock_import(name, *args, **kwargs):
        raise ImportError("No module named 'mppx'")
    monkeypatch.setattr("builtins.__import__", mock_import)
    success, error = task_model.import_mpp("dummy.mpp")
    assert not success
    assert "MPP import requires 'python-mppx' library" in error

def test_save_project(task_model, tmp_path, monkeypatch):
    task_model.run_pde_simulation(diffusion=0.001, reaction_multiplier=2.0, max_delay=0.05)
    task_model.run_classical_simulation()
    save_path = tmp_path / "project_data.json"
    monkeypatch.setattr("model.save_task_list", lambda x: None)  # Mock autosave
    task_model.save_project(
        task_model.simulation_data["u_matrix"],
        task_model.simulation_data["risk_curve"],
        task_model.simulation_data["classical_risk"],
        save_path=save_path
    )
    assert os.path.exists(save_path)

def test_export_mpp(task_model):
    task_model.run_pde_simulation(diffusion=0.001, reaction_multiplier=2.0, max_delay=0.05)
    result, error = task_model.export_mpp()
    assert result is not None
    assert error is None
    assert isinstance(result, str)

def test_validate_tasks_invalid_parent(task_model):
    task_model.task_df.loc[0, "Parent ID"] = "7"  # Invalid Parent ID
    valid, errors = task_model.validate_tasks()
    assert not valid
    assert any("Invalid Parent ID for task 1: '7'" in e for e in errors)

    task_model.task_df.loc[0, "Parent ID"] = "1"  # Task cannot be its own parent
    valid, errors = task_model.validate_tasks()
    assert not valid
    assert any("Invalid Parent ID for task 1: '1'" in e for e in errors)

    task_model.task_df.loc[1, "Parent ID"] = "2"  # Task 2 is a parent
    task_model.task_df.loc[0, "Parent ID"] = "2"
    valid, errors = task_model.validate_tasks()
    assert not valid
    assert any("Task 2 cannot be a parent and have a Parent ID" in e for e in errors)w