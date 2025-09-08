import pytest
import pandas as pd
from model import TaskModel
from controller import EditorController, SimulationController

@pytest.fixture
def task_model():
    return TaskModel()

@pytest.fixture
def editor_controller(task_model):
    return EditorController(task_model)

@pytest.fixture
def simulation_controller(task_model):
    return SimulationController(task_model)

def test_handle_add_row(editor_controller):
    initial_len = len(editor_controller.model.task_df)
    editor_controller.handle_add_row()
    assert len(editor_controller.model.task_df) == initial_len + 1
    assert editor_controller.model.task_df.iloc[-1]["ID"] == initial_len + 1
    assert editor_controller.model.task_df.iloc[-1]["Task"] == ""

def test_handle_delete_rows(editor_controller):
    editor_controller.model.task_df.loc[0, "Select"] = True
    initial_len = len(editor_controller.model.task_df)
    editor_controller.handle_delete_rows()
    assert len(editor_controller.model.task_df) == initial_len - 1
    assert all(editor_controller.model.task_df["ID"] == range(1, initial_len))

def test_handle_task_edit(editor_controller):
    new_df = editor_controller.model.task_df.copy()
    new_df.loc[0, "Task"] = "New Task Name"
    editor_controller.handle_task_edit(new_df)
    assert editor_controller.model.task_df.loc[0, "Task"] == "New Task Name"

def test_handle_simulation_valid(simulation_controller):
    success = simulation_controller.handle_simulation(diffusion=0.001, reaction_multiplier=2.0, max_delay=0.05)
    assert success
    assert simulation_controller.model.simulation_data["u_matrix"] is not None
    assert simulation_controller.model.simulation_data["classical_risk"] is not None

def test_handle_simulation_invalid_duration(simulation_controller):
    simulation_controller.model.task_df.loc[0, "Duration (days)"] = 0
    success = simulation_controller.handle_simulation(diffusion=0.001, reaction_multiplier=2.0, max_delay=0.05)
    assert not success