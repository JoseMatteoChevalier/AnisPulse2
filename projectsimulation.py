# ========================================
# Standalone Python Project Simulation
# ========================================

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# ----------------------------
# Project Setup
# ----------------------------
tasks = ["Requirements", "Database Design", "Backend API",
         "Third-party Integration", "Frontend UI", "Testing & Deployment"]
num_tasks = len(tasks)
baseline_durations = np.array([2, 3, 4, 3, 5, 2])  # in weeks

# Dependencies (adjacency matrix)
adjacency = np.array([
    [0,1,0,0,0,0],
    [0,0,1,0,1,0],
    [0,0,0,1,0,0],
    [0,0,0,0,0,1],
    [0,0,0,0,0,1],
    [0,0,0,0,0,0]
], dtype=float)

T = 20  # total simulation time in weeks
dt = 0.05
steps = int(T/dt)
time = np.linspace(0, T, steps+1)
diffusion = 0.01

# Risk levels (simulate some delays)
risk_levels = np.array([0, 0, 1, 0, 2, 0])  # e.g., task 3 and 5 are risky

# ----------------------------
# PDE Simulation Function
# ----------------------------
def reaction(u_i, duration):
    return 1.0 / duration

def run_pde(adjacency, durations, diffusion, risk_levels):
    u = np.zeros((num_tasks, steps+1))
    for t in range(steps):
        du = np.zeros(num_tasks)
        for i in range(num_tasks):
            preds = np.where(adjacency[:, i] > 0)[0]
            if len(preds) == 0 or all(u[p, t] >= 1.0 for p in preds):
                dur = durations[i] * max(1.0, risk_levels[i])
                du[i] += reaction(u[i, t], dur)
                for j in preds:
                    du[i] += adjacency[j, i] * (u[j, t] - u[i, t]) * diffusion
        u[:, t+1] = u[:, t] + du * dt
        u[:, t+1] = np.clip(u[:, t+1], 0, 1)
    return u

# ----------------------------
# Compute Curves
# ----------------------------
u_baseline = run_pde(adjacency, baseline_durations, diffusion, np.zeros(num_tasks))
baseline_curve = u_baseline.mean(axis=0)

u_risk = run_pde(adjacency, baseline_durations, diffusion, risk_levels)
risk_curve = u_risk.mean(axis=0)

# ----------------------------
# Create output folder
# ----------------------------
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# 3D Plot: Baseline vs Risk
# ----------------------------
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection="3d")

ax.plot(time, np.zeros_like(time), baseline_curve, color="blue", lw=2, label="Baseline")
ax.plot(time, np.ones_like(time), risk_curve, color="red", lw=2, linestyle="--", label="With Risk")

ax.set_xlabel("Time (weeks)")
ax.set_ylabel("Scenario")
ax.set_yticks([0,1])
ax.set_yticklabels(["Baseline","Risk"])
ax.set_zlabel("Completion (0–1)")
ax.set_title("Project Completion: Baseline vs Risk (3D)")
ax.view_init(elev=25, azim=-60)
ax.legend()

fig_path = os.path.join(output_folder, "3d_completion.png")
plt.savefig(fig_path)
print(f"Saved 3D completion plot to: {fig_path}")
plt.close()

# ----------------------------
# 2D Gantt-like Plot
# ----------------------------
fig2, ax2 = plt.subplots(figsize=(10,6))
y_pos = np.arange(num_tasks)
bar_height = 0.4

# Compute simple start/finish times
start_baseline = np.zeros(num_tasks)
finish_baseline = baseline_durations.copy()
for i in range(1, num_tasks):
    preds = np.where(adjacency[:, i] > 0)[0]
    if len(preds) > 0:
        start_baseline[i] = max(finish_baseline[preds])
    finish_baseline[i] = start_baseline[i] + baseline_durations[i]

start_risk = np.zeros(num_tasks)
finish_risk = baseline_durations * np.maximum(1, risk_levels)
for i in range(1, num_tasks):
    preds = np.where(adjacency[:, i] > 0)[0]
    if len(preds) > 0:
        start_risk[i] = max(finish_risk[preds])
    finish_risk[i] = start_risk[i] + baseline_durations[i]*np.maximum(1,risk_levels[i])

# Plot baseline
ax2.barh(y_pos - bar_height/2, finish_baseline-start_baseline, left=start_baseline,
         color="skyblue", edgecolor="black", height=bar_height, label="Baseline")
# Plot risk
ax2.barh(y_pos + bar_height/2, finish_risk-start_risk, left=start_risk,
         color="salmon", edgecolor="black", height=bar_height, label="With Risk")

# Add task names
for i, task in enumerate(tasks):
    ax2.text(finish_risk[i]+0.1, y_pos[i]+bar_height/2, task, va='center')

ax2.set_xlabel("Time (weeks)")
ax2.set_ylabel("Tasks")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(tasks)
ax2.set_title("Gantt-like Project Timeline: Baseline vs Risk")
ax2.legend()

gantt_path = os.path.join(output_folder, "gantt_comparison.png")
plt.savefig(gantt_path)
print(f"Saved Gantt plot to: {gantt_path}")
plt.close()

# ----------------------------
# Completion Times
# ----------------------------
baseline_completion = T * np.argmax(baseline_curve>=1)/steps
risk_completion = T * np.argmax(risk_curve>=1)/steps

print(f"Baseline completion time: {baseline_completion:.2f} weeks")
print(f"Risk-weighted completion time: {risk_completion:.2f} weeks")

# !/usr/bin/env python3
"""
Simple standalone script to compare matplotlib vs plotly
Run this independently to see the difference
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Generate some sample project data
np.random.seed(42)
time_days = np.arange(0, 100, 1)
classical_completion = np.minimum(time_days / 80, 1.0)  # Linear completion
pde_completion = 1 - np.exp(-time_days / 30)  # Exponential approach to 1
monte_carlo_results = np.random.gamma(2, 40, 1000)  # Project completion times

print("=== MATPLOTLIB EXAMPLES (what you currently use) ===")

# Example 1: Line plot (like your simulation results)
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(time_days, classical_completion, 'b-', label='Classical', linewidth=2)
plt.plot(time_days, pde_completion, 'r--', label='PDE', linewidth=2)
plt.xlabel('Time (days)')
plt.ylabel('Completion %')
plt.title('Matplotlib: Project Completion')
plt.legend()
plt.grid(True)

# Example 2: Histogram (Monte Carlo results)
plt.subplot(2, 2, 2)
plt.hist(monte_carlo_results, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(np.mean(monte_carlo_results), color='red', linestyle='--',
            label=f'Mean: {np.mean(monte_carlo_results):.1f}')
plt.xlabel('Project Duration (days)')
plt.ylabel('Frequency')
plt.title('Matplotlib: Monte Carlo Results')
plt.legend()

# Example 3: Simple Gantt chart (like yours)
tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4']
start_times = [0, 10, 20, 35]
durations = [15, 20, 25, 10]

plt.subplot(2, 2, 3)
for i, (task, start, duration) in enumerate(zip(tasks, start_times, durations)):
    plt.barh(i, duration, left=start, height=0.6,
             alpha=0.8, color=plt.cm.Set3(i / len(tasks)))
    plt.text(start + duration / 2, i, f'{task}\n({duration}d)',
             ha='center', va='center', fontsize=8)

plt.yticks(range(len(tasks)), tasks)
plt.xlabel('Time (days)')
plt.title('Matplotlib: Gantt Chart')
plt.grid(True, alpha=0.3)

# Example 4: 3D plot (like your comparison)
ax = plt.subplot(2, 2, 4, projection='3d')
ax.plot(time_days, [0] * len(time_days), classical_completion, 'b-', label='Classical', linewidth=2)
ax.plot(time_days, [1] * len(time_days), pde_completion, 'r-', label='PDE', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('Model')
ax.set_zlabel('Completion')
ax.set_title('Matplotlib: 3D Comparison')
ax.legend()

plt.tight_layout()
plt.show()

print("\n=== PLOTLY EXAMPLES (interactive alternative) ===")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    # Example 1: Interactive line plot
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=time_days, y=classical_completion,
                              mode='lines', name='Classical',
                              hovertemplate='Day %{x}<br>Completion: %{y:.1%}<extra></extra>'))
    fig1.add_trace(go.Scatter(x=time_days, y=pde_completion,
                              mode='lines', name='PDE', line=dict(dash='dash'),
                              hovertemplate='Day %{x}<br>Completion: %{y:.1%}<extra></extra>'))
    fig1.update_layout(title='Plotly: Interactive Project Completion',
                       xaxis_title='Time (days)', yaxis_title='Completion %')
    fig1.show()

    # Example 2: Interactive histogram with statistics
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=monte_carlo_results, nbinsx=30, name='Results',
                                hovertemplate='Duration: %{x:.1f}<br>Count: %{y}<extra></extra>'))

    # Add mean line
    mean_val = np.mean(monte_carlo_results)
    fig2.add_vline(x=mean_val, line_dash="dash", line_color="red",
                   annotation_text=f"Mean: {mean_val:.1f} days")

    # Add percentiles
    p95 = np.percentile(monte_carlo_results, 95)
    fig2.add_vline(x=p95, line_dash="dot", line_color="orange",
                   annotation_text=f"95%: {p95:.1f} days")

    fig2.update_layout(title='Plotly: Interactive Monte Carlo Results',
                       xaxis_title='Project Duration (days)', yaxis_title='Frequency')
    fig2.show()

    # Example 3: Interactive Gantt-style chart
    fig3 = go.Figure()

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for i, (task, start, duration) in enumerate(zip(tasks, start_times, durations)):
        fig3.add_trace(go.Scatter(
            x=[start, start + duration, start + duration, start, start],
            y=[i - 0.3, i - 0.3, i + 0.3, i + 0.3, i - 0.3],
            fill='toself',
            fillcolor=colors[i],
            line=dict(color=colors[i]),
            name=task,
            hovertemplate=f'{task}<br>Start: Day %{{x}}<br>Duration: {duration} days<extra></extra>'
        ))

    fig3.update_layout(
        title='Plotly: Interactive Gantt Chart',
        xaxis_title='Time (days)',
        yaxis=dict(tickmode='array', tickvals=list(range(len(tasks))), ticktext=tasks),
        showlegend=True
    )
    fig3.show()

    # Example 4: Interactive 3D
    fig4 = go.Figure()

    fig4.add_trace(go.Scatter3d(
        x=time_days, y=[0] * len(time_days), z=classical_completion,
        mode='lines', name='Classical',
        line=dict(color='blue', width=6),
        hovertemplate='Day %{x}<br>Classical: %{z:.1%}<extra></extra>'
    ))

    fig4.add_trace(go.Scatter3d(
        x=time_days, y=[1] * len(time_days), z=pde_completion,
        mode='lines', name='PDE',
        line=dict(color='red', width=6),
        hovertemplate='Day %{x}<br>PDE: %{z:.1%}<extra></extra>'
    ))

    fig4.update_layout(
        title='Plotly: Interactive 3D Comparison',
        scene=dict(
            xaxis_title='Time (days)',
            yaxis_title='Model Type',
            zaxis_title='Completion %'
        )
    )
    fig4.show()

    print("✅ Plotly examples completed!")
    print("📝 Key differences:")
    print("   - Plotly: Interactive (zoom, pan, hover)")
    print("   - Plotly: Professional look")
    print("   - Plotly: Better statistics overlays")
    print("   - Matplotlib: Lighter, simpler, what you already use")

except ImportError:
    print("❌ Plotly not installed. Install with: pip install plotly")
    print("🔍 This shows why matplotlib might be better for your use case!")

print(f"\n=== SUMMARY ===")
print("Matplotlib pros: Lightweight, you already use it, simple")
print("Plotly pros: Interactive, professional, better for exploration")
print("For your Monte Carlo enhancement: Stick with matplotlib!")