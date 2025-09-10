import plotly.graph_objects as go
import streamlit as st
import numpy as np

st.title("Plotly Test")

# Simple test data
x = np.linspace(0, 100, 50)
y1 = x * 0.8  # Simulation of your classical risk
y2 = x * 0.9  # Simulation of your PDE risk

# Create plotly figure
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x, y=y1,
    mode='lines',
    name='Classical Risk',
    line=dict(color='#1976d2', width=2)
))

fig.add_trace(go.Scatter(
    x=x, y=y2,
    mode='lines',
    name='PDE Risk',
    line=dict(color='#d32f2f', width=2, dash='dash')
))

fig.update_layout(
    title="Plotly Test - Same Colors as Your Current Charts",
    xaxis_title="Time (days)",
    yaxis_title="Completion",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
st.success("Plotly is working! 🎉")