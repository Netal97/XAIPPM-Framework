"""Plotly-Gauges für das leakage-freie LSTM-PPM-Dashboard."""
from __future__ import annotations

import plotly.graph_objects as go

TEXT_DARK = "#262730"
TEXT_MUTED = "#6c6f7c"
GRID_LINE = "#e6e6e6"

RISK_COLORS = {
    0: "#2e7d32",
    1: "#1565c0",
    2: "#e65100",
    3: "#c62828",
}


def _base_gauge(value, value_max, color, suffix="", steps=None, number_format=".0f"):
    value_max = max(float(value_max), 1.0)
    value = min(max(float(value), 0.0), value_max)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "valueformat": number_format, "font": {"size": 28, "color": TEXT_DARK}},
        gauge={
            "axis": {"range": [0, value_max], "tickcolor": TEXT_MUTED, "tickfont": {"size": 9, "color": TEXT_MUTED}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "bordercolor": GRID_LINE,
            "borderwidth": 1,
            "steps": steps or [],
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": TEXT_DARK},
    )
    return fig




def gauge_remaining_time(remaining_h: float, elapsed_h: float):
    vmax = max(float(elapsed_h) + float(remaining_h), float(remaining_h) * 1.2, 1)
    return _base_gauge(
        value=remaining_h,
        value_max=vmax,
        color="#1565c0",
        suffix=" h",
        steps=[
            {"range": [0, vmax * 0.5], "color": "#eaf6ec"},
            {"range": [vmax * 0.5, vmax * 0.8], "color": "#fdf1e0"},
            {"range": [vmax * 0.8, vmax], "color": "#fbe9e9"},
        ],
    )


def gauge_risk(proba_top: float, klasse: int):
    return _base_gauge(
        value=float(proba_top) * 100,
        value_max=100,
        color=RISK_COLORS.get(int(klasse), "#1565c0"),
        suffix=" %",
        steps=[
            {"range": [0, 40], "color": "#eaf6ec"},
            {"range": [40, 70], "color": "#fdf1e0"},
            {"range": [70, 100], "color": "#fbe9e9"},
        ],
    )
