# ============================================================
# GAUGES — PPM Framework
# Kreisdiagramme (Plotly) für Fortschritt, Restzeit und
# Risiko-Konfidenz. Für hellen Streamlit-Hintergrund gestylt
# (klar lesbar, keine dunklen Flächen).
#
# Importiert in: src/app_v2.py
# ============================================================

import plotly.graph_objects as go

# ── Textfarben, abgestimmt auf hellen Hintergrund ─────────────
TEXT_DARK  = "#262730"   # Standard-Textfarbe von Streamlit (hell)
TEXT_MUTED = "#6c6f7c"
GRID_LINE  = "#e6e6e6"

COLOR_NORMAL    = "#2e7d32"
COLOR_BEOBACHT  = "#1565c0"
COLOR_WARNUNG   = "#e65100"
COLOR_KRITISCH  = "#c62828"

RISK_COLORS = {
    0: COLOR_NORMAL,
    1: COLOR_BEOBACHT,
    2: COLOR_WARNUNG,
    3: COLOR_KRITISCH,
}


def _base_gauge(value, value_max, color, suffix="", steps=None,
                 number_format=".0f"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            "suffix": suffix,
            "valueformat": number_format,
            "font": {"size": 28, "color": TEXT_DARK},
        },
        gauge={
            "axis": {
                "range": [0, value_max],
                "tickcolor": TEXT_MUTED,
                "tickfont": {"size": 9, "color": TEXT_MUTED},
            },
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


def gauge_progress(pct_complete: float):
    """Bearbeitungsfortschritt 0-100 %."""
    return _base_gauge(
        value=pct_complete * 100,
        value_max=100,
        color="#1565c0",
        suffix=" %",
        steps=[{"range": [0, 100], "color": "#eaf1fb"}],
    )


def gauge_remaining_time(remaining_h: float, elapsed_h: float):
    """
    Verbleibende Bearbeitungszeit in Stunden. Die Skala wird relativ
    zum Ticket selbst gesetzt (vergangene + verbleibende Zeit),
    damit der Balken bei sehr kurzen wie bei sehr langen Restzeiten
    stets sichtbar und proportional bleibt — statt an einem fixen
    Datensatz-Referenzwert (z. B. P90), der bei kurzen Restzeiten
    einen unsichtbaren Balken erzeugt.
    """
    vmax = max(elapsed_h + remaining_h, remaining_h * 1.2, 1)
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
    """Konfidenz der vorhergesagten Risikoklasse (0-100 %)."""
    color = RISK_COLORS.get(klasse, "#1565c0")
    return _base_gauge(
        value=proba_top * 100,
        value_max=100,
        color=color,
        suffix=" %",
        steps=[
            {"range": [0, 40],  "color": "#eaf6ec"},
            {"range": [40, 70], "color": "#fdf1e0"},
            {"range": [70, 100], "color": "#fbe9e9"},
        ],
    )
