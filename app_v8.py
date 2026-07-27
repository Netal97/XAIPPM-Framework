# ============================================================
# PPM FRAMEWORK — STREAMLIT DASHBOARD v5
# Helpdesk Italia · LSTM · Integrated Gradients
#
# Start:
#   streamlit run src/app_v5.py
#
# Benötigt in data/cache/deep_learning_bpic14/:
#   selected_f1_model.keras
#   selected_f2_model.keras
#   numerical_scaler.joblib
#   activity_vocabulary.joblib
#   sequence_config.joblib
#
# Optional in data/cache/deep_learning_cross_dataset/:
#   calibrators_dl_helpdesk_italien.joblib
#
# Demo-Daten:
#   data/tickets_it_demo_sequences.csv (bevorzugt)
# oder tickets_it_demo.csv + data/cache/prefix_log_it.pkl
# ============================================================
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import hashlib

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras

from gauges import gauge_remaining_time, gauge_risk
from handlungsempfehlung import (
    RISIKO_BEZEICHNUNG,
    handlungsempfehlung_anzeigen,
    risiko_symbol,
)
from ppm_utils import (
    activity_occlusion_local,
    create_current_case_sequences,
    integrated_gradients_numeric,
    load_demo_sequences,
    predict_lstm,
    validate_feature_contract,
)

st.set_page_config(page_title="PPM Dashboard", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #e7ebf0;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e4e9f0;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 4px 16px rgba(25, 45, 80, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        color: #24324a;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #1261d6;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #e4e9f0;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 4px 14px rgba(25, 45, 80, 0.04);
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    h1, h2, h3 {
        color: #10244d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# ROBUSTE PROJEKTPFADE
#
# Streamlit kann aus unterschiedlichen Arbeitsordnern gestartet werden.
# Deshalb werden Datenpfade nicht mehr relativ zum aktuellen Terminal-
# Verzeichnis gebildet, sondern ausgehend von dieser app_v5.py gesucht.
# ------------------------------------------------------------------
APP_FILE = Path(__file__).resolve()


def _find_project_root() -> Path:
    candidates = [APP_FILE.parent, *APP_FILE.parents, Path.cwd().resolve()]
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "data").exists():
            return candidate

    # Typischer Aufbau .../src/src/app_v5.py -> Projektwurzel parents[2]
    return APP_FILE.parents[2] if len(APP_FILE.parents) > 2 else APP_FILE.parent


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DL_DIR = CACHE_DIR / "deep_learning_bpic14"
CROSS_DIR = CACHE_DIR / "deep_learning_cross_dataset"

ARBEITSSTART_STUNDE = 8
ARBEITSENDE_STUNDE = 16
ARBEITSTAGE_WOCHENTAGE = {0, 1, 2, 3, 4}
STUNDEN_PRO_ARBEITSTAG = 8

FEATURE_LABELS = {
    "prefix_len": "Anzahl bisheriger Ereignisse",
    "elapsed_h": "Vergangene Zeit (h)",
    "time_since_prev_h": "Zeit seit letztem Ereignis (h)",
    "dow_sin": "Wochentag — Sinus",
    "dow_cos": "Wochentag — Kosinus",
    "n_distinct_acts_so_far": "Verschiedene Aktivitäten",
    "gap_max_so_far": "Längste bisherige Pause (h)",
    "pace_ratio": "Tempo-Index",
    "events_per_hour": "Ereignisse pro Stunde",
    "activity_diversity": "Aktivitätsvielfalt",
    "gap_cv": "Wartezeit-Variationskoeffizient",
}

FEATURE_TOOLTIPS = {
    "prefix_len": "Zahl der aktuell beobachteten Prozessereignisse.",
    "elapsed_h": "Zeit seit Eröffnung des Tickets.",
    "time_since_prev_h": "Aktuelle Wartezeit seit dem letzten Ereignis.",
    "dow_sin": "Zyklische Darstellung des Wochentags.",
    "dow_cos": "Zweite Komponente der zyklischen Wochentagdarstellung.",
    "n_distinct_acts_so_far": "Zahl unterschiedlicher bisheriger Aktivitäten.",
    "gap_max_so_far": "Längste bisher beobachtete Unterbrechung.",
    "pace_ratio": "Aktuelle Wartezeit relativ zum bisherigen mittleren Abstand.",
    "events_per_hour": "Mittlere Ereignisrate des laufenden Tickets.",
    "activity_diversity": "Anteil unterschiedlicher Aktivitäten im Präfix.",
    "gap_cv": "Relative Schwankung der Ereignisabstände.",
}


FEATURE_FORMULAS = {
    "prefix_len": "Anzahl beobachteter Ereignisse",
    "elapsed_h": "Aktuelle Zeit − Ticketeröffnung",
    "time_since_prev_h": "Aktuelle Zeit − letztes Ereignis",
    "dow_sin": "sin(2π · Wochentag / 7)",
    "dow_cos": "cos(2π · Wochentag / 7)",
    "n_distinct_acts_so_far": "Anzahl unterschiedlicher Aktivitäten",
    "gap_max_so_far": "Maximum der bisherigen Zeitabstände",
    "pace_ratio": "Aktuelle Wartezeit / mittlere bisherige Wartezeit",
    "events_per_hour": "Ereignisse / vergangene Zeit",
    "activity_diversity": "Unterschiedliche Aktivitäten / Ereignisse",
    "gap_cv": "Standardabweichung der Wartezeiten / Mittelwert",
}


def naechster_arbeitszeitpunkt(dt: datetime) -> datetime:
    while True:
        if dt.weekday() not in ARBEITSTAGE_WOCHENTAGE:
            dt = (dt + timedelta(days=1)).replace(hour=ARBEITSSTART_STUNDE, minute=0, second=0, microsecond=0)
            continue
        if dt.hour < ARBEITSSTART_STUNDE:
            dt = dt.replace(hour=ARBEITSSTART_STUNDE, minute=0, second=0, microsecond=0)
        elif dt.hour >= ARBEITSENDE_STUNDE:
            dt = (dt + timedelta(days=1)).replace(hour=ARBEITSSTART_STUNDE, minute=0, second=0, microsecond=0)
            continue
        return dt


def addiere_arbeitsstunden(start: datetime, stunden: float) -> datetime:
    dt = naechster_arbeitszeitpunkt(start)
    rest = max(float(stunden), 0.0)
    while rest > 1e-9:
        feierabend = dt.replace(hour=ARBEITSENDE_STUNDE, minute=0, second=0, microsecond=0)
        verfuegbar = (feierabend - dt).total_seconds() / 3600
        if rest <= verfuegbar:
            return dt + timedelta(hours=rest)
        rest -= verfuegbar
        dt = naechster_arbeitszeitpunkt((dt + timedelta(days=1)).replace(hour=ARBEITSSTART_STUNDE, minute=0, second=0, microsecond=0))
    return dt


@st.cache_resource
def artefakte_laden():
    required = [
        DL_DIR / "selected_f1_model.keras",
        DL_DIR / "selected_f2_model.keras",
        DL_DIR / "numerical_scaler.joblib",
        DL_DIR / "activity_vocabulary.joblib",
        DL_DIR / "sequence_config.joblib",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        details = "\n".join(f"- {path.resolve()}" for path in missing)
        raise FileNotFoundError(
            "Fehlende Modellartefakte.\n\n"
            f"Erkannte Projektwurzel: {PROJECT_ROOT}\n"
            f"Erwarteter Modellordner: {DL_DIR}\n\n"
            "Bitte kopieren Sie die folgenden Dateien aus dem "
            "Deep-Learning-Trainingsoutput in diesen Ordner:\n"
            f"{details}"
        )

    f1_model = keras.models.load_model(required[0])
    f2_model = keras.models.load_model(required[1])
    scaler = joblib.load(required[2])
    vocabulary = joblib.load(required[3])
    config = joblib.load(required[4])

    features = validate_feature_contract(config["FEATURES"])
    corrector = None
    temperature = float(config.get("temperature", 1.0)) if config.get("use_temperature_scaling", False) else 1.0
    calibration_label = "BPIC14-Validierung"

    target_calibrator_path = CROSS_DIR / "calibrators_dl_helpdesk_italien.joblib"
    if target_calibrator_path.exists():
        target = joblib.load(target_calibrator_path)
        corrector = target.get("regression_bias_corrector")
        temperature = float(target.get("target_temperature", temperature))
        calibration_label = "Helpdesk-spezifische 20%-Nachkalibrierung"

    return f1_model, f2_model, scaler, vocabulary, config, features, corrector, temperature, calibration_label


@st.cache_data
def demodaten_laden():
    data = load_demo_sequences(DATA_DIR, CACHE_DIR)
    if "case_start" in data.columns:
        data["case_start"] = pd.to_datetime(data["case_start"], errors="coerce")
    return data


f1_model, f2_model, scaler, activity_to_index, seq_cfg, FEATURES, reg_corrector, TEMPERATURE, calibration_label = artefakte_laden()
N_CLASSES = int(seq_cfg["N_CLASSES"])
KLASSEN = seq_cfg["CLASS_NAMES"]
ACTIVITY_COL = seq_cfg["ACTIVITY_COL"]
MAX_LEN = int(seq_cfg["MAX_LEN"])
PAD_INDEX = int(seq_cfg["PAD_INDEX"])
OOV_INDEX = int(seq_cfg["OOV_INDEX"])
index_to_activity = {v: k for k, v in activity_to_index.items()}
index_to_activity[PAD_INDEX] = "<PAD>"
index_to_activity[OOV_INDEX] = "<OOV>"

prefix_history = demodaten_laden()


@st.cache_data(ttl=3600, show_spinner=False)
def alle_vorhersagen_berechnen(data_fingerprint: str):
    # Fingerprint dient als Cache-Schlüssel; Daten werden reproduzierbar neu geladen.
    history = demodaten_laden()
    bundle = create_current_case_sequences(
        history, FEATURES, scaler, activity_to_index, ACTIVITY_COL,
        MAX_LEN, OOV_INDEX, PAD_INDEX,
    )
    remaining, probabilities, classes, prediction_diagnostics = predict_lstm(
        f1_model,
        f2_model,
        bundle,
        regression_corrector=reg_corrector,
        temperature=TEMPERATURE,
        calibration_mode="auto",
        return_diagnostics=True,
    )
    return (
        bundle,
        remaining,
        probabilities,
        classes,
        prediction_diagnostics,
    )


def daten_fingerprint(df: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(
        df[[c for c in ["IncidentID", "prefix_len", "time_since_prev_h"] if c in df.columns]],
        index=True,
    ).values.tobytes()
    return hashlib.sha256(payload).hexdigest()


bundle, restzeiten, probas, klassen, prediction_diagnostics = (
    alle_vorhersagen_berechnen(
        daten_fingerprint(prefix_history)
    )
)
df_tickets = bundle.current_rows.copy()
df_tickets["pred_restzeit"] = restzeiten
df_tickets["pred_restzeit_raw"] = prediction_diagnostics["remaining_raw"]
df_tickets["pred_klasse"] = klassen
for i in range(N_CLASSES):
    df_tickets[f"prob_{i}"] = probas[:, i]

if "wartezeit_h" not in df_tickets.columns:
    df_tickets["wartezeit_h"] = df_tickets.get("time_since_prev_h", 0.0)

df_tickets = df_tickets.sort_values(["pred_klasse", "pred_restzeit"], ascending=[False, True]).reset_index(drop=True)


with st.sidebar.expander("🔧 Prognose-Diagnose", expanded=False):
    st.write(
        f"Unterschiedliche F1-Rohprognosen: "
        f"**{prediction_diagnostics['raw_unique_predictions']}**"
    )
    st.write(
        f"Unterschiedliche angezeigte Prognosen: "
        f"**{prediction_diagnostics['final_unique_predictions']}**"
    )
    st.write(
        f"Standardabweichung roh: "
        f"**{prediction_diagnostics['raw_std_h']:.2f} h**"
    )
    st.write(
        f"Standardabweichung final: "
        f"**{prediction_diagnostics['final_std_h']:.2f} h**"
    )
    if np.isfinite(
        prediction_diagnostics["calibration_in_range_share"]
    ):
        st.write(
            "Anteil innerhalb des Kalibrierungsbereichs: "
            f"**{prediction_diagnostics['calibration_in_range_share']:.1%}**"
        )
    st.caption(
        prediction_diagnostics["calibration_reason"]
    )

if prediction_diagnostics["final_unique_predictions"] <= 1 and len(df_tickets) > 1:
    st.warning(
        "Die F1-Prognosen sind weiterhin konstant. "
        "Prüfen Sie im Diagnosebereich, ob bereits die "
        "Zero-Shot-Rohprognosen konstant sind. In diesem Fall "
        "liegen identische Eingabesequenzen oder ein Problem bei "
        "der Rekonstruktion der Ticket-Historien vor."
    )


def attribution_barplot(values, names, title, xlabel, n_top=8):
    pairs = sorted(zip(names, values), key=lambda x: abs(float(x[1])))[-n_top:]
    labels = [FEATURE_LABELS.get(name, name) for name, _ in pairs]
    vals = [float(value) for _, value in pairs]
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, vals, color=colors, alpha=0.88)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        mpatches.Patch(color="#C44E52", label="positiver Beitrag"),
        mpatches.Patch(color="#4C72B0", label="negativer Beitrag"),
    ], fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


st.title("📈 Predictive Process Monitoring — Ticket-Analyse")
st.caption("Prognose, Erklärung und Handlungsempfehlungen für laufende Helpdesk-Tickets")
cols = st.columns(4)
cols[0].metric("Laufende Tickets", len(df_tickets))
cols[1].metric("🔴 Kritisch", int((df_tickets["pred_klasse"] == 3).sum()))
cols[2].metric("🟠 Warnung", int((df_tickets["pred_klasse"] == 2).sum()))
cols[3].metric("🔵 Beobachten", int((df_tickets["pred_klasse"] == 1).sum()))
st.caption(
    f"LSTM-Prognosen · F1 Bias-Korrektur: {'aktiv' if reg_corrector is not None else 'nicht aktiv'} · "
    f"F2 Temperature Scaling (T={TEMPERATURE:.3f}) · {calibration_label}."
)
st.divider()

col_filter, col_time = st.columns([2, 1])
with col_filter:
    filter_klasse = st.selectbox("Risikoklasse filtern:", ["Alle", "🔴 Kritisch", "🟠 Warnung", "🔵 Beobachten", "🟢 Normal"])
with col_time:
    st.write("")
    st.caption(f"Letzte Darstellung: {datetime.now().strftime('%H:%M Uhr')}")

anzeige = df_tickets.copy()
if filter_klasse != "Alle":
    mapping = {"🔴 Kritisch": 3, "🟠 Warnung": 2, "🔵 Beobachten": 1, "🟢 Normal": 0}
    anzeige = anzeige[anzeige["pred_klasse"] == mapping[filter_klasse]]


def aufbereiten(df):
    out = pd.DataFrame()
    out["Ticket-ID"] = df["IncidentID"]
    out["Eröffnet"] = pd.to_datetime(df.get("case_start"), errors="coerce").dt.strftime("%d.%m.%Y %H:%M").fillna("—")
    out["Letzte Aktivität"] = df.get(ACTIVITY_COL, pd.Series("—", index=df.index)).fillna("—")
    out["Wartezeit (h)"] = pd.to_numeric(df["wartezeit_h"], errors="coerce").fillna(0).round(1)
    out["Beobachtete Ereignisse"] = df["prefix_len"].astype(int)
    out["Risikoklasse"] = df["pred_klasse"].apply(lambda k: f"{risiko_symbol(k)} {KLASSEN.get(int(k), '?')}")
    out["Restzeit (h)"] = df["pred_restzeit"].round(1)
    out["_id"] = df["IncidentID"]
    return out


df_tbl = aufbereiten(anzeige)
selection = st.dataframe(
    df_tbl.drop(columns=["_id"]),
    use_container_width=True,
    hide_index=True,
    height=340,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Wartezeit (h)": st.column_config.NumberColumn(format="%.1f h"),
        "Restzeit (h)": st.column_config.NumberColumn("⏳ Restzeit [Prognose]", format="%.1f h"),
        "Risikoklasse": st.column_config.TextColumn("⚠️ Risikoklasse [Prognose]"),
    },
)

if not selection.selection.rows:
    st.info("👆 Ticket auswählen, um LSTM-Erklärungen und Handlungshinweise anzuzeigen.")
else:
    selected_id = df_tbl.iloc[selection.selection.rows[0]]["_id"]
    original_index = int(np.where(bundle.case_ids == selected_id)[0][0])
    row = df_tickets[df_tickets["IncidentID"] == selected_id].iloc[0]
    klasse = int(row["pred_klasse"])
    restzeit = float(row["pred_restzeit"])
    proba = np.asarray([row[f"prob_{i}"] for i in range(N_CLASSES)], dtype=float)
    wartezeit = float(row.get("wartezeit_h", row.get("time_since_prev_h", 0.0)))

    st.divider()
    h1, h2 = st.columns([3, 1])
    h1.subheader(f"{risiko_symbol(klasse)} Ticket {selected_id} — {KLASSEN.get(klasse, '?')}")
    fertig = addiere_arbeitsstunden(datetime.now(), restzeit)
    h2.caption(f"Geschätzte Fertigstellung:\n**{fertig.strftime('%d.%m.%Y %H:%M')}**")

    g1, g2, g3 = st.columns(3)

    with g1:
        st.metric(
            "⏳ Restzeit (vorhergesagt)",
            f"{restzeit:.1f} h",
            help="Vom F1-LSTM vorhergesagte verbleibende Bearbeitungszeit.",
        )
        st.caption(f"≈ {restzeit / 8:.1f} Arbeitstage")

    with g2:
        st.metric(
            "⚠️ Risikoklasse",
            f"{risiko_symbol(klasse)} {KLASSEN.get(klasse, '?')}",
            help="Vom F2-LSTM vorhergesagte Verzögerungsrisikoklasse.",
        )
        st.caption("Vier Klassen: Normal bis Kritisch")

    with g3:
        st.metric(
            "🛡️ Klassenwahrscheinlichkeit",
            f"{float(proba[klasse]):.0%}",
            help=(
                "Geschätzte Wahrscheinlichkeit der vom Modell "
                "ausgewählten Risikoklasse."
            ),
        )
        st.caption(f"Aktuelle Wartezeit: {wartezeit:.1f} h")

    st.divider()
    d1, d2, d3 = st.columns([1.05, 1.25, 1.0])
    with d1:
        st.markdown("**📄 Ticket-Details**")
        start = pd.to_datetime(row.get("case_start"), errors="coerce")
        start_text = start.strftime("%d.%m.%Y %H:%M") if pd.notna(start) else "—"
        st.markdown(
            f"| Merkmal | Wert |\n|---|---|\n"
            f"| Eröffnet | {start_text} |\n"
            f"| Letzte Aktivität | {row.get(ACTIVITY_COL, '—')} |\n"
            f"| Beobachtete Ereignisse | {int(row['prefix_len'])} |\n"
            f"| Wartezeit | {wartezeit:.1f} h |"
        )
        st.markdown("**Klassenwahrscheinlichkeiten:**")
        for k in range(N_CLASSES):
            st.progress(float(proba[k]), text=f"{risiko_symbol(k)} {KLASSEN.get(k, '?')}: {proba[k]:.0%}")
        with st.expander("📋 Finale Modellfeatures", expanded=True):
            items = []

            for feature in FEATURES:
                value = float(row.get(feature, 0.0))

                items.append({
                    "Merkmal": FEATURE_LABELS.get(feature, feature),
                    "Wert": f"{value:.3f}",
                    "Berechnung": FEATURE_FORMULAS.get(feature, "—"),
                    "Bedeutung": FEATURE_TOOLTIPS.get(feature, "—"),
                })

            st.dataframe(
                pd.DataFrame(items),
                hide_index=True,
                use_container_width=True,
                height=410,
                column_config={
                    "Merkmal": st.column_config.TextColumn(
                        "Merkmal",
                        width="medium",
                    ),
                    "Wert": st.column_config.TextColumn(
                        "Wert",
                        width="small",
                    ),
                    "Berechnung": st.column_config.TextColumn(
                        "Berechnung",
                        width="large",
                    ),
                    "Bedeutung": st.column_config.TextColumn(
                        "Bedeutung",
                        width="large",
                    ),
                },
            )

    with d2:
        st.markdown("**💡 Warum hat das Modell so entschieden?**")
        st.caption("Rot erhöht die Vorhersage, Blau senkt sie.")
        x_num = bundle.x_num[original_index:original_index + 1]
        x_act = bundle.x_act[original_index:original_index + 1]
        with st.spinner("Lokale Erklärung wird berechnet …"):
            ig_f1 = integrated_gradients_numeric(f1_model, x_num, x_act, task="F1", steps=24).sum(axis=1)[0]
            ig_f2 = integrated_gradients_numeric(f2_model, x_num, x_act, task="F2", class_id=klasse, steps=24).sum(axis=1)[0]
            selected_history = prefix_history[
                prefix_history["IncidentID"].astype(str)
                == str(selected_id)
            ].sort_values("prefix_len")

            if "is_demo_current" in selected_history.columns:
                current_mask = (
                    selected_history["is_demo_current"]
                    .fillna(False)
                    .astype(bool)
                )
                if current_mask.any():
                    current_prefix = int(
                        selected_history.loc[
                            current_mask,
                            "prefix_len"
                        ].max()
                    )
                    selected_history = selected_history[
                        selected_history["prefix_len"]
                        <= current_prefix
                    ]

            raw_activity_sequence = (
                selected_history[ACTIVITY_COL]
                .fillna("Unknown")
                .astype(str)
                .tolist()
            )[-MAX_LEN:]

            act_f2 = activity_occlusion_local(
                f2_model,
                x_num,
                x_act,
                index_to_activity,
                OOV_INDEX,
                PAD_INDEX,
                task="F2",
                class_id=klasse,
                raw_activities=raw_activity_sequence,
            )

        fig1 = attribution_barplot(
            ig_f1, FEATURES, "Restzeit — numerische Einflussfaktoren",
            "Beitrag zur Restzeitvorhersage",
        )
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
        fig2 = attribution_barplot(
            ig_f2, FEATURES, f"Risikoklasse — numerische Einflussfaktoren",
            "Beitrag zur Klassenwahrscheinlichkeit",
        )
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        if not act_f2.empty:
            oov_share = float(
                act_f2["is_oov"].mean()
            )

            # Wiederholte Aktivitäten positionsbezogen kennzeichnen, damit
            # mehrere identische Labels im Diagramm sichtbar bleiben.
            act_f2 = act_f2.copy()
            act_f2["plot_label"] = (
                act_f2["activity"].astype(str)
                + " · Pos. "
                + act_f2["relative_position"].astype(str)
            )

            top_act = (
                act_f2
                .reindex(
                    act_f2["contribution"]
                    .abs()
                    .sort_values()
                    .index
                )
                .tail(8)
            )

            max_abs_contribution = float(
                top_act["contribution"]
                .abs()
                .max()
            )

            if max_abs_contribution < 1e-10:
                st.warning(
                    "Die Aktivitätssequenz verändert die vorhergesagte "
                    "Klassenwahrscheinlichkeit für dieses Ticket praktisch "
                    "nicht. Dies ist ein valides lokales Ergebnis und kein "
                    "Darstellungsfehler."
                )
            else:
                fig3 = attribution_barplot(
                    top_act[
                        "contribution"
                    ].to_numpy(),
                    top_act[
                        "plot_label"
                    ].tolist(),
                    "Einfluss der Aktivitäten (Sequenz)",
                    "Maskierungs-/Occlusion-Wirkung auf die "
                    "Klassenwahrscheinlichkeit",
                )
                st.pyplot(
                    fig3,
                    use_container_width=True
                )
                plt.close(fig3)


    with d3:
        st.markdown("**🎯 Handlungsempfehlungen**")
        handlungsempfehlung_anzeigen(
            klasse=klasse,
            restzeit_h=restzeit,
            proba=proba.tolist(),
            wartezeit_h=wartezeit,
        )

st.divider()
st.caption(
    "Explainable PPM Prototype · LSTM-basierte Prognose · "
    "Die Prognosen und Handlungshinweise unterstützen Entscheidungen, garantieren aber keine Prozessausgänge."
)
