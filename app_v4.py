# ============================================================
# PPM FRAMEWORK — STREAMLIT DASHBOARD v3
# Helpdesk Italia · Predictive Process Monitoring
#
# Starten mit:
#   streamlit run src/app_v2.py
#
# Benötigte Dateien in data/cache/:
#   - models_classical.joblib
#   - isotonic_calibrators.joblib
#   - config.pkl
#
# Benötigte Dateien in data/:
#   - tickets_it_demo.csv
#
# Benötigte Module in src/:
#   - handlungsempfehlung.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
import pickle
from datetime import datetime, timedelta

# ============================================================
# ARBEITSZEIT-HILFSFUNKTIONEN
#
# WICHTIG: remaining_h (und alle abgeleiteten Groessen wie
# heuristic_remaining_h) sind Modell-intern weiterhin in
# KALENDERSTUNDEN berechnet — das aendert dieser Block NICHT
# (dafuer muesste 04_feature_engineering neu berechnet und die
# gesamte Modell-Pipeline neu trainiert werden).
#
# Diese Funktionen betreffen NUR die ANZEIGE in der UI: "wie
# viele Arbeitstage sind das?" und "wann ist die Erledigung
# realistisch zu erwarten?" sollen die uebliche Arbeitswoche
# widerspiegeln (Mo-Fr, 8h/Tag) statt naiv 24h/Tag bzw. Naechte
# und Wochenenden als produktive Zeit zu zaehlen.
#
# Arbeitszeitfenster hier: 08:00-16:00 Uhr, Mo-Fr — bei Bedarf
# an die tatsaechlichen Geschaeftszeiten anpassen.
# ============================================================
ARBEITSSTART_STUNDE = 8
ARBEITSENDE_STUNDE  = 16
ARBEITSTAGE_WOCHENTAGE = {0, 1, 2, 3, 4}  # Montag=0 ... Freitag=4
STUNDEN_PRO_ARBEITSTAG = ARBEITSENDE_STUNDE - ARBEITSSTART_STUNDE  # 8


def naechster_arbeitszeitpunkt(dt: datetime) -> datetime:
    """Verschiebt dt auf den naechsten gueltigen Arbeitszeitpunkt
    (Mo-Fr, innerhalb 08:00-16:00), falls dt ausserhalb liegt."""
    while True:
        if dt.weekday() not in ARBEITSTAGE_WOCHENTAGE:
            dt = (dt + timedelta(days=1)).replace(
                hour=ARBEITSSTART_STUNDE, minute=0,
                second=0, microsecond=0)
            continue
        if dt.hour < ARBEITSSTART_STUNDE:
            dt = dt.replace(hour=ARBEITSSTART_STUNDE, minute=0,
                             second=0, microsecond=0)
        elif dt.hour >= ARBEITSENDE_STUNDE:
            dt = (dt + timedelta(days=1)).replace(
                hour=ARBEITSSTART_STUNDE, minute=0,
                second=0, microsecond=0)
            continue
        return dt


def addiere_arbeitsstunden(start: datetime, stunden: float) -> datetime:
    """Addiert 'stunden' Arbeitsstunden zu 'start', unter
    Beruecksichtigung von Mo-Fr / 08:00-16:00 (Naechte und
    Wochenenden werden uebersprungen, nicht mitgezaehlt)."""
    dt = naechster_arbeitszeitpunkt(start)
    rest = float(stunden)
    while rest > 1e-9:
        feierabend = dt.replace(hour=ARBEITSENDE_STUNDE, minute=0,
                                 second=0, microsecond=0)
        verfuegbar_h = (feierabend - dt).total_seconds() / 3600
        if rest <= verfuegbar_h:
            dt = dt + timedelta(hours=rest)
            rest = 0.0
        else:
            rest -= verfuegbar_h
            dt = naechster_arbeitszeitpunkt(
                (dt + timedelta(days=1)).replace(
                    hour=ARBEITSSTART_STUNDE, minute=0,
                    second=0, microsecond=0))
    return dt


def stunden_zu_arbeitstagen(stunden: float) -> float:
    """Wandelt eine Kalenderstunden-Dauer in Arbeitstage um
    (8h/Tag), rein fuer die Anzeige."""
    return stunden / STUNDEN_PRO_ARBEITSTAG


from handlungsempfehlung import (
    risiko_symbol,
    risiko_bezeichnung,
    handlungsempfehlung_anzeigen,
    RISIKO_SYMBOL,
    RISIKO_BEZEICHNUNG,
    RISIKO_FARBE,
)
from gauges import gauge_progress, gauge_remaining_time, gauge_risk

# ── Seitenkonfiguration ───────────────────────────────────────
st.set_page_config(
    page_title="PPM Dashboard",
    page_icon="🎯",
    layout="wide"
)

CACHE_DIR = 'data/cache'
DATA_DIR  = 'data'

# ── Feature Labels (für Detail-Panel) ────────────────────────
FEATURE_LABELS = {
    'prefix_len'             : 'Anzahl bisheriger Ereignisse',
    'pct_complete'           : 'Fortschritt (0–1)',
    'elapsed_h'              : 'Vergangene Zeit (h)',
    'time_since_prev_h'      : 'Zeit seit letztem Ereignis (h)',
    'dow_sin'                : 'Wochentag — Sinus',
    'dow_cos'                : 'Wochentag — Kosinus',
    'n_distinct_acts_so_far' : 'Verschiedene Aktivitäten',
    'heuristic_remaining_h'  : 'Naive Restzeit-Schätzung (h)',
    'gap_max_so_far'         : 'Längste bisherige Pause (h)',
    'pace_ratio'             : 'Tempo-Index',
    'events_per_hour'        : 'Ereignisse pro Stunde',
    'activity_diversity'     : 'Aktivitätsvielfalt (0–1)',
    'gap_cv'                 : 'Wartezeit-Variationskoeff.',
}

FEATURE_TOOLTIPS = {
    'prefix_len'             : 'Wie viele Schritte bisher protokolliert?',
    'pct_complete'           : '0 % = gerade geöffnet | 100 % = abgeschlossen',
    'elapsed_h'              : 'Gesamtdauer seit Ticketerstellung',
    'time_since_prev_h'      : 'Lange Pause = mögliche Blockierung',
    'dow_sin'                : 'Zyklische Wochentag-Kodierung',
    'dow_cos'                : 'Zyklische Wochentag-Kodierung',
    'n_distinct_acts_so_far' : 'Viele verschiedene Aktivitäten = komplexer Fall',
    'heuristic_remaining_h'  : 'Schätzung: vergangene Zeit × (1/Fortschritt − 1)',
    'gap_max_so_far'         : 'Die längste Pause im bisherigen Verlauf',
    'pace_ratio'             : '< 1 = Ticket beschleunigt | > 1 = Ticket verlangsamt',
    'events_per_hour'        : 'Hohe Rate = aktiv bearbeitetes Ticket',
    'activity_diversity'     : 'Nahe 1 = viele verschiedene Bearbeitungsschritte',
    'gap_cv'                 : 'Hoch = sehr unregelmäßiger Bearbeitungsrhythmus',
}


# ============================================================
# LADEN
# ============================================================

@st.cache_resource
def modelle_laden():
    modelle      = joblib.load(f'{CACHE_DIR}/models_classical.joblib')
    kalibratoren = joblib.load(
        f'{CACHE_DIR}/isotonic_calibrators.joblib')
    with open(f'{CACHE_DIR}/config.pkl', 'rb') as f:
        cfg = pickle.load(f)
    erklaerer_f1 = shap.TreeExplainer(modelle['xgb_reg'])
    erklaerer_f2 = shap.TreeExplainer(modelle['xgb_clf'])
    return modelle, kalibratoren, cfg, erklaerer_f1, erklaerer_f2


@st.cache_data
def tickets_laden():
    df = pd.read_csv(f'{DATA_DIR}/tickets_it_demo.csv')
    df['case_start'] = pd.to_datetime(
        df['case_start'], utc=False, errors='coerce')
    return df


modelle, kalibratoren, cfg, erklaerer_f1, erklaerer_f2 = modelle_laden()
FEATURES  = cfg['FEATURES_GENERIC']
N_CLASSES = cfg['N_CLASSES']
KLASSEN   = cfg['CLASS_NAMES']
df_raw    = tickets_laden()


# ============================================================
# VORHERSAGE FÜR ALLE TICKETS (stündlich gecacht)
# ============================================================

@st.cache_data(ttl=3600)
def alle_vorhersagen_berechnen(csv_hash: int):
    """
    Berechnet F1 + F2 für alle Tickets.
    ttl=3600 → automatische Aktualisierung jede Stunde.
    csv_hash wird als Cache-Schlüssel übergeben.
    """
    df = tickets_laden()
    X  = df[FEATURES].fillna(0).replace([np.inf, -np.inf], 0)

    # F1 — Verbleibende Bearbeitungszeit
    pred_log  = modelle['xgb_reg'].predict(X)
    restzeit  = np.expm1(pred_log)
    restzeit  = np.clip(restzeit, 0, 8760)

    # F2 — Klassenwahrscheinlichkeiten + Kalibrierung
    proba_roh = modelle['xgb_clf'].predict_proba(X)  # (n, 4)
    proba_kal = np.zeros_like(proba_roh)
    for k, iso in kalibratoren.items():
        proba_kal[:, k] = iso.predict(proba_roh[:, k])
    # Normalisieren
    summen = proba_kal.sum(axis=1, keepdims=True)
    proba_kal = proba_kal / np.where(summen > 0, summen, 1)

    klassen_pred = proba_kal.argmax(axis=1)

    return restzeit, proba_kal, klassen_pred


def vorhersage_einzeln(zeile: pd.Series):
    """Berechnet F1 + F2 für ein einzelnes Ticket (manueller Refresh)."""
    X = pd.DataFrame([{
        m: float(zeile[m]) for m in FEATURES
        if m in zeile.index
    }]).reindex(columns=FEATURES, fill_value=0)

    pred_log = modelle['xgb_reg'].predict(X)[0]
    restzeit = float(np.expm1(pred_log))
    restzeit = max(0.0, min(restzeit, 8760.0))

    proba_roh = modelle['xgb_clf'].predict_proba(X)[0]
    proba_kal = np.zeros(N_CLASSES)
    for k, iso in kalibratoren.items():
        proba_kal[k] = float(iso.predict([proba_roh[k]])[0])
    s = proba_kal.sum()
    if s > 0:
        proba_kal /= s
    klasse = int(proba_kal.argmax())
    # Explicitement Python float pour éviter dtype float32 dans pandas
    return float(restzeit), [float(p) for p in proba_kal], klasse


def shap_extrahieren(shap_out, k):
    """Robuste Extraktion der SHAP-Werte für Klasse k."""
    if isinstance(shap_out, list):
        arr = shap_out[k]
        return arr[0] if arr.ndim == 2 else arr
    if hasattr(shap_out, 'ndim'):
        if shap_out.ndim == 3:
            return shap_out[0, :, k]
        if shap_out.ndim == 2:
            return shap_out[0]
    return shap_out


def shap_barplot(shap_werte, feature_werte, feature_namen,
                 titel, n_top=8):
    paare = sorted(
        zip(feature_namen, shap_werte, feature_werte),
        key=lambda x: abs(x[1])
    )[-n_top:]
    bezeichnungen = [FEATURE_LABELS.get(f, f) for f, _, _ in paare]
    werte  = [w for _, w, _ in paare]
    farben = ['#C44E52' if w > 0 else '#4C72B0' for w in werte]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(bezeichnungen, werte, color=farben, alpha=0.85,
            edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(titel, fontsize=10, fontweight='bold', pad=10)
    ax.set_xlabel(
        'SHAP-Wert (→ erhöht | ← senkt)', fontsize=8)
    ax.set_facecolor('none')
    rot  = mpatches.Patch(color='#C44E52', label='↑ erhöht')
    blau = mpatches.Patch(color='#4C72B0', label='↓ senkt')
    ax.legend(handles=[rot, blau], fontsize=8, loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig


# ============================================================
# VORHERSAGEN FÜR ALLE TICKETS BERECHNEN
# ============================================================

# Manueller Refresh-State pro Ticket
if 'refresh_state' not in st.session_state:
    st.session_state.refresh_state = {}

csv_hash = hash(tuple(df_raw['IncidentID'].tolist()))
restzeiten, probas, klassen = alle_vorhersagen_berechnen(csv_hash)

# Manuell neu berechnete Tickets überschreiben
df_tickets = df_raw.copy()
df_tickets['pred_restzeit'] = restzeiten
df_tickets['pred_klasse']   = klassen
for i in range(N_CLASSES):
    df_tickets[f'prob_{i}'] = probas[:, i]

# Manuelle Refreshes anwenden
# Konvertierung zu Python float/int um dtype-Konflikte zu vermeiden
for inc_id, vals in st.session_state.refresh_state.items():
    mask = df_tickets['IncidentID'] == inc_id
    df_tickets.loc[mask, 'pred_restzeit'] = float(vals['restzeit'])
    df_tickets.loc[mask, 'pred_klasse']   = int(vals['klasse'])
    for i in range(N_CLASSES):
        df_tickets.loc[mask, f'prob_{i}'] = float(vals['proba'][i])

# Sortieren: Kritisch zuerst, dann Restzeit aufsteigend
df_tickets = df_tickets.sort_values(
    ['pred_klasse', 'pred_restzeit'],
    ascending=[False, True]
).reset_index(drop=True)


# ============================================================
# KOPFZEILE
# ============================================================
st.title("🎯 PPM Dashboard — Helpdesk")

n_krit = (df_tickets['pred_klasse'] == 3).sum()
n_warn = (df_tickets['pred_klasse'] == 2).sum()
col_t1, col_t2, col_t3, col_t4 = st.columns(4)
col_t1.metric("Tickets gesamt", len(df_tickets))
col_t2.metric("🔴 Kritisch",    int(n_krit))
col_t3.metric("🟠 Warnung",     int(n_warn))
col_t4.metric("🔵 Beobachten",
              int((df_tickets['pred_klasse'] == 1).sum()))

st.caption(
    "Prognose wird automatisch **stündlich** aktualisiert "
    "(XGBoost · Isotonic-Kalibrierung) · "
    "Manueller Refresh pro Ticket über den Button ↻ möglich"
)
st.divider()


# ============================================================
# FILTER-TOOLBAR
# ============================================================
st.subheader("📋 Laufende Tickets")

col_f1, col_f2 = st.columns([2, 1])
with col_f1:
    filter_klasse = st.selectbox(
        "Risikoklasse filtern:",
        options=[
            "Alle",
            "🔴 Kritisch",
            "🟠 Warnung",
            "🔵 Beobachten",
            "🟢 Normal"
        ]
    )

with col_f2:
    st.write("")
    st.write("")
    letzte_update = datetime.now().strftime('%H:%M Uhr')
    st.caption(f"Letzte Aktualisierung: {letzte_update}")

# Filter anwenden
df_anzeige = df_tickets.copy()
if filter_klasse != "Alle":
    k_map = {
        "🔴 Kritisch": 3, "🟠 Warnung": 2,
        "🔵 Beobachten": 1, "🟢 Normal": 0
    }
    df_anzeige = df_anzeige[
        df_anzeige['pred_klasse'] == k_map[filter_klasse]]



# ============================================================
# TICKET-TABELLE MIT PROGNOSE-SPALTEN
# ============================================================

# Tabelle aufbereiten
def aufbereiten(df):
    out = pd.DataFrame()
    out['Ticket-ID']        = df['IncidentID']
    out['Eröffnet']         = pd.to_datetime(
        df['case_start'], errors='coerce'
    ).dt.strftime('%d.%m.%Y %H:%M').fillna('—')
    out['Letzte Aktivität'] = df['concept:name'].fillna('—')
    out['Fortschritt']      = (df['pct_complete'] * 100).round(1)
    out['Wartezeit (h)']    = df['time_since_prev_h'].round(1)
    out['Ereignisse']       = (
        df['prefix_len'].astype(int).astype(str) +
        ' / ' + df['case_len'].astype(int).astype(str)
    )
    # Prognose-Spalten (als eigene Gruppe erkennbar)
    out['Risikoklasse']     = df['pred_klasse'].apply(
        lambda k: f"{risiko_symbol(int(k))} "
                  f"{KLASSEN.get(int(k),'?').capitalize()}")
    out['Restzeit (h)']     = df['pred_restzeit'].round(1)
    out['_klasse']          = df['pred_klasse']
    out['_id']              = df['IncidentID']
    return out

df_tbl = aufbereiten(df_anzeige)

tabellenauswahl = st.dataframe(
    df_tbl.drop(columns=['_klasse', '_id']),
    use_container_width=True,
    hide_index=True,
    height=320,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Fortschritt": st.column_config.ProgressColumn(
            "Fortschritt (%)",
            min_value=0, max_value=100,
            format="%.1f %%"
        ),
        "Wartezeit (h)": st.column_config.NumberColumn(
            "Wartezeit (h)", format="%.1f h"
        ),
        "Restzeit (h)": st.column_config.NumberColumn(
            "⏳ Restzeit (h) [Prognose]",
            format="%.1f h"
        ),
        "Risikoklasse": st.column_config.TextColumn(
            "⚠️ Risikoklasse [Prognose]"
        ),
        "Eröffnet": st.column_config.TextColumn(
            "Eröffnet", width="medium"
        ),
    }
)

# Ausgewählte Zeile
if tabellenauswahl.selection.rows:
    sel_idx   = tabellenauswahl.selection.rows[0]
    sel_id    = df_tbl.iloc[sel_idx]['_id']
    sel_zeile = df_tickets[
        df_tickets['IncidentID'] == sel_id].iloc[0]
else:
    sel_id    = None
    sel_zeile = None


# ============================================================
# DETAIL-PANEL — nach Klick auf Zeile
# ============================================================

st.divider()

if sel_zeile is not None:

    klasse_pred = int(sel_zeile['pred_klasse'])
    restzeit    = float(sel_zeile['pred_restzeit'])
    fortschritt = float(sel_zeile['pct_complete'])
    proba       = np.array([
        sel_zeile[f'prob_{i}'] for i in range(N_CLASSES)])

    # ── Ticket-Header ─────────────────────────────────────────
    col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
    with col_h1:
        st.subheader(
            f"{risiko_symbol(klasse_pred)} "
            f"Ticket {sel_id} — "
            f"{KLASSEN.get(klasse_pred,'?').capitalize()}")
    with col_h2:
        if st.button(
                "↻ Prognose neu berechnen",
                key=f"refresh_{sel_id}",
                type="secondary"):
            with st.spinner("Berechne Prognose..."):
                rz, pr, kl = vorhersage_einzeln(sel_zeile)
                st.session_state.refresh_state[sel_id] = {
                    'restzeit': float(rz),
                    'proba'   : [float(p) for p in pr],
                    'klasse'  : int(kl)
                }
                klasse_pred = kl
                restzeit    = rz
                proba       = pr
                st.rerun()
    with col_h3:
        fertig = addiere_arbeitsstunden(datetime.now(), restzeit)
        st.caption(
            f"Geschätzte Fertigstellung (Arbeitszeit Mo–Fr, "
            f"{ARBEITSSTART_STUNDE}–{ARBEITSENDE_STUNDE} Uhr):\n"
            f"**{fertig.strftime('%d.%m.%Y %H:%M')}**"
        )

    # ── 3 Gauges + 1 Kennzahl ─────────────────────────────────
    warte = float(sel_zeile['time_since_prev_h'])
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.plotly_chart(
            gauge_progress(fortschritt),
            use_container_width=True, config={'displayModeBar': False})
        st.markdown(
            f"<p style='text-align:center; margin-top:-12px;'>"
            f"📊 <b>Fortschritt</b><br>"
            f"<span style='color:#6c6f7c; font-size:12px;'>"
            f"Ereignis {int(sel_zeile['prefix_len'])} von "
            f"{int(sel_zeile['case_len'])}</span></p>",
            unsafe_allow_html=True)
    with g2:
        st.plotly_chart(
            gauge_remaining_time(restzeit, float(sel_zeile['elapsed_h'])),
            use_container_width=True, config={'displayModeBar': False})
        st.markdown(
            f"<p style='text-align:center; margin-top:-12px;'>"
            f"⏳ <b>Verbleibende Zeit</b><br>"
            f"<span style='color:#6c6f7c; font-size:12px;'>"
            f"≈ {stunden_zu_arbeitstagen(restzeit):.1f} Arbeitstage</span></p>",
            unsafe_allow_html=True)
    with g3:
        st.plotly_chart(
            gauge_risk(proba[klasse_pred], klasse_pred),
            use_container_width=True, config={'displayModeBar': False})
        st.markdown(
            f"<p style='text-align:center; margin-top:-12px;'>"
            f"⚠️ <b>Risikoklasse — {risiko_symbol(klasse_pred)} "
            f"{KLASSEN.get(klasse_pred,'?').capitalize()}</b><br>"
            f"<span style='color:#6c6f7c; font-size:12px;'>"
            f"Konfidenz der Vorhersage</span></p>",
            unsafe_allow_html=True)
    with g4:
        st.write("")
        st.write("")
        st.metric(
            "⏸ Wartezeit seit letzter Aktivität",
            f"{warte:.1f} h",
            delta=f"{'⚠️ Blockierung möglich' if warte > 10 else '✓ Normal'}",
            delta_color="inverse"
        )

    st.divider()

    # ── 3 Spalten: Details | SHAP | Empfehlung ────────────────
    dcol1, dcol2, dcol3 = st.columns([1.1, 1, 1])

    # ── Spalte 1: Ticket-Details + Features ───────────────────
    with dcol1:
        st.markdown("**📄 Ticket-Details**")

        # Basis-Info
        case_start = pd.to_datetime(
            sel_zeile['case_start'], errors='coerce')
        date_str = case_start.strftime(
            '%d.%m.%Y %H:%M') if pd.notna(case_start) else '—'

        # Ressourcen aus dem Log

        st.markdown(f"""
| Merkmal | Wert |
|---|---|
| Eröffnet | {date_str} |
| Letzte Aktivität | {sel_zeile['concept:name']} |
| Wartezeit seit letzter Akt. | **{warte:.1f} h** |
| Ereignisse | {int(sel_zeile['prefix_len'])} / {int(sel_zeile['case_len'])} |
""")

        # Klassenwahrscheinlichkeiten
        st.markdown("**Klassenwahrscheinlichkeiten:**")
        for k in range(N_CLASSES):
            farbe = ['🟢','🔵','🟠','🔴'][k]
            st.progress(
                float(proba[k]),
                text=f"{farbe} {KLASSEN.get(k,'?').capitalize()}: "
                     f"{proba[k]:.0%}"
            )

        # Prozess-Features
        with st.expander("🔧 Prozess-Features (Modell-Input)",
                         expanded=False):
            feat_data = []
            for f in FEATURES:
                val = float(sel_zeile[f]) \
                    if f in sel_zeile.index else 0.0
                feat_data.append({
                    'Merkmal'  : FEATURE_LABELS.get(f, f),
                    'Wert'     : f"{val:.3f}",
                    'Bedeutung': FEATURE_TOOLTIPS.get(f, '—')
                })
            st.dataframe(
                pd.DataFrame(feat_data),
                hide_index=True,
                use_container_width=True
            )

    # ── Spalte 2: SHAP ────────────────────────────────────────
    with dcol2:
        st.markdown("**🔍 Einflussfaktoren (SHAP)**")
        st.caption(
            "🔴 Rot = erhöht Risiko · "
            "🔵 Blau = senkt Risiko"
        )

        eingabe_df = pd.DataFrame([{
            m: float(sel_zeile[m])
            for m in FEATURES if m in sel_zeile.index
        }]).reindex(columns=FEATURES, fill_value=0)

        # F1 SHAP
        shap_f1_out = erklaerer_f1.shap_values(eingabe_df)
        shap_f1 = shap_f1_out[0] \
            if hasattr(shap_f1_out, 'ndim') \
               and shap_f1_out.ndim == 2 \
            else shap_f1_out
        abb_f1 = shap_barplot(
            shap_f1, eingabe_df.values[0], FEATURES,
            f'F1 — Verbleibende Zeit\n'
            f'{sel_id} | {fortschritt:.0%} Fortschritt'
        )
        st.pyplot(abb_f1, use_container_width=True)
        plt.close()

        # F2 SHAP
        shap_f2_out = erklaerer_f2.shap_values(eingabe_df)
        shap_f2     = shap_extrahieren(shap_f2_out, klasse_pred)
        abb_f2 = shap_barplot(
            shap_f2, eingabe_df.values[0], FEATURES,
            f'F2 — Klasse {klasse_pred} '
            f'({KLASSEN.get(klasse_pred,"?").capitalize()})\n'
            f'{sel_id} | P = {proba[klasse_pred]:.0%}'
        )
        st.pyplot(abb_f2, use_container_width=True)
        plt.close()

    # ── Spalte 3: Handlungsempfehlung ─────────────────────────
    with dcol3:
        st.markdown("**💡 Handlungsempfehlung**")
        handlungsempfehlung_anzeigen(
            klasse      = klasse_pred,
            fortschritt = fortschritt,
            restzeit_h  = restzeit,
            proba       = proba.tolist()
        )

# ── Kein Ticket ausgewählt ────────────────────────────────────
else:
    st.info(
        "👆 Klicken Sie auf eine Zeile in der Tabelle, "
        "um die Details, SHAP-Erklärungen und die "
        "Handlungsempfehlung zu sehen."
    )
    st.markdown("""
    **So funktioniert das Dashboard:**
    1. **Prognose läuft automatisch** — alle 100 Tickets erhalten
       stündlich eine aktualisierte Risikoklasse und Restzeit
    2. **Tabelle sortiert** nach Kritikalität (Kritisch zuerst)
    3. **Klick auf Zeile** → Details, SHAP-Erklärungen,
       Handlungsempfehlung
    4. **↻ Neu berechnen** → sofortiger manueller Refresh
       für ein einzelnes Ticket
    """)

# ── Fußzeile ──────────────────────────────────────────────────
st.divider()
st.caption(
    "🎓 PPM Framework — Masterarbeit Wirtschaftsinformatik  |  "
    "Arnaud Franklin Wafo Totso  |  TH Brandenburg 2026  |  "
    "Betreuer: Prof. Dr. Arthur Tarassow"
)
