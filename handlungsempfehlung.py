# ============================================================
# HANDLUNGSEMPFEHLUNG — PPM Framework
# Modul für die Risikoklassifikation und Handlungsempfehlung.
#
# Importiert in: src/app_v2.py
#
# F2 liefert direkt 4 Risikoklassen (0–3):
#   0 = Normal     → kein Handlungsbedarf
#   1 = Beobachten → reguläre Kontrolle
#   2 = Warnung    → engmaschige Überwachung
#   3 = Kritisch   → Sofortmaßnahmen
#
# Der Bearbeitungsfortschritt (pct_complete) kontextualisiert
# den angezeigten Text ohne die Klassenzugehörigkeit zu ändern.
# ============================================================

import streamlit as st
import numpy as np


# ── Risikoklassen-Mapping ─────────────────────────────────────

RISIKO_SYMBOL = {
    0: "🟢",
    1: "🔵",
    2: "🟠",
    3: "🔴"
}

RISIKO_BEZEICHNUNG = {
    0: "Normal",
    1: "Beobachten",
    2: "Warnung",
    3: "Kritisch"
}

RISIKO_FARBE = {
    0: "#2e7d32",   # Grün
    1: "#f57f17",   # Gelb
    2: "#e65100",   # Orange
    3: "#c62828"    # Rot
}


def risiko_symbol(klasse: int) -> str:
    """Gibt das farbige Symbol für eine Risikoklasse zurück."""
    return RISIKO_SYMBOL.get(klasse, "⚪")


def risiko_bezeichnung(klasse: int) -> str:
    """Gibt die textuelle Bezeichnung einer Risikoklasse zurück."""
    return RISIKO_BEZEICHNUNG.get(klasse, "Unbekannt")


# ── Handlungsempfehlung anzeigen ──────────────────────────────

def handlungsempfehlung_anzeigen(klasse: int,
                                  fortschritt: float,
                                  restzeit_h: float,
                                  proba: list) -> None:
    """
    Zeigt die Handlungsempfehlung basierend auf der vom
    Modell vorhergesagten Risikoklasse (0–3).

    Der Bearbeitungsfortschritt kontextualisiert den Text —
    Klasse 3 bei 20% Fortschritt erfordert andere Maßnahmen
    als dieselbe Klasse bei 90% Fortschritt.

    Parameter:
    ----------
    klasse      : Vorhergesagte Risikoklasse (0/1/2/3)
    fortschritt : pct_complete (0–1)
    restzeit_h  : Vorhergesagte Restzeit in Stunden (F1)
    proba       : Klassenwahrscheinlichkeiten [P0, P1, P2, P3]
    """
    restzeit_tage = restzeit_h / 24

    # Fortschrittskontext für die Textanpassung
    if fortschritt >= 0.7:
        fortschritt_kontext = "späten"
    elif fortschritt >= 0.4:
        fortschritt_kontext = "mittleren"
    else:
        fortschritt_kontext = "frühen"

    # Konfidenz der Vorhersage
    konfidenz = proba[klasse] if klasse < len(proba) else 0.0

    # ── Klasse 0 — Normal ────────────────────────────────────
    if klasse == 0:
        st.success(f"""
        ✅ **NORMAL — Ticket im erwarteten Bearbeitungsrahmen**

        Das Ticket befindet sich in der **{fortschritt_kontext} Phase**
        ({fortschritt:.0%} Fortschritt) und liegt im normalen Zeitplan.
        Vorhergesagte Restzeit: ca. **{restzeit_tage:.1f} Tage**.
        *(Modell-Konfidenz: {konfidenz:.0%})*

        **Empfohlene Maßnahmen:**
        - 📝 Kein sofortiger Handlungsbedarf
        - 🔄 Reguläre Überprüfung **gemäß SLA** beibehalten
        - ✓ Bearbeitungsrhythmus liegt im normalen Bereich
        """)

    # ── Klasse 1 — Beobachten ────────────────────────────────
    elif klasse == 1:
        st.info(f"""
        🔵 **BEOBACHTEN — Moderates Risiko**

        Das Ticket befindet sich in der **{fortschritt_kontext} Phase**
        ({fortschritt:.0%} Fortschritt) mit einem leicht erhöhten Risiko.
        Noch ca. **{restzeit_tage:.1f} Tage** bis zum Abschluss.
        *(Modell-Konfidenz: {konfidenz:.0%})*

        **Empfohlene Maßnahmen:**
        - 📋 Im **nächsten geplanten Team-Review** ansprechen
        - 🔄 Reguläre Überprüfung **gemäß SLA** beibehalten
        - 📌 Bei Verschlechterung des Tempos **erneut bewerten**
        """)

    # ── Klasse 2 — Warnung ───────────────────────────────────
    elif klasse == 2:
        if fortschritt >= 0.7:
            # Spätphase — dringender
            st.warning(f"""
            ⚠️ **WARNUNG — Erhöhtes Risiko in der Spätphase**

            Das Ticket befindet sich bei **{fortschritt:.0%} Fortschritt**
            mit erhöhtem Verzögerungsrisiko. Noch ca.
            **{restzeit_tage:.1f} Tage** — die Zeit für Gegenmaßnahmen
            wird knapper. *(Modell-Konfidenz: {konfidenz:.0%})*

            **Empfohlene Maßnahmen:**
            - 📞 Zuständiges Team **zeitnah informieren**
            - ⏰ Fortschritt **innerhalb der nächsten Stunde** prüfen
            - 📌 Priorität erhöhen falls keine Fortschritte erkennbar
            - 💬 Bearbeiter direkt nach **Blockierungen befragen**
            """)
        else:
            # Früh- oder Mittelphase — noch Zeit
            st.warning(f"""
            ⚠️ **WARNUNG — Erhöhtes Risiko in der {fortschritt_kontext.capitalize()}phase**

            Das Ticket befindet sich bei **{fortschritt:.0%} Fortschritt**
            mit erhöhtem Verzögerungsrisiko.
            Noch ca. **{restzeit_tage:.1f} Tage** bis zum Abschluss.
            *(Modell-Konfidenz: {konfidenz:.0%})*

            **Empfohlene Maßnahmen:**
            - 👁️ Ticket im **nächsten Team-Review** besprechen
            - ⏰ Fortschritt in **2–3 Stunden** erneut prüfen
            - 📌 Ggf. Priorität erhöhen bei ausbleibendem Fortschritt
            - 💬 Blockierungen aktiv identifizieren
            """)

    # ── Klasse 3 — Kritisch ──────────────────────────────────
    elif klasse == 3:
        if fortschritt >= 0.7:
            # Spätphase — Sofortmaßnahmen unbedingt
            st.error(f"""
            🚨 **KRITISCH — Sofortmaßnahmen erforderlich**

            Das Ticket befindet sich bei **{fortschritt:.0%} Fortschritt**
            mit kritischem Verzögerungsrisiko. Bei dieser Kombination
            ist eine SLA-Verletzung sehr wahrscheinlich.
            *(Modell-Konfidenz: {konfidenz:.0%})*

            **Empfohlene Maßnahmen:**
            - 🔺 Ticket **sofort eskalieren** und höchste Priorität setzen
            - 📞 Teamleiter **unmittelbar informieren**
            - 👤 Zusätzliche Ressourcen **sofort zuweisen**
            - 📋 Kunden proaktiv über drohende SLA-Verletzung informieren
            """)
        else:
            # Früh- oder Mittelphase — kritisch aber noch Zeit
            st.error(f"""
            🚨 **KRITISCH — Frühes Warnsignal**

            Das Ticket befindet sich erst bei **{fortschritt:.0%} Fortschritt**
            — aber das Modell stuft es bereits als kritisch ein.
            Noch ca. **{restzeit_tage:.1f} Tage**: präventive Maßnahmen
            sind jetzt besonders wirksam.
            *(Modell-Konfidenz: {konfidenz:.0%})*

            **Empfohlene Maßnahmen:**
            - 🔍 Ursachen für das erhöhte Risiko **sofort analysieren**
            - 📌 Priorität **präventiv erhöhen** — noch Zeit für Eingriff
            - 👁️ Ticket in **engen Überwachungszyklus** aufnehmen
            - 📞 Bearbeiter direkt nach **Hindernissen befragen**
            """)
