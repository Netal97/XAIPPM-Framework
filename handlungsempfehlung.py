"""Regelbasierte Entscheidungsunterstützung für das LSTM-PPM-Dashboard.

Die Empfehlungen sind nicht kausal validiert. Sie übersetzen Prognosen in
nachvollziehbare operative Hinweise und werden in der UI entsprechend markiert.
"""
from __future__ import annotations

import streamlit as st

RISIKO_SYMBOL = {0: "🟢", 1: "🔵", 2: "🟠", 3: "🔴"}
RISIKO_BEZEICHNUNG = {0: "Normal", 1: "Beobachten", 2: "Warnung", 3: "Kritisch"}
RISIKO_FARBE = {0: "#2e7d32", 1: "#1565c0", 2: "#e65100", 3: "#c62828"}


def risiko_symbol(klasse: int) -> str:
    return RISIKO_SYMBOL.get(int(klasse), "⚪")


def risiko_bezeichnung(klasse: int) -> str:
    return RISIKO_BEZEICHNUNG.get(int(klasse), "Unbekannt")



def handlungsempfehlung_anzeigen(
    klasse: int,
    restzeit_h: float,
    proba: list[float],
    wartezeit_h: float,
) -> None:
    """Zeigt transparente, regelbasierte Handlungshinweise."""
    klasse = int(klasse)
    konfidenz = float(proba[klasse]) if 0 <= klasse < len(proba) else 0.0
    restzeit_arbeitstage = float(restzeit_h) / 8.0
    blockiert = float(wartezeit_h) > 10

    st.caption(
        "Die folgenden Hinweise sind regelbasiert und nicht kausal bzw. im Realbetrieb validiert. "
        "Sie unterstützen die Priorisierung, ersetzen aber keine fachliche Entscheidung."
    )

    common = (
        f"Prognostizierte Restzeit: ca. **{restzeit_arbeitstage:.1f} Arbeitstage**. "
        f"Klassenwahrscheinlichkeit: **{konfidenz:.0%}**."
    )

    waiting_hint = (
        " Die aktuelle Wartezeit deutet zusätzlich auf eine mögliche Blockierung hin."
        if blockiert else ""
    )

    if klasse == 0:
        st.success(
            f"✅ **NORMAL — reguläre Bearbeitung**\n\n{common}{waiting_hint}\n\n"
            "**Empfohlene Maßnahmen:**\n"
            "- Reguläre Kontrolle gemäß interner Servicevereinbarung beibehalten\n"
            "- Bei deutlich steigender Wartezeit erneut bewerten\n"
            "- Keine automatische Eskalation auslösen"
        )
    elif klasse == 1:
        st.info(
            f"🔵 **BEOBACHTEN — moderates Risiko**\n\n{common}{waiting_hint}\n\n"
            "**Empfohlene Maßnahmen:**\n"
            "- Im nächsten Team-Review prüfen\n"
            "- Bearbeitungsrhythmus und neue Ereignisse beobachten\n"
            "- Bei steigender Risikoklasse oder Wartezeit priorisieren"
        )
    elif klasse == 2:
        st.warning(
            f"⚠️ **WARNUNG — erhöhtes Verzögerungsrisiko**\n\n{common}{waiting_hint}\n\n"
            "**Empfohlene Maßnahmen:**\n"
            "- Zuständiges Team zeitnah informieren\n"
            "- Mögliche Blockierungen aktiv klären\n"
            "- Fortschritt kurzfristig erneut bewerten\n"
            "- Priorität bei ausbleibendem Fortschritt erhöhen"
        )
    elif klasse == 3:
        st.error(
            f"🚨 **KRITISCH — unmittelbare Prüfung erforderlich**\n\n{common}{waiting_hint}\n\n"
            "**Empfohlene Maßnahmen:**\n"
            "- Ticket und Ursachen sofort fachlich prüfen\n"
            "- Teamleitung bzw. verantwortliche Rolle informieren\n"
            "- Ressourcen- oder Prioritätsanpassung erwägen\n"
            "- Externe Kommunikation nur nach fachlicher Bestätigung auslösen"
        )
