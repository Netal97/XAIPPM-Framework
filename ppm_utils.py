"""Hilfsfunktionen für das LSTM-basierte PPM-Prototype.

Die Funktionen entsprechen der finalen experimentellen Pipeline:
- elf generische, zum Prognosezeitpunkt verfügbare Features;
- keine Verwendung von case_len, pct_complete oder heuristic_remaining_h;
- numerische Sequenzen plus Aktivitätsindizes;
- Temperature Scaling für F2;
- Integrated Gradients und Activity Occlusion für lokale Erklärungen.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import tensorflow as tf


FINAL_FEATURES = [
    "prefix_len",
    "elapsed_h",
    "time_since_prev_h",
    "dow_sin",
    "dow_cos",
    "n_distinct_acts_so_far",
    "gap_max_so_far",
    "pace_ratio",
    "events_per_hour",
    "activity_diversity",
    "gap_cv",
]

FORBIDDEN_MODEL_FEATURES = {
    "case_len",
    "pct_complete",
    "heuristic_remaining_h",
}


@dataclass(frozen=True)
class SequenceBundle:
    x_num: np.ndarray
    x_act: np.ndarray
    case_ids: np.ndarray
    prefix_len: np.ndarray
    current_rows: pd.DataFrame


def _single_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Gibt auch bei doppelten Spaltennamen zuverlässig eine Series zurück."""
    value = df.loc[:, column]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return value


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    p = np.clip(p, 1e-12, np.inf)
    sums = p.sum(axis=1, keepdims=True)
    invalid = ~np.isfinite(sums).ravel() | (sums.ravel() <= 0)
    if invalid.any():
        p[invalid] = 1.0 / p.shape[1]
        sums = p.sum(axis=1, keepdims=True)
    return p / sums


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Klassisches multiclass Temperature Scaling auf Log-Probabilitäten."""
    t = max(float(temperature), 1e-6)
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) / t
    logits -= logits.max(axis=1, keepdims=True)
    return normalize_probabilities(np.exp(logits))


def validate_feature_contract(features: Iterable[str]) -> list[str]:
    features = list(features)
    forbidden = sorted(set(features) & FORBIDDEN_MODEL_FEATURES)
    if forbidden:
        raise ValueError(
            "Informationsleckage: Nicht zulässige Modellfeatures: "
            + ", ".join(forbidden)
        )
    missing_final = [f for f in FINAL_FEATURES if f not in features]
    if missing_final:
        raise ValueError(
            "Die gespeicherte Feature-Liste entspricht nicht der finalen Pipeline. "
            "Fehlend: " + ", ".join(missing_final)
        )
    return features


def build_prefix_log(events: pd.DataFrame, case_info: pd.DataFrame) -> pd.DataFrame:
    """Erzeugt den finalen, leakage-freien Präfix-Log.

    `remaining_h`, `late` und optionale Abschlussinformationen sind Targets bzw.
    Evaluationsspalten und dürfen nicht als Modellfeatures verwendet werden.
    """
    required_events = {"IncidentID", "time:timestamp", "concept:name"}
    missing = required_events - set(events.columns)
    if missing:
        raise KeyError(f"Fehlende Ereignisspalten: {sorted(missing)}")

    merge_cols = [c for c in ["IncidentID", "case_start", "case_end", "late"] if c in case_info.columns]
    if "IncidentID" not in merge_cols or "case_start" not in merge_cols:
        raise KeyError("case_info benötigt mindestens IncidentID und case_start.")

    df = events.merge(case_info[merge_cols], on="IncidentID", how="inner")
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce", utc=True)
    df["case_start"] = pd.to_datetime(df["case_start"], errors="coerce", utc=True)
    if "case_end" in df.columns:
        df["case_end"] = pd.to_datetime(df["case_end"], errors="coerce", utc=True)

    df = df.dropna(subset=["IncidentID", "time:timestamp", "case_start"])
    df = df.sort_values(["IncidentID", "time:timestamp"]).reset_index(drop=True)
    g = df.groupby("IncidentID", sort=False)

    df["prefix_len"] = g.cumcount() + 1
    df["elapsed_h"] = (
        (df["time:timestamp"] - df["case_start"]).dt.total_seconds() / 3600
    ).clip(lower=0)
    if "case_end" in df.columns:
        df["remaining_h"] = (
            (df["case_end"] - df["time:timestamp"]).dt.total_seconds() / 3600
        ).clip(lower=0)

    df["time_since_prev_h"] = (
        g["time:timestamp"].diff().dt.total_seconds().fillna(0) / 3600
    ).clip(lower=0)

    dow = df["time:timestamp"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Kumulative Anzahl unterschiedlicher Aktivitäten ohne One-Hot-Ausgabe.
    df["n_distinct_acts_so_far"] = (
        df.groupby("IncidentID", sort=False)["concept:name"]
        .transform(lambda s: s.expanding().apply(lambda x: pd.Series(x).nunique(), raw=False))
        .astype(float)
    )

    # Expanding-Statistiken der bisherigen Zeitabstände.
    df["gap_mean_so_far"] = g["time_since_prev_h"].transform(lambda s: s.expanding().mean())
    df["gap_std_so_far"] = g["time_since_prev_h"].transform(lambda s: s.expanding().std()).fillna(0)
    df["gap_max_so_far"] = g["time_since_prev_h"].transform(lambda s: s.expanding().max())

    eps = 0.01
    df["pace_ratio"] = df["time_since_prev_h"] / (df["gap_mean_so_far"] + eps)
    df["events_per_hour"] = df["prefix_len"] / (df["elapsed_h"] + 0.1)
    df["activity_diversity"] = df["n_distinct_acts_so_far"] / df["prefix_len"].clip(lower=1)
    df["gap_cv"] = df["gap_std_so_far"] / (df["gap_mean_so_far"] + eps)

    return df


def load_demo_sequences(
    data_dir: str | Path,
    cache_dir: str | Path,
    case_id_col: str = "IncidentID",
) -> pd.DataFrame:
    """Lädt Demo-Präfixhistorien mit mehreren robusten Fallbacks.

    Bevorzugt wird `tickets_it_demo_sequences.csv`. Alternativ werden die in
    `tickets_it_demo.csv` gewählten Tickets mit `prefix_log_it.pkl` verbunden.
    """
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir)
    seq_csv = data_dir / "tickets_it_demo_sequences.csv"
    demo_csv_candidates = [data_dir / "tickets_it_demo.csv", cache_dir / "tickets_it_demo.csv"]
    prefix_pickle = cache_dir / "prefix_log_it.pkl"

    if seq_csv.exists():
        df = pd.read_csv(seq_csv)
        return df

    demo_csv = next((p for p in demo_csv_candidates if p.exists()), None)
    if demo_csv is None:
        raise FileNotFoundError(
            "Weder tickets_it_demo_sequences.csv noch tickets_it_demo.csv wurde gefunden."
        )

    selected = pd.read_csv(demo_csv)
    if prefix_pickle.exists():
        full = pd.read_pickle(prefix_pickle)
        pieces = []
        for _, row in selected.iterrows():
            case_id = row[case_id_col]
            current_prefix = int(row.get("prefix_len", 10**9))
            history = full[(full[case_id_col] == case_id) & (full["prefix_len"] <= current_prefix)].copy()
            if history.empty:
                continue
            history["is_demo_current"] = history["prefix_len"].eq(current_prefix)
            for meta_col in ["wartezeit_h"]:
                if meta_col in row.index:
                    history[meta_col] = row[meta_col]
            pieces.append(history)
        if pieces:
            return pd.concat(pieces, ignore_index=True)

    # Wissenschaftlich weniger idealer Fallback: nur der aktuelle Präfixzustand.
    selected = selected.copy()
    selected["is_demo_current"] = True
    selected.attrs["single_step_fallback"] = True
    return selected


def create_current_case_sequences(
    prefix_df: pd.DataFrame,
    features: list[str],
    scaler: Any,
    activity_to_index: dict[str, int],
    activity_col: str,
    max_len: int,
    oov_index: int,
    pad_index: int = 0,
    case_id_col: str = "IncidentID",
) -> SequenceBundle:
    """Erzeugt pro Demo-Ticket genau eine Sequenz bis zum aktuellen Präfix."""
    validate_feature_contract(features)
    required = set(features + [case_id_col, "prefix_len", activity_col])
    missing = sorted(required - set(prefix_df.columns))
    if missing:
        raise KeyError("Fehlende Spalten für die LSTM-Sequenz: " + ", ".join(missing))

    x_num, x_act, case_ids, prefix_lengths, rows = [], [], [], [], []
    for case_id, group in prefix_df.groupby(case_id_col, sort=False):
        group = group.sort_values("prefix_len").reset_index(drop=True)
        if "is_demo_current" in group.columns and group["is_demo_current"].astype(bool).any():
            current_index = int(np.where(group["is_demo_current"].astype(bool).to_numpy())[0][-1])
            group = group.iloc[: current_index + 1].copy()
        current = group.iloc[-1].copy()

        numeric = group[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        numeric_scaled = scaler.transform(numeric).astype(np.float32)

        activity_series = _single_column(group, activity_col).fillna("Unknown").astype(str)
        activities = (
            activity_series.map(activity_to_index).fillna(oov_index).astype(np.int32).to_numpy()
        )

        numeric_scaled = numeric_scaled[-max_len:]
        activities = activities[-max_len:]
        padding = max_len - len(numeric_scaled)
        if padding > 0:
            numeric_scaled = np.vstack([
                np.zeros((padding, len(features)), dtype=np.float32),
                numeric_scaled,
            ])
            activities = np.concatenate([
                np.full(padding, pad_index, dtype=np.int32),
                activities,
            ])

        x_num.append(numeric_scaled)
        x_act.append(activities)
        case_ids.append(case_id)
        prefix_lengths.append(int(current["prefix_len"]))
        rows.append(current)

    return SequenceBundle(
        x_num=np.asarray(x_num, dtype=np.float32),
        x_act=np.asarray(x_act, dtype=np.int32),
        case_ids=np.asarray(case_ids),
        prefix_len=np.asarray(prefix_lengths, dtype=np.int32),
        current_rows=pd.DataFrame(rows).reset_index(drop=True),
    )


def predict_lstm(
    f1_model: Any,
    f2_model: Any,
    bundle: SequenceBundle,
    batch_size: int = 256,
    regression_corrector: Any | None = None,
    temperature: float = 1.0,
    calibration_mode: str = "auto",
    return_diagnostics: bool = False,
):
    """Berechnet F1 und F2 mit robuster externer Nachkalibrierung.

    Bei einer IsotonicRegression mit ``out_of_bounds="clip"`` können viele
    unterschiedliche Zero-Shot-Prognosen auf denselben Randwert abgebildet
    werden, wenn sie außerhalb des Kalibrierungsbereichs liegen. Das führte
    im Prototyp beispielsweise zu identischen 849,4 Stunden.

    ``calibration_mode="auto"`` verwendet die Bias-Korrektur nur, wenn:
    - die Rohprognosen innerhalb des gelernten Kalibrierungsbereichs liegen;
    - die Korrektur die vorhandene Variation nicht praktisch vollständig
      zusammenbrechen lässt.

    Außerhalb des Kalibrierungsbereichs bleibt deshalb die Zero-Shot-
    Prognose erhalten. Das ist methodisch sauberer als eine konstante
    Extrapolation am Rand der IsotonicRegression.
    """
    pred_log = f1_model.predict(
        [bundle.x_num, bundle.x_act],
        batch_size=batch_size,
        verbose=0,
    ).reshape(-1)

    remaining_raw = np.clip(
        np.expm1(pred_log),
        0,
        8760,
    ).astype(float)

    remaining = remaining_raw.copy()
    calibration_applied = False
    calibration_reason = "Kein Regressionskalibrator geladen"
    in_range_share = np.nan

    if regression_corrector is not None and calibration_mode != "off":
        x_thresholds = np.asarray(
            getattr(regression_corrector, "X_thresholds_", []),
            dtype=float,
        )

        if x_thresholds.size >= 2:
            lower = float(np.nanmin(x_thresholds))
            upper = float(np.nanmax(x_thresholds))
            in_range = (
                np.isfinite(remaining_raw)
                & (remaining_raw >= lower)
                & (remaining_raw <= upper)
            )
            in_range_share = float(np.mean(in_range))

            corrected = remaining_raw.copy()
            if in_range.any():
                corrected[in_range] = regression_corrector.predict(
                    remaining_raw[in_range]
                )

            corrected = np.clip(corrected, 0, 8760)

            raw_std = float(np.nanstd(remaining_raw))
            corrected_std = float(np.nanstd(corrected))
            raw_unique = int(np.unique(np.round(remaining_raw, 3)).size)
            corrected_unique = int(np.unique(np.round(corrected, 3)).size)

            collapsed = (
                len(corrected) > 1
                and raw_unique > 1
                and (
                    corrected_unique <= 1
                    or (
                        raw_std > 1e-9
                        and corrected_std / raw_std < 0.01
                    )
                )
            )

            if calibration_mode == "force":
                remaining = corrected
                calibration_applied = True
                calibration_reason = "Bias-Korrektur erzwungen"
            elif collapsed:
                remaining = remaining_raw
                calibration_reason = (
                    "Bias-Korrektur verworfen: Prognosevariation "
                    "wäre zusammengebrochen"
                )
            elif not in_range.any():
                remaining = remaining_raw
                calibration_reason = (
                    "Bias-Korrektur nicht angewendet: alle Prognosen "
                    "liegen außerhalb des Kalibrierungsbereichs"
                )
            else:
                remaining = corrected
                calibration_applied = True
                calibration_reason = (
                    "Bias-Korrektur nur innerhalb des "
                    "Kalibrierungsbereichs angewendet"
                )
        else:
            # Fallback für andere Regressionskalibratoren.
            corrected = np.clip(
                regression_corrector.predict(remaining_raw),
                0,
                8760,
            )
            if (
                np.unique(np.round(remaining_raw, 3)).size > 1
                and np.unique(np.round(corrected, 3)).size <= 1
                and calibration_mode == "auto"
            ):
                calibration_reason = (
                    "Bias-Korrektur verworfen: konstante Ausgabe"
                )
            else:
                remaining = corrected
                calibration_applied = True
                calibration_reason = "Bias-Korrektur angewendet"

    probabilities = f2_model.predict(
        [bundle.x_num, bundle.x_act],
        batch_size=batch_size,
        verbose=0,
    )
    probabilities = temperature_scale(
        probabilities,
        temperature,
    )
    classes = probabilities.argmax(axis=1).astype(int)

    diagnostics = {
        "remaining_raw": remaining_raw,
        "calibration_applied": calibration_applied,
        "calibration_reason": calibration_reason,
        "calibration_in_range_share": in_range_share,
        "raw_unique_predictions": int(
            np.unique(np.round(remaining_raw, 3)).size
        ),
        "final_unique_predictions": int(
            np.unique(np.round(remaining, 3)).size
        ),
        "raw_std_h": float(np.nanstd(remaining_raw)),
        "final_std_h": float(np.nanstd(remaining)),
    }

    result = (
        remaining.astype(float),
        probabilities.astype(float),
        classes,
    )

    if return_diagnostics:
        return (*result, diagnostics)

    return result


def integrated_gradients_numeric(
    model: Any,
    x_num: np.ndarray,
    x_act: np.ndarray,
    task: str,
    class_id: int | None = None,
    steps: int = 24,
) -> np.ndarray:
    """Integrated Gradients für ein einzelnes Präfix und numerische Eingaben."""
    x_num_t = tf.convert_to_tensor(x_num, dtype=tf.float32)
    x_act_t = tf.convert_to_tensor(x_act, dtype=tf.int32)
    baseline = tf.zeros_like(x_num_t)
    gradients = []

    for alpha in tf.linspace(0.0, 1.0, steps + 1):
        interpolated = baseline + alpha * (x_num_t - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            output = model([interpolated, x_act_t], training=False)
            if task == "F1":
                target = output[:, 0]
            elif task == "F2" and class_id is not None:
                target = output[:, int(class_id)]
            else:
                raise ValueError("Für F2 muss class_id angegeben werden.")
        gradients.append(tape.gradient(target, interpolated))

    gradients = tf.stack(gradients, axis=0)
    avg = tf.reduce_mean((gradients[:-1] + gradients[1:]) / 2.0, axis=0)
    return ((x_num_t - baseline) * avg).numpy()


def activity_occlusion_local(
    model: Any,
    x_num: np.ndarray,
    x_act: np.ndarray,
    index_to_activity: dict[int, str],
    oov_index: int,
    pad_index: int,
    task: str,
    class_id: int | None = None,
    raw_activities: list[str] | None = None,
) -> pd.DataFrame:
    """Lokale Beiträge der Aktivitätssequenz.

    Bekannte Aktivitäten werden durch OOV ersetzt. Eine bereits als OOV
    kodierte Aktivität darf nicht erneut durch OOV ersetzt werden, weil dies
    keine Eingabeänderung und damit zwangsläufig einen Beitrag von null
    erzeugt. In diesem Fall wird die Position stattdessen maskiert (PAD).

    Für externe Logs wie Helpdesk Italia können alle Aktivitätsbezeichnungen
    gegenüber dem auf BPIC14 trainierten Vokabular unbekannt sein. Dann misst
    die Erklärung den Einfluss des Vorhandenseins der Aktivität an der
    jeweiligen Sequenzposition, nicht den semantischen Effekt einer im
    Training bekannten Aktivitätskategorie.
    """
    if len(x_num) != 1:
        raise ValueError(
            "activity_occlusion_local erwartet genau ein Beispiel."
        )

    base = model.predict(
        [x_num, x_act],
        verbose=0,
    )

    if task == "F1":
        base_value = float(
            np.expm1(base.reshape(-1)[0])
        )
    elif task == "F2" and class_id is not None:
        base_value = float(
            base[0, int(class_id)]
        )
    else:
        raise ValueError(
            "Für F2 muss class_id angegeben werden."
        )

    observed_positions = np.where(
        x_act[0] != pad_index
    )[0]

    raw_activities = list(
        raw_activities or []
    )

    # Nur die letzten tatsächlich beobachteten Aktivitäten gehören zu den
    # nicht gepaddeten Positionen.
    if raw_activities:
        raw_activities = raw_activities[
            -len(observed_positions):
        ]

    rows = []

    for relative_position, position in enumerate(
        observed_positions
    ):
        original_index = int(
            x_act[0, position]
        )

        occluded = x_act.copy()

        if original_index == oov_index:
            # OOV -> OOV wäre identisch und ergäbe immer 0.
            replacement_index = pad_index
            perturbation = "OOV-Position maskiert"
        else:
            replacement_index = oov_index
            perturbation = "Aktivität durch OOV ersetzt"

        occluded[0, position] = replacement_index

        out = model.predict(
            [x_num, occluded],
            verbose=0,
        )

        if task == "F1":
            value = float(
                np.expm1(
                    out.reshape(-1)[0]
                )
            )
        else:
            value = float(
                out[0, int(class_id)]
            )

        vocabulary_label = index_to_activity.get(
            original_index,
            "<UNKNOWN>",
        )

        if relative_position < len(raw_activities):
            raw_label = str(
                raw_activities[
                    relative_position
                ]
            )
        else:
            raw_label = vocabulary_label

        display_label = raw_label

        rows.append({
            "position": int(position),
            "relative_position": int(
                relative_position + 1
            ),
            "activity_index": original_index,
            "activity": display_label,
            "vocabulary_activity":
                vocabulary_label,
            "is_oov":
                original_index == oov_index,
            "perturbation":
                perturbation,
            "contribution":
                base_value - value,
        })

    return pd.DataFrame(rows)
