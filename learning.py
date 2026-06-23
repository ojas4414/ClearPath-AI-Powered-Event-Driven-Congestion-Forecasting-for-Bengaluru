"""
ClearPath :: learning.py
========================
The POST-EVENT LEARNING loop — the pain point the brief names ("No post-event learning
system") and which the rest of ClearPath did not address.

WHY this module exists:
ClearPath predicts an expected resolution time for each event from its congestion intensity.
Without a feedback loop, that prediction never improves — it stays exactly as calibrated on
day one. This module closes the loop: every time an event's ACTUAL resolution time comes in,
we compare it to what we predicted and nudge an online calibration term so the next prediction
is a little less wrong. Over a stream of events the mean error shrinks — visibly, on a chart.

HOW it learns (deliberately simple and honest):
  pred_raw      = CALM + intensity * (BUSY - CALM)        # the static model's guess
  pred_cal      = pred_raw + bias                         # bias is the learned correction
  on each event: bias += LR * (actual - pred_cal)         # online gradient step (EWMA-like)
We track the running mean-absolute-error of the raw vs calibrated prediction. The calibrated
curve dropping below the raw curve IS the learning, measured on real Astram resolution times.
"""

import os
import json
import datetime as dt

import numpy as np
import pandas as pd

import recommender as R

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_CSV = os.path.join(BASE_DIR, "processed_events.csv")
STATE_PATH = os.path.join(BASE_DIR, "learning_state.json")

LR = 0.05            # online learning rate for the calibration bias
CHECKPOINT_EVERY = 50  # how often to record an (n, mae_raw, mae_cal) point for the chart

CALM = float(R.CALM_BASELINE_MIN)
BUSY = float(R.BUSY_BASELINE_MIN)


def _intensity_from_recent(recent_count):
    """Map an event's prior-2h corridor activity onto a 0-1 congestion intensity (tiered)."""
    if recent_count <= 0:
        return 0.0
    if recent_count <= 2:
        return 0.33
    if recent_count <= 5:
        return 0.66
    return 1.0


def _predict_raw(intensity):
    """The static model's expected resolution time for a given congestion intensity."""
    return CALM + float(np.clip(intensity, 0.0, 1.0)) * (BUSY - CALM)


def backtest(save=True):
    """
    WHAT: Replay every historical event with a known resolution time, CHRONOLOGICALLY, through
    the online calibration. Produces the error-reduction curve the dashboard shows and seeds
    learning_state.json so the loop has history before any live feedback arrives.
    WHY chronological: learning from the future is cheating. Ordering by created_date makes this
    an honest "as events happened, the model got better" simulation.
    """
    df = pd.read_csv(PROCESSED_CSV)
    created = pd.to_datetime(df["created_date"], utc=True, errors="coerce")
    closed = pd.to_datetime(df["closed_datetime"], utc=True, errors="coerce")
    df["resolution_minutes"] = (closed - created).dt.total_seconds() / 60.0
    df["_created"] = created
    valid = df[(df["resolution_minutes"] > 0) & (df["resolution_minutes"] < 1440)
               & (df["_created"].notna())].copy()
    valid = valid.sort_values("_created")

    bias = 0.0
    n = 0
    sum_abs_raw = 0.0
    sum_abs_cal = 0.0
    history = []
    for _, row in valid.iterrows():
        intensity = _intensity_from_recent(int(row.get("events_in_corridor_last_2hrs", 0)))
        actual = float(row["resolution_minutes"])
        pred_raw = _predict_raw(intensity)
        pred_cal = pred_raw + bias

        sum_abs_raw += abs(actual - pred_raw)
        sum_abs_cal += abs(actual - pred_cal)
        n += 1
        bias += LR * (actual - pred_cal)  # online correction step

        if n % CHECKPOINT_EVERY == 0:
            history.append({"n": n,
                            "mae_raw": round(sum_abs_raw / n, 2),
                            "mae_cal": round(sum_abs_cal / n, 2)})

    state = {
        "events_seen": n,
        "bias": round(bias, 3),
        "sum_abs_raw": sum_abs_raw,
        "sum_abs_cal": sum_abs_cal,
        "mae_raw": round(sum_abs_raw / n, 2) if n else 0.0,
        "mae_cal": round(sum_abs_cal / n, 2) if n else 0.0,
        "history": history,
        "feedback_log": [],
        "seeded_from_backtest": True,
    }
    if save:
        with open(STATE_PATH, "w") as f:
            json.dump(state, f, indent=2)
    return state


def _load_state():
    """Load learning state, seeding it from the historical backtest on first use."""
    if not os.path.exists(STATE_PATH):
        return backtest(save=True)
    with open(STATE_PATH) as f:
        return json.load(f)


def record_outcome(predicted_min, actual_min, intensity=None):
    """
    WHAT: Feed one real post-event outcome into the loop — updates the calibration bias and the
    running error, appends to the feedback log, and extends the chart history.
    WHY: This is the live half of the loop. A control room logs how long an event ACTUALLY took
    to clear; the model corrects itself for the next one. Returns the updated status.
    """
    state = _load_state()
    actual = float(actual_min)
    # If a raw prediction is supplied use it; else reconstruct from intensity so the API can
    # post just (actual, intensity) and let the model own the prediction.
    if predicted_min is None and intensity is not None:
        predicted_min = _predict_raw(intensity)
    pred_raw = float(predicted_min)

    bias = float(state.get("bias", 0.0))
    pred_cal = pred_raw + bias

    state["sum_abs_raw"] = float(state.get("sum_abs_raw", 0.0)) + abs(actual - pred_raw)
    state["sum_abs_cal"] = float(state.get("sum_abs_cal", 0.0)) + abs(actual - pred_cal)
    state["events_seen"] = int(state.get("events_seen", 0)) + 1
    bias += LR * (actual - pred_cal)
    state["bias"] = round(bias, 3)

    n = state["events_seen"]
    state["mae_raw"] = round(state["sum_abs_raw"] / n, 2)
    state["mae_cal"] = round(state["sum_abs_cal"] / n, 2)
    state.setdefault("history", []).append(
        {"n": n, "mae_raw": state["mae_raw"], "mae_cal": state["mae_cal"]}
    )
    state.setdefault("feedback_log", []).append({
        "predicted_min": round(pred_raw, 1),
        "actual_min": round(actual, 1),
        "calibrated_min": round(pred_cal, 1),
        "residual_min": round(actual - pred_cal, 1),
        "bias_after": state["bias"],
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
    })
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    return status()


def status():
    """
    WHAT: Return the learning loop's current state for the dashboard — the MAE curve (raw vs
    calibrated), headline improvement, current bias, and the recent feedback log.
    """
    state = _load_state()
    mae_raw = state.get("mae_raw", 0.0)
    mae_cal = state.get("mae_cal", 0.0)
    improvement = round(100.0 * (mae_raw - mae_cal) / mae_raw, 1) if mae_raw else 0.0
    return {
        "events_seen": state.get("events_seen", 0),
        "current_bias_min": state.get("bias", 0.0),
        "mae_raw_min": mae_raw,
        "mae_calibrated_min": mae_cal,
        "improvement_pct": improvement,
        "history": state.get("history", []),
        "feedback_log": state.get("feedback_log", [])[-10:],
        "methodology": (
            "Each historical event's actual resolution time is compared to ClearPath's "
            "prediction; an online calibration bias is nudged by the residual (lr=0.05). "
            "MAE is mean absolute error in minutes. The calibrated curve falling below the "
            "raw curve is the model learning from outcomes — measured on real Astram data, "
            "replayed in chronological order."
        ),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("ClearPath :: Post-Event Learning — backtest seed")
    print("=" * 60)
    st = backtest(save=True)
    print(f"events_seen={st['events_seen']}  MAE raw={st['mae_raw']}  "
          f"MAE calibrated={st['mae_cal']}  learned_bias={st['bias']} min")
    s = status()
    print(f"improvement={s['improvement_pct']}%  checkpoints={len(s['history'])}")
