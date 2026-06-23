"""
ClearPath :: recommender.py
===========================
The decision brain. Fuses the two models into one actionable response for a traffic controller.

WHY this layer exists:
A probability and a forecast are not decisions. A duty officer needs "how many officers,
barricades yes/no, divert where, and why". This module converts model outputs into a single
0-100 impact score and maps that score to concrete resourcing — the part a hackathon judge
(or a real control room) actually cares about.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import joblib

import models as M

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XGB_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")
HOURLY_CSV = os.path.join(BASE_DIR, "hourly_corridor_counts.csv")
RESOLUTION_STATS_PATH = os.path.join(BASE_DIR, "resolution_stats.json")
ROUTING_PATH = os.path.join(BASE_DIR, "corridor_routing.json")

with open(RESOLUTION_STATS_PATH) as _f:
    RESOLUTION_STATS = json.load(_f)

# Honest per-corridor routing: which corridors' LSTM actually beat the persistence baseline
# on held-out data. Corridors that didn't are served by the climatology fallback instead of
# a forecaster we proved is worse than doing nothing. Loaded once; tolerant if file is absent.
try:
    with open(ROUTING_PATH) as _f:
        CORRIDOR_ROUTING = json.load(_f)
except (FileNotFoundError, json.JSONDecodeError):
    CORRIDOR_ROUTING = {}

# ---------------------------------------------------------------------------------------
# Data-derived response-time model (replaces the old hand-picked severity multipliers).
# ---------------------------------------------------------------------------------------
# EMPIRICAL FINDING (resolution_stats.json, from real Astram closed-event durations):
# resolution time scales with congestion context, not severity label —
#   calm     (0 prior-2h events): ~80 min
#   light    (1-2):               ~92 min
#   moderate (3-5):              ~105 min
#   busy     (6+):              ~242 min   (3x the calm baseline)
# ClearPath's mechanism is forecasting hotspots and pre-positioning resources BEFORE that
# congestion builds, so its benefit is removing the *congestion penalty* — the gap between an
# event's congestion-tier resolution time and the calm baseline. That gap is measured from
# data; we do NOT invent per-severity speed-ups.
#
# The SINGLE remaining assumption (explicit, bounded, conservative): pre-positioning recovers
# this fraction of the measured congestion penalty. Calm events have ~no penalty, so ClearPath
# honestly claims little for them — the savings come from the congested tail, where they're real.
CLEARPATH_EFFECTIVENESS = 0.65

CALM_BASELINE_MIN = float(RESOLUTION_STATS.get("calm_baseline_minutes",
                                               RESOLUTION_STATS["avg_baseline_minutes"]))
_TIER_BASELINE = {
    t: v["mean_minutes"] for t, v in RESOLUTION_STATS.get("by_congestion_tier", {}).items()
}
# Worst observed tier — the top of the empirical resolution-time range a congested event trends toward.
BUSY_BASELINE_MIN = float(_TIER_BASELINE.get("busy", RESOLUTION_STATS["avg_baseline_minutes"]))


def _estimate_response(congestion_intensity):
    """
    WHAT: Map a model-predicted congestion intensity in [0,1] onto the EMPIRICAL resolution-time
    range (calm baseline -> busy baseline, both measured from data), then apply the disclosed
    effectiveness factor to the congestion penalty. Returns
    (expected_baseline_min, estimated_with_clearpath_min, time_saved_min).
    WHY: resolution time is driven by congestion, and ClearPath's own LSTM P(busy) + hour-of-day
    risk ARE its congestion prediction. So a high-risk rush-hour event is expected to resolve
    near the busy baseline (~242 min) and has a large recoverable penalty; a calm event sits near
    the calm baseline (~80 min) and honestly saves almost nothing. Only EFFECTIVENESS is assumed;
    the calm/busy endpoints are observed.
    """
    intensity = float(np.clip(congestion_intensity, 0.0, 1.0))
    expected = CALM_BASELINE_MIN + intensity * (BUSY_BASELINE_MIN - CALM_BASELINE_MIN)
    penalty = expected - CALM_BASELINE_MIN
    estimated = expected - CLEARPATH_EFFECTIVENESS * penalty
    return round(expected, 1), round(estimated, 1), round(expected - estimated, 1)

# Hard-coded diversion playbook: top corridors -> a pre-approved alternate route.
# WHY hard-coded: diversions are an operational/policy decision (which roads can absorb
# overflow), not something to infer from data. Encoding the official playbook keeps the
# recommendation trustworthy and auditable.
DIVERSION_MAP = {
    "Mysore Road": "Magadi Road",
    "Bellary Road": "Tumkur Road",
    "Bellary Road 1": "Tumkur Road",
    "Bellary Road 2": "Tumkur Road",
    "Hosur Road": "Bannerghatta Road",
    "ORR East": "Old Madras Road",
    "ORR East 1": "Old Madras Road",
    "ORR East 2": "Old Madras Road",
    "Tumkur Road": "West of Chord Road",
}

# Per-event-cause weighting applied on top of the base impact score.
# WHY: severity probability and forecast volume capture "how bad/busy", but certain causes
# (a public event, a procession, a VIP movement) cause disruption far beyond what historical
# priority/volume alone would suggest, because they block roads predictably and at scale.
# This multiplier lets the recommender react to cause-specific real-world impact directly.
EVENT_CAUSE_MULTIPLIER = {
    'public_event':        1.40,
    'procession':          1.30,
    'protest':             1.30,  # same disruption profile as procession
    'vip_movement':        1.25,
    'accident':            1.20,
    'congestion':          1.20,
    'construction':        1.15,
    'Fog / Low Visibility': 1.10,
    'water_logging':       1.10,
    'vehicle_breakdown':   1.00,
    'test_demo':           1.00,  # synthetic test entries; neutral weight
    'road_conditions':     0.95,
    'tree_fall':           0.95,
    'pot_holes':           0.90,
    'others':              0.90,
    'Debris':              0.90,
    'debris':              0.90,  # case-variant in raw data (1 row)
}

# Loaded once at import; reused across requests.
_xgb = None
_encoders = None
_hourly = None
_hourly_profile = None  # corridor -> np.array[24] of avg event_count by hour-of-day


def _lazy_load():
    """
    WHAT: Load the XGBoost model, encoders, hourly history, and a per-corridor hour-of-day
    profile once (cached at module level).
    WHY: Loading on every request would be wasteful under the FastAPI server. Lazy + cached
    keeps the first call cheap and every subsequent call instant. The hour-of-day profile is
    what makes the SAME corridor/cause query at different hours actually produce different
    scores -- without it, both the recent-count proxy and the LSTM forecast were hour-blind.
    """
    global _xgb, _encoders, _hourly, _hourly_profile
    if _xgb is None:
        _xgb = joblib.load(XGB_PATH)
        print("[reco] loaded xgb_model.pkl")
    if _encoders is None:
        _encoders = joblib.load(ENCODERS_PATH)
        print("[reco] loaded encoders.pkl")
    if _hourly is None:
        _hourly = pd.read_csv(HOURLY_CSV, parse_dates=["hour_bucket"])
        print("[reco] loaded hourly history")
    if _hourly_profile is None:
        tmp = _hourly.copy()
        tmp["hod"] = tmp["hour_bucket"].dt.hour
        profile = tmp.groupby(["corridor", "hod"])["event_count"].mean()
        _hourly_profile = {}
        for corridor in tmp["corridor"].unique():
            arr = np.zeros(24, dtype=np.float32)
            for h in range(24):
                arr[h] = profile.get((corridor, h), 0.0)
            _hourly_profile[corridor] = arr
        print(f"[reco] built hour-of-day profiles for {len(_hourly_profile)} corridors")


def _safe_encode(encoder, value, fallback_label="Unknown"):
    """
    WHAT: Encode a categorical value with a fitted LabelEncoder, gracefully handling unseen labels.
    WHY: Live requests may carry a corridor/cause the encoder never saw. Rather than crash, we
    fall back to the 'Unknown' code (or 0) so the API stays robust in a demo.
    """
    classes = list(encoder.classes_)
    if value in classes:
        return int(encoder.transform([value])[0])
    if fallback_label in classes:
        return int(encoder.transform([fallback_label])[0])
    return 0


def predict_severity(corridor, event_cause, hour, day_of_week,
                     is_weekend=None, requires_road_closure=False,
                     recent_count=None):
    """
    WHAT: Run the XGBoost model to get P(High priority) for this event context.
    WHY: This is the 'how bad is THIS event' signal. We return the probability (not just the
    class) so the impact score can blend it smoothly with the volume forecast.
    """
    _lazy_load()
    if is_weekend is None:
        is_weekend = 1 if day_of_week in (5, 6) else 0
    if recent_count is None:
        recent_count = _recent_corridor_count(corridor, hour)

    row = {
        "hour": hour,
        "day_of_week": day_of_week,
        "month": 6,  # default to current month; not a strong feature
        "is_weekend": int(is_weekend),
        "corridor_encoded": _safe_encode(_encoders["corridor"], corridor),
        "event_cause_encoded": _safe_encode(_encoders["event_cause"], event_cause),
        "requires_road_closure": int(bool(requires_road_closure)),
        "events_in_corridor_last_2hrs": recent_count,
    }
    X = pd.DataFrame([row])[_xgb["features"]]
    proba = float(_xgb["model"].predict_proba(X)[0][1])
    label = "High" if proba >= 0.5 else "Low"
    return proba, label


def _recent_corridor_count(corridor, hour):
    """
    WHAT: Estimate of recent event volume on a corridor AT THIS HOUR (2h window), using the
    corridor's hour-of-day profile rather than an all-day median.
    WHY: The rolling-2h feature isn't known for a hypothetical future request, so we proxy it
    with the corridor's typical load for that specific hour (and the hour before it) — this is
    what makes a 2am query and a 6pm rush-hour query genuinely differ for the same corridor.
    """
    _lazy_load()
    profile = _hourly_profile.get(corridor)
    if profile is None:
        g = _hourly[_hourly["corridor"] == corridor]
        return int(round(g["event_count"].median() * 2)) if len(g) else 0
    prev_hour = (hour - 1) % 24
    return int(round(profile[hour] + profile[prev_hour]))


def forecast_load(corridor, hour=None):
    """
    WHAT: Use the corridor's LSTM to produce a load signal for the given hour.
    Returns (value, is_prob): when is_binary models are loaded, value is P(busy) in [0,1];
    otherwise value is a predicted event count.  Returning a flag avoids the caller needing
    to know which model generation is installed.
    WHY binary: sparse Poisson counts are hard to regress; "will it be busy?" is a
    well-posed classification problem.  The hour-of-day busy profile (built from training data
    only) is fed as input so inference matches the validation path exactly.
    """
    _lazy_load()
    g = _hourly[_hourly["corridor"] == corridor].sort_values("hour_bucket")
    bundle = M.load_lstm(corridor)
    seq_len = bundle["seq_len"] if bundle else 6

    if bundle is not None and bundle.get("is_binary", False):
        hod_busy = bundle.get("hod_busy_profile")
        if hod_busy is None or hour is None:
            return 0.5, True
        # HONEST ROUTING: only trust the LSTM on corridors where it beat persistence on
        # held-out AUROC. Everywhere else, serve the climatological busy-rate for this
        # hour-of-day (the persistence baseline) rather than a forecaster proven worse.
        route = CORRIDOR_ROUTING.get(corridor, {})
        # Bundle flag wins if present (baked at train time); else fall back to routing file.
        use_lstm = bundle.get("use_lstm")
        if use_lstm is None:
            use_lstm = route.get("use_lstm", False)
        if not use_lstm:
            return float(np.clip(hod_busy[hour % 24], 0.0, 1.0)), True
        hours_seq = [(hour - seq_len + i) % 24 for i in range(seq_len)]
        seq = hod_busy[hours_seq]
        x = torch.from_numpy(seq.reshape(1, seq_len, 1).astype(np.float32))
        with torch.no_grad():
            logit = bundle["model"](x).item()
        p_busy = 1.0 / (1.0 + float(np.exp(-logit)))
        return p_busy, True

    # Legacy regression path (fallback for old .pt files without is_binary flag)
    if hour is None:
        seq = g["event_count"].to_numpy(dtype=np.float32)[-seq_len:] if len(g) else np.array([])
    else:
        profile = _hourly_profile.get(corridor)
        if profile is None:
            seq = np.array([])
        else:
            hours = [(hour - seq_len + i) % 24 for i in range(seq_len)]
            seq = profile[hours]

    if bundle is None or len(seq) < seq_len:
        return (float(g["event_count"].mean()) if len(g) else 0.0), False

    scale = bundle["scale"]
    x = torch.from_numpy((seq / scale).reshape(1, seq_len, 1).astype(np.float32))
    with torch.no_grad():
        pred = bundle["model"](x).item() * scale
    return max(0.0, float(pred)), False


def _corridor_load_ceiling():
    """
    WHAT: Compute a reference 'busy' level (95th percentile of hourly counts citywide).
    WHY: To turn an absolute forecast into a 0-100 score we need a ceiling. The 95th percentile
    is a stable 'very busy hour' benchmark that ignores rare outliers.
    """
    _lazy_load()
    return max(1.0, float(_hourly["event_count"].quantile(0.95)))


def _hour_risk(corridor, hour):
    """
    WHAT: How busy this corridor typically is AT THIS HOUR, normalised 0-1 against its own
    busiest hour (e.g. Mysore Road peaks ~6am/9pm, near-zero midday).
    WHY: XGBoost's severity probability saturates near 100% for several corridors regardless
    of hour (corridor_encoded dominates its feature importance at 0.78 vs hour at 0.007), and
    the LSTM forecast for low-volume corridors barely moves either. Without an explicit
    time-of-day term, the SAME corridor+cause query would score almost identically at 3am and
    7pm, which doesn't match reality. This factor restores that real, data-driven variation.
    """
    _lazy_load()
    profile = _hourly_profile.get(corridor)
    if profile is None or float(profile.max()) == 0:
        return 0.0
    return float(profile[hour] / profile.max())


def compute_impact_score(severity_proba, forecast_count, requires_road_closure=False,
                          event_cause=None, hour_risk=0.0, forecast_is_prob=False):
    """
    WHAT: Blend severity probability + forecast volume + hour-of-day risk (+ closure flag),
    then apply the event-cause multiplier, into a 0-100 score.
    WHY: Officers need ONE number to triage. We weight severity 50% (a high-severity event is
    dangerous even when quiet), volume 30% (congestion risk), and hour-of-day risk 20% (so the
    same corridor/cause genuinely scores higher at rush hour than at 3am), add a closure bump
    because a road closure inherently multiplies disruption, then scale by
    EVENT_CAUSE_MULTIPLIER because the same reading means something very different for a
    pothole vs. a public event blocking the road. Weights are explicit and tunable.
    forecast_is_prob=True: forecast_count is already P(busy) in [0,1], use directly as volume_norm.
    """
    if forecast_is_prob:
        volume_norm = float(np.clip(forecast_count, 0.0, 1.0))
    else:
        ceiling = _corridor_load_ceiling()
        volume_norm = min(1.0, forecast_count / ceiling)
    base_score = (0.50 * severity_proba + 0.30 * volume_norm + 0.20 * hour_risk) * 100.0
    if requires_road_closure:
        base_score = min(100.0, base_score + 12.0)  # closures amplify real-world impact

    multiplier = EVENT_CAUSE_MULTIPLIER.get(event_cause, 1.0)
    score = min(100.0, base_score * multiplier)
    return round(float(base_score), 1), round(float(score), 1), multiplier


def recommend(corridor, event_cause, hour, day_of_week,
              is_weekend=None, requires_road_closure=False):
    """
    WHAT: End-to-end recommendation — severity (XGBoost) + load (LSTM) -> impact score ->
    concrete resourcing decision (officers, barricades, diversion) with a human-readable reason.
    WHY: This is the single function the API calls. It encapsulates the full ClearPath decision
    so the policy lives in exactly one auditable place.

    Thresholds:
        impact > 75  -> Critical : 10 officers, barricades, diversion
        impact > 45  -> High     :  5 officers, barricades, no diversion
        else         -> Normal   :  2 officers, no barricades, no diversion
    """
    proba, sev_label = predict_severity(
        corridor, event_cause, hour, day_of_week,
        is_weekend=is_weekend, requires_road_closure=requires_road_closure,
    )
    forecast, forecast_is_prob = forecast_load(corridor, hour)
    hour_risk = _hour_risk(corridor, hour)
    base_impact, impact, multiplier = compute_impact_score(
        proba, forecast, requires_road_closure, event_cause=event_cause, hour_risk=hour_risk,
        forecast_is_prob=forecast_is_prob,
    )

    if impact > 75:
        severity = "Critical"
        officers, barricades, diversion = 10, True, True
    elif impact > 45:
        severity = "High"
        officers, barricades, diversion = 5, True, False
    else:
        severity = "Normal"
        officers, barricades, diversion = 2, False, False

    suggested = DIVERSION_MAP.get(corridor, "No pre-approved diversion") if diversion else ""

    # Data-derived response estimate. Congestion intensity = the model's own congestion signals
    # (LSTM P(busy) + hour-of-day risk), mapped onto the empirical calm->busy resolution range.
    # time saved is the congestion penalty ClearPath's pre-positioning removes. No invented
    # per-severity multipliers — only the disclosed effectiveness factor.
    forecast_prob = forecast if forecast_is_prob else hour_risk
    congestion_intensity = float(np.clip(0.5 * hour_risk + 0.5 * forecast_prob, 0.0, 1.0))
    baseline_min, estimated_min, time_saved_min = _estimate_response(congestion_intensity)

    if forecast_is_prob:
        forecast_str = f"LSTM busy-hour classifier: P(busy)={forecast:.0%}"
    else:
        forecast_str = f"LSTM forecasts ~{forecast:.1f} events next hour"
    reason = (
        f"{event_cause.replace('_', ' ')} on {corridor} at {hour:02d}:00. "
        f"Model severity P(High)={proba:.0%} ({sev_label}); "
        f"{forecast_str}; "
        f"hour-of-day risk for this corridor: {hour_risk:.0%}. "
        f"Event type multiplier: {multiplier}x. "
        f"Combined impact score {impact}/100 -> {severity}. "
    )
    reason += (
        f"Predicted congestion intensity {congestion_intensity:.0%} -> expected resolution "
        f"{baseline_min} min (vs {CALM_BASELINE_MIN:.0f}-min calm / {BUSY_BASELINE_MIN:.0f}-min "
        f"busy baselines from data); pre-positioning -> {estimated_min} min, saving "
        f"{time_saved_min} min of the congestion penalty. "
    )
    if diversion:
        reason += f"Divert traffic via {suggested}. "
    if requires_road_closure:
        reason += "Road closure requested, resourcing escalated."

    # Transparency note for the dashboard: flag when severity is corridor-driven
    model_note = ""
    if proba > 0.95 or proba < 0.05:
        model_note = (
            "Severity is strongly corridor-driven (corridor_encoded importance=0.78). "
            "The LSTM forecast and hour-of-day risk provide the genuinely predictive "
            "time-varying signal in this score."
        )

    return {
        "severity": severity,
        "impact_score": impact,
        "officers_recommended": officers,
        "barricades_needed": barricades,
        "diversion_suggested": diversion,
        "suggested_diversion": suggested,
        "baseline_response_min": baseline_min,
        "estimated_response_min": estimated_min,
        "time_saved_min": time_saved_min,
        "reason": reason.strip(),
        "model_note": model_note,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("ClearPath :: Recommender self-test")
    print("=" * 60)
    import json
    for args in [
        ("Mysore Road", "accident", 19, 1, False, True),
        ("Hosur Road", "vehicle_breakdown", 9, 2, False, False),
        ("Non-corridor", "pot_holes", 14, 6, True, False),
    ]:
        print(f"\nrecommend{args} ->")
        print(json.dumps(recommend(*args), indent=2))
