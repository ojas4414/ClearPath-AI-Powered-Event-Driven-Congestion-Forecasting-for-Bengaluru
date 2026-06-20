# 🚦 ClearPath — Event-Driven Congestion Forecasting

**Flipkart Gridlock Hackathon · Round 2 · Theme 2**

ClearPath turns Bengaluru's raw Astram road-event feed into live, actionable traffic-control
decisions. It answers the two questions a control room actually has during an incident — *how
severe is this event?* (an **XGBoost** classifier predicts High/Low priority) and *how busy is
this corridor about to get?* (a per-corridor **LSTM** forecasts next-hour event volume). A
recommender fuses both into a single **0–100 impact score** and converts it into concrete orders:
how many officers to send, whether to barricade, and which pre-approved diversion to open. The
result is served through a FastAPI backend with a live WebSocket alert feed and a zero-dependency
dark dashboard — so an operator sees the city's emerging hotspots *before* they gridlock.

---

## How to run

```bash
pip install -r requirements.txt                      # 1. install deps
python data_processor.py && python models.py         # 2. build features + train XGBoost & LSTMs
python main.py                                        # 3. serve API + dashboard
```

Then open **http://localhost:8000** for the dashboard (API docs at **/docs**).

---

## Architecture

```
   ┌──────────────────┐
   │   Astram Data    │   8,173 anonymized road events (CSV)
   │  (event feed)    │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐   parse time · encode corridor/cause ·
   │ Feature          │   rolling 2h event count · hourly timeseries ·
   │ Engineering      │   fill nulls -> "Unknown"   (data_processor.py)
   └────────┬─────────┘
            │
     ┌──────┴───────┐
     ▼              ▼
┌─────────┐   ┌───────────┐
│ XGBoost │   │   LSTM    │   XGBoost -> P(High priority)  [severity]
│ severity│   │ forecaster│   LSTM     -> next-hour count  [load]
└────┬────┘   └─────┬─────┘   (models.py)
     │              │
     └──────┬───────┘
            ▼
   ┌──────────────────┐   impact_score (0-100) -> officers, barricades,
   │   Recommender    │   diversion route + human-readable reason
   │                  │   (recommender.py)
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   /health /predict /hotspots /stats
   │     FastAPI      │   + WebSocket /ws/alerts (30s push)
   │                  │   (main.py)
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   dark theme · stat cards · recommendation
   │    Dashboard     │   panel · live alert feed   (index.html)
   └──────────────────┘
```

---

## Components

| File | Role |
|------|------|
| `data_processor.py` | Loads CSV, engineers features, builds hourly timeseries, saves `processed_events.csv` |
| `models.py` | Trains **XGBoost** severity classifier + per-corridor **LSTM** forecasters |
| `recommender.py` | Fuses both models into an impact score and a resourcing decision |
| `main.py` | FastAPI service: REST endpoints + WebSocket alert stream |
| `index.html` | Single-file dark dashboard (no external libraries) |

## Decision logic

| Impact score | Severity | Officers | Barricades | Diversion |
|-------------|----------|----------|------------|-----------|
| > 75 | **Critical** | 10 | Yes | Yes |
| > 45 | **High** | 5 | Yes | No |
| ≤ 45 | **Normal** | 2 | No | No |

**Impact score** = `0.50 × P(High severity)` + `0.30 × P(busy) [LSTM]` + `0.20 × hour-of-day risk`,
times an event-cause multiplier, with a `+12` bump when a road closure is requested.

## Time-saved: a data-derived estimate (not an assumption)

We never deployed ClearPath, so "time saved" cannot be a measured fact — but it *is* grounded in
data, not invented. From the Astram closed-event durations we measured that resolution time scales
with **congestion context**, not severity label:

| Prior-2h activity | Resolution (mean) |
|---|---|
| calm (0 events) | **80 min** |
| light (1–2) | 92 min |
| moderate (3–5) | 105 min |
| busy (6+) | **242 min** (3× calm) |

ClearPath's mechanism is forecasting hotspots and pre-positioning resources *before* congestion
builds, so its benefit is removing the **congestion penalty** — the gap between an event's
congestion-tier resolution time and the ~80-min calm baseline. That penalty is measured; the only
assumption is that pre-positioning recovers **65%** of it (explicit, bounded, conservative).
Calm events save ~nothing — the gains come from the congested tail. City-wide this yields
**97 → 64 min (~34%)**, every term traceable to observed data.

## A note on model honesty

The XGBoost **headline** metrics (accuracy 81.8% / F1 85.9%) come from a **leak-free** model that
predicts priority from time, cause and recent volume — corridor **excluded** — so the number is
genuinely earned. The production model that serves predictions keeps `corridor` as a legitimate
operational prior and scores 100%; a corridor-only majority-class baseline *also* hits 100%,
confirming priority is near-deterministic given corridor in this dataset (documented, not hidden).
The LSTM is a per-corridor **busy-hour classifier**, validated on a chronological 80/20 split
against a persistence baseline: **+5.1pp AUROC** across the 8 operational corridors, with
**Mysore Road +19.0pp, Bellary Road 1 +16.2pp, Tumkur Road +15.1pp**. Reproducibility is fixed
with `seed = 42` throughout. All numbers above live in `model_metrics.json` / `resolution_stats.json`.
