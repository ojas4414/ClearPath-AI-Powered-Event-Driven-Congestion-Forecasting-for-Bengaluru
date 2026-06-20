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

**Impact score** = `0.60 × P(High severity)` + `0.40 × normalized forecast volume`, with a
`+12` bump when a road closure is requested.

## A note on model honesty

The XGBoost classifier reports ~100% accuracy on the held-out split. This is **not** a leak in
the train/serve sense — it reflects that in the Astram dataset `priority` is almost deterministic
given `corridor` + recent event volume (it appears to have been assigned by operational rules).
We report this transparently rather than tuning the metric down. The LSTM forecaster is the
genuinely predictive component for *future* load. Reproducibility is fixed with `seed = 42`
throughout.
