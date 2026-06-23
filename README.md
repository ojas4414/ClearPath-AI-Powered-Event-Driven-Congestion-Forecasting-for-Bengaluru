# 🚦 ClearPath — Event-Driven Congestion Forecasting

**Flipkart Gridlock Hackathon · Round 2 · Theme 2**

ClearPath turns Bengaluru's raw Astram road-event feed into live, actionable traffic-control
decisions. It answers the two questions a control room actually has during an incident — *how
severe is this event?* (an **XGBoost** classifier predicts High/Low priority) and *how busy is
this corridor about to get?* (a per-corridor **LSTM** forecasts next-hour event volume). A
recommender fuses both into a single **0–100 impact score** and converts it into concrete orders:
how many officers to send, whether to barricade, and which pre-approved diversion to open.

Three capabilities make it answer the *event-driven* brief end-to-end:

- **Plan an Event** — describe a specific event (venue, date, time, expected crowd, type: rally /
  festival / match / VIP) and ClearPath forecasts which corridors it will hit (by real geographic
  proximity), sizes the manpower to the crowd (**1 officer / 2,000 attendees**), orders barricades
  and diversions, and lights up a live map. *(`events.py`, `POST /plan-event`)*
- **Post-event learning loop** — every resolved event's *actual* clear-time is compared to the
  prediction and an online calibration corrects the next one; the mean error visibly shrinks over
  a chronological replay of real Astram data. *(`learning.py`, `/learning-status`, `/event-feedback`)*
- **Honest LSTM routing** — each corridor is served by the LSTM *only* where it actually beat the
  persistence baseline on held-out AUROC (**11/23 corridors**); the rest fall back to a climatology
  baseline, so a forecaster proven worse than doing nothing never reaches an operator.

The result is served through a FastAPI backend with a live WebSocket alert feed and a single-file
dark dashboard (Leaflet map + Chart.js from CDN) — so an operator sees the city's emerging hotspots
*before* they gridlock.

---

## How to run

**Local (zero infra — uses the stdlib SQLite fallback):**
```bash
pip install -r requirements.txt                      # 1. install deps
python data_processor.py && python models.py         # 2. build features + train XGBoost & LSTMs
python main.py                                        # 3. serve API + dashboard
```

**Containerised (app + Postgres, production-style):**
```bash
docker compose up --build                             # one command: FastAPI app + Postgres
```

Either way, open **http://localhost:8000** for the dashboard (API docs at **/docs**).

## Architecture choices (and one we deliberately rejected)

- **Docker + docker-compose** — the app and a real **Postgres** database come up with one command.
- **Postgres for the *operational* data only** — live-reported incidents and post-event feedback
  live in `db.py`, which talks to Postgres when `DATABASE_URL` is set and falls back to a stdlib
  **SQLite** file locally, so `python main.py` needs no database running. The 8k static training
  CSV stays a file — a DB would add nothing there.
- **Live ingestion path** — `POST /event` reports an incident *now*: ClearPath scores it, persists
  it, and broadcasts it to every open dashboard's alert feed instantly over WebSocket. This is what
  makes it an operational system, not a static analysis.
- **We did NOT rewrite the hot path in C++.** Inference is already sub-100ms on tiny XGBoost/LSTM
  models — there is no compute bottleneck to fix. C++ would have added large risk for zero user-
  visible speed-up; engineering effort went into the live data path instead, which is where the
  real value (and the real gap) was.

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
   ┌──────────────────┐   /predict /plan-event /optimize /learning-status
   │     FastAPI      │   + POST /event (live ingestion) + WebSocket hub
   │                  │   (main.py)
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   live incidents + feedback log
   │  Postgres /      │   (Postgres in Docker, SQLite locally)
   │  SQLite (db.py)  │
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   quick-start · plan-an-event map · risk heatmap ·
   │    Dashboard     │   live alert feed · learning curve   (index.html)
   └──────────────────┘
```

---

## Components

| File | Role |
|------|------|
| `data_processor.py` | Loads CSV, engineers features, builds hourly timeseries, saves `processed_events.csv` |
| `models.py` | Trains **XGBoost** severity classifier + per-corridor **LSTM** forecasters |
| `recommender.py` | Fuses both models into an impact score and a resourcing decision; routes each corridor to the LSTM or a climatology fallback per `corridor_routing.json` |
| `events.py` | **Planned-event engine**: venue → affected corridors (proximity) → crowd-sized manpower, barricades, diversions |
| `learning.py` | **Post-event learning loop**: replays real resolution times, online-calibrates the prediction, exposes the shrinking-error curve |
| `db.py` | **Operational data store**: live incidents + feedback log; Postgres when `DATABASE_URL` is set, stdlib SQLite fallback otherwise |
| `main.py` | FastAPI service: REST endpoints, **live `POST /event` ingestion**, WebSocket broadcast hub |
| `Dockerfile` / `docker-compose.yml` | Containerised app + Postgres (`docker compose up --build`) |
| `index.html` | Single-file dark dashboard (Leaflet + Chart.js from CDN) |

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
