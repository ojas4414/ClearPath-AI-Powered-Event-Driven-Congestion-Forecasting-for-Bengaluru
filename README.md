<div align="center">

# 🚦 ClearPath

### Event-Driven Congestion Forecasting & Resource Deployment for Bengaluru

*See the city's hotspots **before** they gridlock — and act on them.*

<p>
  <img alt="Python"      src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="FastAPI"     src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img alt="PyTorch"     src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="XGBoost"     src="https://img.shields.io/badge/XGBoost-1A7DC0?style=for-the-badge&logo=xgboost&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white">
</p>
<p>
  <img alt="Docker"      src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white">
  <img alt="PostgreSQL"  src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white">
  <img alt="Leaflet"     src="https://img.shields.io/badge/Leaflet-199900?style=for-the-badge&logo=leaflet&logoColor=white">
  <img alt="Chart.js"    src="https://img.shields.io/badge/Chart.js-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white">
</p>
<p>
  <img alt="Hackathon" src="https://img.shields.io/badge/Flipkart-Gridlock%202.0%20·%20Round%202-2874F0?style=flat-square">
  <img alt="Theme"     src="https://img.shields.io/badge/Theme-Event--Driven%20Congestion-orange?style=flat-square">
  <img alt="Impact"    src="https://img.shields.io/badge/Response%20time-↓%2034%25-success?style=flat-square">
</p>

</div>

---

ClearPath is a **real-time AI decision system** that converts Bengaluru's raw road-event feed
(**Astram BTP dataset**) into actionable traffic-control commands — *officers to deploy, barricades
to place, diversions to activate* — in milliseconds. It answers the two questions a control room
actually has during an incident:

> **How severe is this event?** &nbsp;→&nbsp; an **XGBoost** classifier predicts High/Low priority
> **How busy is this corridor about to get?** &nbsp;→&nbsp; a per-corridor **LSTM** forecasts load

A recommender fuses both into a single **0–100 impact score** and maps it to concrete orders:
how many officers to send, whether to barricade, and which pre-approved diversion to open.

## ✨ What makes it answer the *event-driven* brief end-to-end

| | Feature | What it does |
|---|---|---|
| 🎪 | **Plan an Event** | Describe a real event (venue, date, time, crowd, type) → forecasts the corridors it will hit by **geographic proximity**, sizes manpower to the crowd (**1 officer / 2,000 attendees**), opens diversions, and **lights up a live map**. |
| 🧠 | **Post-Event Learning** | Every event's *actual* clear-time recalibrates the next prediction — mean error shrinks **111 → 102 min** over 2,464 real events. *(closes the brief's "no learning system" gap)* |
| 🚨 | **Live Incident Reporting** | `POST /event` reports an incident **now** → scored, saved to the DB, and **broadcast to every dashboard instantly** over WebSocket (🔴 LIVE feed). |
| ✅ | **Honest LSTM routing** | Each corridor uses the LSTM *only* where it beat a persistence baseline on held-out data (**11/23**); the rest fall back to climatology — *a model proven worse than nothing never reaches an operator.* |

---

## 🚀 How to run

> **Recommended — Docker** (ships its own Python, dependencies & Postgres; runs anywhere)
> *Prerequisite: Docker Desktop installed and running.*
> ```bash
> docker compose up --build          # app + Postgres, one command
> ```
> Then open **http://localhost:8000**  ·  if 8000 is busy: `HOST_PORT=8090 docker compose up --build`

> **Fallback — plain Python** (no Docker needed) · *Python 3.10+*
> ```bash
> pip install -r requirements.txt    # 1. install deps
> python main.py                     # 2. serve API + dashboard  (trained models are included)
> ```
> Open **http://localhost:8000** (API docs at **/docs**). *Optional retrain:* `python data_processor.py && python models.py`

---

## 🧱 Architecture

```
   ┌──────────────────┐
   │   Astram Data    │   8,057 anonymized road events (23 corridors)
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   parse time · encode corridor/cause ·
   │ Feature Eng.     │   rolling 2h event count · hourly timeseries
   └────────┬─────────┘   (data_processor.py)
     ┌──────┴───────┐
     ▼              ▼
┌─────────┐   ┌───────────┐
│ XGBoost │   │   LSTM    │   XGBoost → P(High priority)   [severity]
│ severity│   │ per-road  │   LSTM    → P(busy next hour)  [load]
└────┬────┘   └─────┬─────┘   (models.py)
     └──────┬───────┘
            ▼
   ┌──────────────────┐   impact_score (0–100) → officers, barricades,
   │   Recommender    │   diversion route + human-readable reason
   └────────┬─────────┘   (recommender.py)
            ▼
   ┌──────────────────┐   /predict /plan-event /optimize /learning-status
   │     FastAPI      │   + POST /event (live ingestion) + WebSocket hub
   └────────┬─────────┘   (main.py)
            ▼
   ┌──────────────────┐   live incidents + feedback log
   │ Postgres /SQLite │   (Postgres in Docker · SQLite locally)  (db.py)
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   plan-an-event map · risk heatmap ·
   │    Dashboard     │   live alert feed · learning curve   (index.html)
   └──────────────────┘
```

### 🧩 Components

| File | Role |
|------|------|
| `data_processor.py` | Loads CSV, engineers features, builds hourly timeseries |
| `models.py` | Trains **XGBoost** severity classifier + per-corridor **LSTM** forecasters |
| `recommender.py` | Fuses both models into an impact score + resourcing decision; routes each corridor to the LSTM or a climatology fallback |
| `events.py` | 🎪 **Planned-event engine** — venue → affected corridors → crowd-sized manpower, barricades, diversions |
| `learning.py` | 🧠 **Post-event learning loop** — replays real resolution times, online-calibrates, exposes the shrinking-error curve |
| `db.py` | 🗄️ **Operational store** — live incidents + feedback; Postgres (`DATABASE_URL`) or SQLite fallback |
| `main.py` | FastAPI service: REST + **live `POST /event` ingestion** + WebSocket broadcast hub |
| `Dockerfile` · `docker-compose.yml` | Containerised app + Postgres |
| `index.html` | Single-file dark dashboard (Leaflet + Chart.js, no build step) |

---

## ⚙️ Decision logic

| Impact score | Severity | Officers | Barricades | Diversion |
|:-----------:|:--------:|:--------:|:----------:|:---------:|
| **> 75** | 🔴 Critical | 10 | ✅ | ✅ |
| **> 45** | 🟠 High | 5 | ✅ | — |
| **≤ 45** | 🟢 Normal | 2 | — | — |

**Impact score** = `0.50 × P(High severity)` + `0.30 × P(busy) [LSTM]` + `0.20 × hour-of-day risk`,
× an event-cause multiplier, with a `+12` bump when a road closure is requested.

---

## 📈 Impact — a data-derived estimate (not an assumption)

From the Astram closed-event durations, resolution time scales with **congestion context**, not the
severity label:

| Prior-2h activity | Mean resolution |
|---|:---:|
| calm (0 events) | **80 min** |
| light (1–2) | 92 min |
| moderate (3–5) | 105 min |
| busy (6+) | **242 min** *(3× calm)* |

ClearPath's value is **removing the congestion penalty** by pre-positioning resources *before* it
builds. That penalty is **measured**; the only assumption is a conservative **65% recovery**.
Across 2,464 real events this yields:

<div align="center">

### ⏱️ 97.2 min → 63.8 min &nbsp;·&nbsp; **33.4 min saved (~34% faster)**

</div>

Calm events save ~nothing (honest) — the gains come from the congested tail.

---

## 🔬 A note on model honesty

- **XGBoost headline: 81.8% accuracy · 85.9% F1 · 90% recall** — from a **leak-free** model that
  *excludes corridor*, so it genuinely learns from time, cause & recent volume. We *disclose* that
  the production model (with corridor) scores 100% because priority is near-deterministic given
  corridor in this dataset — documented, not hidden.
- **LSTM** is a per-corridor **busy-hour classifier**, validated on a chronological 80/20 split vs a
  persistence baseline: **+5.1pp AUROC** across the 8 operational corridors — **Mysore +19.0pp ·
  Bellary Rd 1 +16.2pp · Tumkur +15.1pp**. Served only where it wins (**11/23 corridors**).
- Reproducible with `seed = 42` throughout. All numbers live in `model_metrics.json` /
  `resolution_stats.json`.

> **Engineering choice we rejected:** rewriting the hot path in C++ "for speed." Inference is already
> sub-100ms — there's no compute bottleneck. Effort went into the live-data path instead, where the
> real value was.

<div align="center">

**Flipkart Gridlock 2.0 · Round 2 · Theme 2 — Event-Driven Congestion**

</div>
