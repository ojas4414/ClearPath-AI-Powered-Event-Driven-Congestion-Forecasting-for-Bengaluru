"""
ClearPath :: models.py
======================
Trains the two predictive engines behind ClearPath:

  Model A  XGBoost classifier  -> "Is this event HIGH priority?"  (severity)
  Model B  LSTM forecaster     -> "How many events will hit this corridor next hour?" (load)

WHY two models:
Severity and volume are different questions. A single breakdown can be high-severity even on
a quiet corridor; conversely a corridor can be about to flood with low-severity events. ClearPath
fuses both signals in the recommender, so each is trained as a specialist here.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
from xgboost import XGBClassifier

import data_processor as dp

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XGB_PATH = os.path.join(BASE_DIR, "xgb_model.pkl")
LSTM_DIR = os.path.join(BASE_DIR, "lstm_models")
HOURLY_CSV = os.path.join(BASE_DIR, "hourly_corridor_counts.csv")
PROCESSED_CSV = os.path.join(BASE_DIR, "processed_events.csv")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "corridor_encoded", "event_cause_encoded",
    "requires_road_closure", "events_in_corridor_last_2hrs",
]
SEQ_LEN = 6  # six hours of history -> predict the seventh


# --------------------------------------------------------------------------------------
# Model A — XGBoost severity classifier
# --------------------------------------------------------------------------------------
def train_xgboost(df=None):
    """
    WHAT: Train an XGBoost binary classifier to predict priority (High=1/Low=0) and report
    accuracy / precision / recall / F1 on a held-out 20% test split.
    WHY XGBoost: the features are heterogeneous, mostly categorical-numeric, with non-linear
    interactions (hour x corridor x cause). Gradient-boosted trees handle that natively, need
    no scaling, are fast to train on ~8k rows, and give a calibrated probability we reuse as
    the severity component of the impact score.
    """
    print("\n----- Model A: XGBoost severity classifier -----")
    if df is None:
        df = pd.read_csv(PROCESSED_CSV)

    X = df[FEATURES]
    y = df["target"]
    # stratify keeps the High/Low ratio identical in train and test -> trustworthy metrics.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    print(f"[xgb] train={len(X_train)}  test={len(X_test)}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])  # [[TN, FP], [FN, TP]]
    print(f"[xgb] accuracy = {acc:.4f}")
    print(f"[xgb] precision= {prec:.4f}")
    print(f"[xgb] recall   = {rec:.4f}")
    print(f"[xgb] f1       = {f1:.4f}")

    # Persist the model AND the feature order so inference cannot silently misalign columns.
    joblib.dump({"model": model, "features": FEATURES}, XGB_PATH)
    print(f"[xgb] saved -> {XGB_PATH}")

    importance = dict(sorted(
        zip(FEATURES, (float(x) for x in model.feature_importances_)),
        key=lambda t: -t[1],
    ))
    print("[xgb] feature importances:")
    for f, imp in importance.items():
        print(f"        {f:<32} {imp:.3f}")

    metrics = {
        "xgb_accuracy": round(float(acc), 4),
        "xgb_precision": round(float(prec), 4),
        "xgb_recall": round(float(rec), 4),
        "xgb_f1": round(float(f1), 4),
        "feature_importance": {k: round(v, 4) for k, v in importance.items()},
        "confusion_matrix": cm.tolist(),
    }
    return model, metrics


# --------------------------------------------------------------------------------------
# Model B — LSTM per-corridor load forecaster
# --------------------------------------------------------------------------------------
class CorridorLSTM(nn.Module):
    """
    WHAT: A small 2-layer LSTM that maps a sequence of 6 hourly counts -> next-hour count.
    WHY LSTM: hourly event load is a temporal signal with short-term momentum (a busy hour
    tends to be followed by a busy hour). An LSTM captures that autocorrelation far better than
    a static regressor. We keep it tiny (hidden=32) because each corridor has limited history —
    a bigger net would overfit.
    """

    def __init__(self, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1). Take the final time-step's hidden state -> linear head.
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _make_sequences(series, seq_len=SEQ_LEN):
    """
    WHAT: Slide a window over a 1-D count series to build (X=6 steps, y=next step) pairs.
    WHY: Supervised forecasting needs explicit input/target windows. Sliding maximises the
    number of training examples we can squeeze from each corridor's short history.
    """
    xs, ys = [], []
    for i in range(len(series) - seq_len):
        xs.append(series[i:i + seq_len])
        ys.append(series[i + seq_len])
    if not xs:
        return None, None
    X = np.array(xs, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(ys, dtype=np.float32).reshape(-1, 1)
    return X, y


def train_lstm(min_hours=60, epochs=40, val_pct=0.20):
    """
    WHAT: Train one LSTM per corridor on its hourly count series; save to lstm_models/{corridor}.pt.
    Uses a chronological 80/20 train/val split so the reported MAE is a genuine held-out metric,
    not training loss. Also reports MAE against a naive last-value baseline so the improvement
    is meaningful and verifiable, not just a loss number.
    WHY per-corridor: each corridor has its own rhythm and base volume. A dedicated model per
    corridor learns that local pattern instead of being averaged into a generic city-wide curve.
    WHY chronological split: shuffling a time series leaks future information into training.
    """
    print("\n----- Model B: LSTM per-corridor forecaster -----")
    os.makedirs(LSTM_DIR, exist_ok=True)
    hourly = pd.read_csv(HOURLY_CSV, parse_dates=["hour_bucket"])

    trained, skipped, final_losses = 0, 0, []
    all_mae, all_naive_mae = [], []

    for corridor, g in hourly.groupby("corridor"):
        g = g.sort_values("hour_bucket")
        series = g["event_count"].to_numpy(dtype=np.float32)
        if len(series) < min_hours:
            skipped += 1
            continue

        # Chronological 80/20 split — never shuffle a time series.
        split = int(len(series) * (1 - val_pct))
        train_series = series[:split]
        val_series   = series[split:]

        scale = float(train_series.max()) or 1.0
        norm_train = train_series / scale
        X, y = _make_sequences(norm_train)
        if X is None:
            skipped += 1
            continue

        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        model = CorridorLSTM()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(model(Xt), yt)
            loss.backward()
            opt.step()

        # Held-out evaluation: rolling one-step-ahead forecast on val set.
        # Seed the LSTM with the last SEQ_LEN training points, then advance
        # using actual observations (teacher-forcing) so errors don't compound.
        model.eval()
        if len(val_series) > 0:
            seed = norm_train[-SEQ_LEN:].copy()
            preds, targets = [], []
            for actual in val_series:
                x = torch.from_numpy(seed.reshape(1, SEQ_LEN, 1).astype(np.float32))
                with torch.no_grad():
                    pred = model(x).item() * scale
                preds.append(pred)
                targets.append(float(actual))
                seed = np.append(seed[1:], actual / scale)  # shift window

            mae = float(np.mean(np.abs(np.array(preds) - np.array(targets))))
            # Naive baseline: always predict the last observed training value.
            naive_mae = float(np.mean(np.abs(train_series[-1] - np.array(targets))))
            all_mae.append(mae)
            all_naive_mae.append(naive_mae)
            val_str = f"val_MAE={mae:.3f} naive_MAE={naive_mae:.3f}"
        else:
            val_str = "val=n/a"

        safe = corridor.replace("/", "_").replace(" ", "_")
        path = os.path.join(LSTM_DIR, f"{safe}.pt")
        torch.save({"state_dict": model.state_dict(), "scale": scale,
                    "corridor": corridor, "seq_len": SEQ_LEN}, path)
        trained += 1
        final_losses.append(float(loss.item()))
        print(f"[lstm] {corridor:<22} hours={len(series):<5} "
              f"train_loss={loss.item():.4f}  {val_str}")

    avg_final_loss = round(float(np.mean(final_losses)), 4) if final_losses else 0.0
    avg_mae        = round(float(np.mean(all_mae)),       4) if all_mae else 0.0
    avg_naive_mae  = round(float(np.mean(all_naive_mae)), 4) if all_naive_mae else 0.0
    improvement    = round(100.0 * (avg_naive_mae - avg_mae) / avg_naive_mae, 1) if avg_naive_mae else 0.0

    print(f"\n[lstm] trained={trained}  skipped={skipped} (insufficient history)")
    print(f"[lstm] Held-out val MAE  (LSTM):  {avg_mae:.4f} events/hr")
    print(f"[lstm] Held-out val MAE  (naive): {avg_naive_mae:.4f} events/hr")
    print(f"[lstm] LSTM beats naive by        {improvement:.1f}%")

    return {
        "lstm_final_loss":                avg_final_loss,
        "lstm_corridors_trained":         trained,
        "lstm_val_mae":                   avg_mae,
        "lstm_naive_baseline_mae":        avg_naive_mae,
        "lstm_improvement_over_naive_pct": improvement,
    }


def load_lstm(corridor):
    """
    WHAT: Load a trained per-corridor LSTM (returns model + scale) or None if absent.
    WHY: Shared loader used by the recommender so forecasting logic lives in one place.
    """
    safe = corridor.replace("/", "_").replace(" ", "_")
    path = os.path.join(LSTM_DIR, f"{safe}.pt")
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CorridorLSTM()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {"model": model, "scale": ckpt["scale"], "seq_len": ckpt["seq_len"]}


if __name__ == "__main__":
    print("=" * 60)
    print("ClearPath :: Model Training")
    print("=" * 60)
    # Ensure processed artefacts exist; rebuild if missing.
    if not os.path.exists(PROCESSED_CSV) or not os.path.exists(HOURLY_CSV):
        print("[setup] processed data missing -> running data_processor first")
        dp.process()
    _, xgb_metrics = train_xgboost()
    lstm_metrics = train_lstm()

    metrics = {**xgb_metrics, **lstm_metrics}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[metrics] saved -> {METRICS_PATH}")
    print("[done] models.py complete.")
