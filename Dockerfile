# ClearPath — containerised FastAPI service.
# Python 3.11 (stable wheels for torch/xgboost/psycopg2) rather than the host's 3.14.
FROM python:3.11-slim

WORKDIR /app

# Install deps first so the layer caches across code changes.
# psycopg2-binary is added here (not in requirements.txt) because it is only needed in the
# container's Postgres path — locally the app runs on the stdlib SQLite fallback, no extra dep.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt psycopg2-binary

# Copy the app + pre-trained model artifacts (xgb_model.pkl, lstm_models/, encoders, processed
# CSVs, metrics, routing). They're committed, so the image is ready to serve with no training step.
COPY . .

EXPOSE 8000
ENV PORT=8000

# main.py respects $PORT and $DATABASE_URL (set by docker-compose to point at Postgres).
CMD ["python", "main.py"]
