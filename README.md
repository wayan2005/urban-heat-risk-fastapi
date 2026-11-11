# 🌡️ Urban Heat Risk Prediction

Classifies a neighbourhood's heat risk as **Low / Moderate / High** using Logistic Regression.

**Note:** the dataset (`data/synthetic_urban_heat_risk.csv`) is SYNTHETIC. It was generated
by `data/generate_synthetic_heat_data.py` from assumed relationships (more pavement and heat
-> hotter, more trees and wind -> cooler) plus random noise. It is not real measured data.

## Files
- `data/` - synthetic dataset + the script that generated it
- `train.py` - trains and saves the model
- `app.py` - loads the model and predicts
- `frontend.py` - Streamlit web page
- `test.py` - quick prediction test

## Run
```bash
pip install -r requirements.txt
python train.py
python test.py
streamlit run frontend.py
```

## Labels
Heat score (`heat_anomaly_c`): below 1.9 = Low, 1.9 to 3.8 = Moderate, 3.8 or more = High.

## API (FastAPI)
```bash
uvicorn api:api --reload          # API at http://localhost:8000  (docs at /docs)
API_URL=http://localhost:8000 streamlit run frontend.py
docker compose up --build         # runs API + frontend together
```
Endpoints: `GET /health`, `POST /predict`.

## Config (env vars - change these to switch cloud platform, no code changes)
- `PORT` - port the API listens on (set automatically by most platforms)
- `API_URL` - where the frontend finds the API
- `CORS_ORIGINS` - comma-separated allowed frontend URLs (default `*`)
- `MODEL_PATH` - path to the model file
