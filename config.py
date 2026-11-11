"""All settings come from environment variables, so switching cloud
platform = changing env vars, not code."""
import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/heat_risk_model.pkl")
# Comma-separated list of allowed frontend origins, e.g. "https://myapp.onrender.com"
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
# Used by the Streamlit frontend to find the API
API_URL = os.getenv("API_URL", "http://localhost:8000")
