from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import predict_heat_risk
from config import CORS_ORIGINS

api = FastAPI(title="Urban Heat Risk API", version="1.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HeatInput(BaseModel):
    month: int = Field(..., ge=1, le=12)
    impervious_pct: float = Field(..., ge=0, le=100)
    tree_canopy_pct: float = Field(..., ge=0, le=70)
    ndvi: float = Field(..., ge=0, le=0.8)
    pop_density_km2: float = Field(..., ge=200, le=60000)
    building_height_m: float = Field(..., ge=3, le=60)
    albedo: float = Field(..., ge=0.08, le=0.35)
    air_temp_c: float = Field(..., ge=-5, le=45)
    humidity_pct: float = Field(..., ge=15, le=95)
    wind_speed_ms: float = Field(..., ge=0.3, le=12)


class HeatOutput(BaseModel):
    heat_risk: str
    probabilities: dict[str, float]


@api.get("/health")
def health():
    return {"status": "ok"}


@api.post("/predict", response_model=HeatOutput)
def predict(item: HeatInput):
    result, probs = predict_heat_risk(
        item.month, item.impervious_pct, item.tree_canopy_pct, item.ndvi,
        item.pop_density_km2, item.building_height_m, item.albedo,
        item.air_temp_c, item.humidity_pct, item.wind_speed_ms,
    )
    return {"heat_risk": str(result),
            "probabilities": {str(k): float(v) for k, v in probs.items()}}
