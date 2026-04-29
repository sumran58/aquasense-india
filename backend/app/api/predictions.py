from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import pandas as pd
import time

from app.utils.database import get_db, GroundwaterReading, DistrictPrediction, DistrictMeta
from app.utils.config import classify_risk, RISK_META
from app.ml.pipeline import AquaSensePredictor, engineer_features, FEATURE_COLUMNS

router = APIRouter()
_predictor = None
_cache = {}
CACHE_TTL = 300

def cache_get(key):
    if key in _cache:
        val, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return val
    return None

def cache_set(key, val):
    _cache[key] = (val, time.time())

def cache_clear():
    _cache.clear()

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = AquaSensePredictor()
    return _predictor

class PredictRequest(BaseModel):
    state: str
    district: str
    current_level_mbgl: float
    rainfall_mm_last_quarter: float = 80.0
    population_density: float = 400.0
    agricultural_area_pct: float = 55.0
    irrigation_wells_per_km2: float = 4.0
    ndvi_mean: float = 0.3
    quarters_ahead: int = Field(default=4, ge=1, le=8)

class PredictionOut(BaseModel):
    district: str
    state: str
    current_level_mbgl: float
    overall_risk: str
    risk_color: str
    summary: str
    predictions: List[dict]


def build_features_from_history(district: str, db: Session, user_overrides: dict = None):
    """
    Build ML features using the EXACT SAME method as training.
    
    Instead of manually constructing lags/rolling stats (which caused all the bugs),
    we pull the district's full history, run it through engineer_features() — the same
    function used during training — and take the last row's features.
    
    This guarantees feature parity between training and inference.
    """
    
    # Pull ALL readings for this district (same as training data)
    readings = (db.query(GroundwaterReading)
        .filter(GroundwaterReading.district == district)
        .order_by(GroundwaterReading.year, GroundwaterReading.quarter)
        .all())
    
    if not readings:
        return None
    
    # Build a DataFrame exactly like the training pipeline expects
    rows = []
    for r in readings:
        rows.append({
            "district": r.district,
            "state": r.state,
            "year": r.year,
            "quarter": r.quarter,
            "water_level_mbgl": r.water_level_mbgl,
            "rainfall_mm": r.rainfall_mm if r.rainfall_mm else 0.0,
            "latitude": r.latitude,
            "longitude": r.longitude,
            "population_density": user_overrides.get("population_density", 0.0) if user_overrides else 0.0,
            "agricultural_area_pct": user_overrides.get("agricultural_area_pct", 0.0) if user_overrides else 0.0,
            "irrigation_wells_per_km2": user_overrides.get("irrigation_wells_per_km2", 0.0) if user_overrides else 0.0,
            "ndvi_mean": user_overrides.get("ndvi_mean", 0.0) if user_overrides else 0.0,
        })
    
    df = pd.DataFrame(rows)
    
    # Run through the SAME engineer_features() used during training
    df_feat = engineer_features(df)
    
    # Take the last row — this has all correctly computed lags, rolling stats, etc.
    last_row = df_feat.iloc[-1]
    
    # Extract features as dict
    features = {}
    for col in FEATURE_COLUMNS:
        if col in last_row.index:
            val = last_row[col]
            features[col] = float(val) if pd.notna(val) else 0.0
        else:
            features[col] = 0.0
    
    # Apply user overrides for non-historical features
    if user_overrides:
        if "rainfall_mm" in user_overrides:
            features["rainfall_mm"] = user_overrides["rainfall_mm"]
        if "population_density" in user_overrides:
            features["population_density"] = user_overrides["population_density"]
        if "agricultural_area_pct" in user_overrides:
            features["agricultural_area_pct"] = user_overrides["agricultural_area_pct"]
        if "irrigation_wells_per_km2" in user_overrides:
            features["irrigation_wells_per_km2"] = user_overrides["irrigation_wells_per_km2"]
        if "ndvi_mean" in user_overrides:
            features["ndvi_mean"] = user_overrides["ndvi_mean"]
    
    # If user provided a different current level, update lag_1q
    if user_overrides and "current_level_mbgl" in user_overrides:
        db_latest = last_row["water_level_mbgl"] if "water_level_mbgl" in last_row.index else features["level_lag_1q"]
        user_level = user_overrides["current_level_mbgl"]
        if abs(user_level - db_latest) > 0.01:
            features["level_lag_1q"] = user_level
    
    return features


@router.post("/district", response_model=PredictionOut)
async def predict_district(req: PredictRequest, db: Session = Depends(get_db)):
    predictor = get_predictor()
    if not predictor.is_trained():
        raise HTTPException(503, "Model not trained.")

    # Build features using the SAME function as training
    features = build_features_from_history(
        district=req.district,
        db=db,
        user_overrides={
            "current_level_mbgl": req.current_level_mbgl,
            "rainfall_mm": req.rainfall_mm_last_quarter,
            "population_density": req.population_density,
            "agricultural_area_pct": req.agricultural_area_pct,
            "irrigation_wells_per_km2": req.irrigation_wells_per_km2,
            "ndvi_mean": req.ndvi_mean,
        }
    )
    
    if features is None:
        raise HTTPException(404, f"No data found for {req.district}")

    preds = predictor.predict_district(features, quarters_ahead=req.quarters_ahead)
    worst = max(preds, key=lambda p: RISK_META[p["risk_level"]]["priority"])
    current = features["level_lag_1q"]
    delta = preds[-1]["predicted_level_mbgl"] - current
    summary = (f"{req.district} will {'deepen' if delta > 0 else 'recover'} by "
               f"{abs(delta):.1f} mbgl over {req.quarters_ahead} quarters. "
               f"Risk: {worst['risk_label']}.")

    return PredictionOut(
        district=req.district, state=req.state,
        current_level_mbgl=current,
        overall_risk=worst["risk_level"],
        risk_color=RISK_META[worst["risk_level"]]["color"],
        summary=summary, predictions=preds)


@router.get("/stats")
async def prediction_stats(db: Session = Depends(get_db)):
    predictor = get_predictor()
    cached = cache_get("stats")
    if cached:
        cached["model_trained"] = predictor.is_trained()
        return cached

    try:
        row = db.execute(text("""
            SELECT
                COUNT(*) as districts,
                AVG(avg_level) as avg_level,
                MAX(avg_level) as max_level,
                SUM(CASE WHEN avg_level > 20 THEN 1 ELSE 0 END) as critical,
                SUM(CASE WHEN avg_level > 12 AND avg_level <= 20 THEN 1 ELSE 0 END) as warning,
                SUM(CASE WHEN avg_level >= 5 AND avg_level <= 12 THEN 1 ELSE 0 END) as stable,
                SUM(CASE WHEN avg_level < 5 THEN 1 ELSE 0 END) as recovering
            FROM (
                SELECT district, AVG(water_level_mbgl) as avg_level
                FROM groundwater_readings
                GROUP BY district
            )
        """)).fetchone()
    except Exception as e:
        raise HTTPException(500, f"DB error: {e}")

    if not row or not row.districts:
        return {"total_districts": 0, "critical": 0, "warning": 0,
                "stable": 0, "recovering": 0, "avg_level_mbgl": 0,
                "max_level_mbgl": 0, "pct_at_risk": 0,
                "model_trained": predictor.is_trained(),
                "model_metrics": predictor.get_metrics()}

    total = int(row.districts)
    at_risk = int(row.critical or 0) + int(row.warning or 0)

    result = {
        "total_districts": total,
        "critical": int(row.critical or 0),
        "warning": int(row.warning or 0),
        "stable": int(row.stable or 0),
        "recovering": int(row.recovering or 0),
        "avg_level_mbgl": round(float(row.avg_level or 0), 2),
        "max_level_mbgl": round(float(row.max_level or 0), 2),
        "pct_at_risk": round(at_risk / max(total, 1) * 100, 1),
        "model_trained": predictor.is_trained(),
        "model_metrics": predictor.get_metrics(),
    }
    cache_set("stats", result)
    return result


@router.get("/critical")
async def get_critical(db: Session = Depends(get_db)):
    cached = cache_get("critical")
    if cached: return cached

    rows = db.execute(text("""
        SELECT district, state, AVG(water_level_mbgl) as avg_level,
               MAX(year) as year, MAX(quarter) as quarter
        FROM groundwater_readings
        GROUP BY district, state
        HAVING AVG(water_level_mbgl) > 20
        ORDER BY avg_level DESC
        LIMIT 20
    """)).fetchall()

    critical = [{"district": r.district, "state": r.state,
                 "level_mbgl": round(r.avg_level, 1),
                 "year": r.year, "quarter": r.quarter}
                for r in rows]
    result = {"count": len(critical), "districts": critical}
    cache_set("critical", result)
    return result


@router.get("/geojson")
async def get_geojson(db: Session = Depends(get_db)):
    cached = cache_get("geojson")
    if cached: return cached

    rows = db.execute(text("""
        SELECT district, state, 
               AVG(water_level_mbgl) as water_level_mbgl,
               AVG(latitude) as latitude, 
               AVG(longitude) as longitude,
               MAX(year) as year, MAX(quarter) as quarter
        FROM groundwater_readings
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        GROUP BY district, state
        ORDER BY water_level_mbgl DESC
    """)).fetchall()

    features = []
    for r in rows:
        risk = classify_risk(r.water_level_mbgl)    
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r.longitude, r.latitude]},
            "properties": {"district": r.district, "state": r.state,
                           "level_mbgl": r.water_level_mbgl, "risk": risk,
                           "color": RISK_META[risk]["color"],
                           "year": r.year, "quarter": r.quarter}
        })

    result = {"type": "FeatureCollection", "features": features}
    cache_set("geojson", result)
    return result