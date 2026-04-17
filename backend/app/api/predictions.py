from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import time

from app.utils.database import get_db, GroundwaterReading, DistrictPrediction, DistrictMeta
from app.utils.config import classify_risk, RISK_META
from app.ml.pipeline import AquaSensePredictor

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


@router.post("/district", response_model=PredictionOut)
async def predict_district(req: PredictRequest, db: Session = Depends(get_db)):
    predictor = get_predictor()
    if not predictor.is_trained():
        raise HTTPException(503, "Model not trained.")

    historical = (db.query(GroundwaterReading)
        .filter(GroundwaterReading.district == req.district)
        .order_by(GroundwaterReading.year.desc(), GroundwaterReading.quarter.desc())
        .limit(8).all())

    if historical:
        levels = [r.water_level_mbgl for r in historical]
        roll_mean = float(np.mean(levels[:4]))
        roll_std = float(np.std(levels[:4]))
        yoy_chg = float(levels[0] - levels[-1])
        rain_vals = [r.rainfall_mm for r in historical if r.rainfall_mm]
        rain_def = (float(np.mean(rain_vals)) if rain_vals else req.rainfall_mm_last_quarter) - req.rainfall_mm_last_quarter
    else:
        levels = [req.current_level_mbgl]
        roll_mean = req.current_level_mbgl
        roll_std = 1.0
        yoy_chg = 0.0
        rain_def = 0.0

    features = {
        "level_lag_1q": req.current_level_mbgl,
        "level_lag_2q": req.current_level_mbgl * 0.97,
        "level_lag_4q": req.current_level_mbgl * 0.93,
        "level_lag_8q": req.current_level_mbgl * 0.88,
        "level_roll_mean_4q": roll_mean,
        "level_roll_mean_8q": roll_mean * 0.97,
        "level_roll_std_4q": roll_std,
        "level_roll_min_4q": min(levels),
        "yoy_change": yoy_chg,
        "level_accel": 0.0,
        "consecutive_depletion": 2,
        "rainfall_mm": req.rainfall_mm_last_quarter,
        "rainfall_lag_1q": req.rainfall_mm_last_quarter,
        "rainfall_deficit": rain_def,
        "cum_deficit_4q": rain_def * 3,
        "population_density": req.population_density,
        "agricultural_area_pct": req.agricultural_area_pct,
        "irrigation_wells_per_km2": req.irrigation_wells_per_km2,
        "ndvi_mean": req.ndvi_mean,
        "quarter": 1,
        "year_normalized": 0.8,
        "is_monsoon": 0,
        "is_rabi": 1,
    }

    preds = get_predictor().predict_district(features, quarters_ahead=req.quarters_ahead)
    worst = max(preds, key=lambda p: RISK_META[p["risk_level"]]["priority"])
    delta = preds[-1]["predicted_level_mbgl"] - req.current_level_mbgl
    summary = (f"{req.district} will {'deepen' if delta > 0 else 'recover'} by "
               f"{abs(delta):.1f} mbgl over {req.quarters_ahead} quarters. "
               f"Risk: {worst['risk_label']}.")

    return PredictionOut(
        district=req.district, state=req.state,
        current_level_mbgl=req.current_level_mbgl,
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
        # Single fast query - no joins, no subqueries
        # Use per-district averages for accurate risk classification
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
        SELECT DISTINCT district, state, water_level_mbgl, latitude, longitude, year, quarter
        FROM groundwater_readings
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        ORDER BY water_level_mbgl DESC
        LIMIT 300
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