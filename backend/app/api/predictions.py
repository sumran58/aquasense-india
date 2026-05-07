from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import pandas as pd
import time
from datetime import datetime

from app.utils.database import get_db, GroundwaterReading, DistrictPrediction, DistrictMeta
from app.utils.config import classify_risk, RISK_META
from app.ml.pipeline import AquaSensePredictor, engineer_features, FEATURE_COLUMNS

router = APIRouter()
_predictor = None
_cache = {}
CACHE_TTL = 3600


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
    # The level from the DB (last recorded reading)
    current_level_mbgl: float
    # The model-estimated level for TODAY after bridging the gap
    estimated_current_level_mbgl: float
    db_last_year: int
    db_last_quarter: int
    overall_risk: str
    risk_color: str
    summary: str
    trend: str
    predictions: List[dict]
    shap_explanation: Optional[dict] = None
    data_note: Optional[str] = None


def build_features_from_history(district: str, db: Session, user_overrides: dict = None):
    """
    Build ML features using the EXACT SAME method as training.
    Aggregates multiple wells to district quarterly averages FIRST.
    Returns (features_dict, last_year, last_quarter) — the DB's last data point.
    """
    readings = (db.query(GroundwaterReading)
                .filter(GroundwaterReading.district == district)
                .order_by(GroundwaterReading.year, GroundwaterReading.quarter)
                .all())

    if not readings:
        return None, None, None

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

    # Aggregate wells to district quarterly averages
    # FIX: exclude groupby keys (district, year, quarter) from agg_cols
    # pandas raises "cannot insert district, already exists" if they are included
    agg_cols = {
        "water_level_mbgl": "mean",
        "rainfall_mm": "mean",
        "latitude": "first",
        "longitude": "first",
        "state": "first",
        "population_density": "first",
        "agricultural_area_pct": "first",
        "irrigation_wells_per_km2": "first",
        "ndvi_mean": "first",
    }
    df = df.groupby(["district", "year", "quarter"]).agg(agg_cols).reset_index()
    df = df.sort_values(["year", "quarter"])

    df_feat = engineer_features(df)
    last_row = df_feat.iloc[-1]

    # These are the DB's actual last year and quarter
    last_year = int(last_row["year"])
    last_quarter = int(last_row["quarter"])

    features = {}
    for col in FEATURE_COLUMNS:
        if col in last_row.index:
            val = last_row[col]
            features[col] = float(val) if pd.notna(val) else 0.0
        else:
            features[col] = 0.0

    # Apply user overrides for contextual features
    if user_overrides:
        for key in ["rainfall_mm", "population_density", "agricultural_area_pct",
                    "irrigation_wells_per_km2", "ndvi_mean"]:
            if key in user_overrides:
                features[key] = user_overrides[key]

    # If user provided a significantly different current level, update lag_1q
    if user_overrides and "current_level_mbgl" in user_overrides:
        try:
            db_latest = float(last_row["water_level_mbgl"])
        except (KeyError, TypeError):
            db_latest = features.get("level_lag_1q", 0.0)
        user_level = user_overrides["current_level_mbgl"]
        if abs(user_level - db_latest) > 0.01:
            features["level_lag_1q"] = user_level

    return features, last_year, last_quarter


@router.post("/district", response_model=PredictionOut)
async def predict_district(req: PredictRequest, db: Session = Depends(get_db)):
    predictor = get_predictor()
    if not predictor.is_trained():
        raise HTTPException(503, "Model not trained.")

    features, db_last_year, db_last_quarter = build_features_from_history(
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

    # Bridge to today first to get the correct estimated current level
    # (bridged_level is the 4th return value — the actual model-estimated level today)
    bridged_features, today_year, today_quarter, bridged_level = predictor.bridge_to_today(
        features, db_last_year, db_last_quarter
    )

    preds, shap_data = predictor.predict_district(
        features,
        quarters_ahead=req.quarters_ahead,
        start_year=db_last_year,
        start_quarter=db_last_quarter,
    )

    # Correct estimated current level — directly from bridge, not SHAP approximation
    estimated_current = round(bridged_level, 2)

    worst = max(preds, key=lambda p: RISK_META[p["risk_level"]]["priority"])
    current = round(features["level_lag_1q"], 2)
    last_pred = preds[-1]["predicted_level_mbgl"]
    delta = last_pred - estimated_current

    if len(preds) >= 2:
        first_p = preds[0]["predicted_level_mbgl"]
        last_p  = preds[-1]["predicted_level_mbgl"]
        overall_trend = ("worsening" if last_p > first_p + 0.3
                         else "improving" if last_p < first_p - 0.3
                         else "stable")
    else:
        overall_trend = preds[0]["trend"]

    now = datetime.now()
    today_q = (now.month - 1) // 3 + 1

    summary = (
        f"{req.district} water table is estimated at {estimated_current:.1f} mbgl today "
        f"(Q{today_q} {now.year}). Forecast shows "
        f"{'deepening' if delta > 0 else 'recovery'} of {abs(delta):.1f} mbgl "
        f"over {req.quarters_ahead} quarters "
        f"({preds[0]['label']} → {preds[-1]['label']}). "
        f"Peak risk: {worst['risk_label']}."
    )

    # Actual gap in quarters (capped bridge may differ from real gap)
    quarters_gap = (now.year - db_last_year) * 4 + (today_q - db_last_quarter)
    bridge_used  = min(quarters_gap, 12)  # matches cap in bridge_to_today
    data_note = None
    if quarters_gap > 4:
        data_note = (
            f"Note: Latest DB data is Q{db_last_quarter} {db_last_year} "
            f"({quarters_gap} quarters ago; model bridged {bridge_used} quarters). "
            f"Upload recent CGWB data for best accuracy."
        )

    return PredictionOut(
        district=req.district,
        state=req.state,
        current_level_mbgl=current,
        estimated_current_level_mbgl=estimated_current,
        db_last_year=db_last_year,
        db_last_quarter=db_last_quarter,
        overall_risk=worst["risk_level"],
        risk_color=RISK_META[worst["risk_level"]]["color"],
        summary=summary,
        trend=overall_trend,
        predictions=preds,
        shap_explanation=shap_data,
        data_note=data_note,
    )


# ─────────────────────────────────────────────
# SHAP ENDPOINT
# ─────────────────────────────────────────────

@router.get("/shap/{district}")
async def get_shap_explanation(district: str, db: Session = Depends(get_db)):
    """
    SHAP feature importance for a district — reflects bridged 2026 state,
    so every district gets unique values based on its actual current condition.
    """
    predictor = get_predictor()
    if not predictor.is_trained():
        raise HTTPException(503, "Model not trained.")

    features, last_year, last_quarter = build_features_from_history(district=district, db=db)
    if features is None:
        raise HTTPException(404, f"No data found for {district}")

    # Bridge to today — unpack all 4 return values correctly
    bridged_features, _, _, _ = predictor.bridge_to_today(features, last_year, last_quarter)

    explanation = predictor.explain(bridged_features, top_n=10)
    explanation["district"] = district
    return explanation


# ─────────────────────────────────────────────
# STATS, CRITICAL, GEOJSON
# ─────────────────────────────────────────────

@router.get("/stats")
async def prediction_stats(db: Session = Depends(get_db)):
    """
    Returns dashboard stats based on BRIDGED 2026 estimates — not raw DB levels.
    Reuses the current_estimates cache so no extra bridging is needed.
    Falls back to DB stats if current_estimates not yet cached.
    """
    predictor = get_predictor()

    # Try to reuse already-computed bridged estimates from current_estimates cache
    estimates_cache = cache_get("current_estimates")
    if estimates_cache:
        features_data = estimates_cache.get("features", [])
        if features_data:
            levels = [f["properties"]["level_mbgl"] for f in features_data]
            total  = len(levels)
            critical   = sum(1 for l in levels if l > 20)
            warning    = sum(1 for l in levels if 12 < l <= 20)
            stable     = sum(1 for l in levels if 5 <= l <= 12)
            recovering = sum(1 for l in levels if l < 5)
            at_risk    = critical + warning
            result = {
                "total_districts": total,
                "critical":        critical,
                "warning":         warning,
                "stable":          stable,
                "recovering":      recovering,
                "avg_level_mbgl":  round(float(sum(levels) / max(total, 1)), 2),
                "max_level_mbgl":  round(float(max(levels)), 2),
                "pct_at_risk":     round(at_risk / max(total, 1) * 100, 1),
                "model_trained":   predictor.is_trained(),
                "model_metrics":   predictor.get_metrics(),
                "data_as_of":      "Q2 2026 (model-estimated)",
            }
            cache_set("stats", result)
            return result

    # Fallback: use cached stats if available
    cached = cache_get("stats")
    if cached:
        cached["model_trained"] = predictor.is_trained()
        return cached

    # Last resort: compute from DB directly
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
                SELECT g.district, AVG(g.water_level_mbgl) as avg_level
                FROM groundwater_readings g
                INNER JOIN (
                    SELECT district, MAX(year * 10 + quarter) AS max_yq
                    FROM groundwater_readings GROUP BY district
                ) latest ON g.district = latest.district
                       AND (g.year * 10 + g.quarter) = latest.max_yq
                GROUP BY g.district
            )
        """)).fetchone()
    except Exception as e:
        raise HTTPException(500, f"DB error: {e}")

    if not row or not row.districts:
        return {
            "total_districts": 0, "critical": 0, "warning": 0,
            "stable": 0, "recovering": 0, "avg_level_mbgl": 0,
            "max_level_mbgl": 0, "pct_at_risk": 0,
            "model_trained": predictor.is_trained(),
            "model_metrics": predictor.get_metrics(),
            "data_as_of": "DB (historical)",
        }

    total  = int(row.districts)
    at_risk = int(row.critical or 0) + int(row.warning or 0)
    result = {
        "total_districts": total,
        "critical":        int(row.critical or 0),
        "warning":         int(row.warning or 0),
        "stable":          int(row.stable or 0),
        "recovering":      int(row.recovering or 0),
        "avg_level_mbgl":  round(float(row.avg_level or 0), 2),
        "max_level_mbgl":  round(float(row.max_level or 0), 2),
        "pct_at_risk":     round(at_risk / max(total, 1) * 100, 1),
        "model_trained":   predictor.is_trained(),
        "model_metrics":   predictor.get_metrics(),
        "data_as_of":      "DB (historical — visit Risk Map to load 2026 estimates)",
    }
    cache_set("stats", result)
    return result


@router.get("/critical")
async def get_critical(db: Session = Depends(get_db)):
    """
    Returns most critical districts based on BRIDGED 2026 estimates.
    Reuses current_estimates cache — no extra bridging needed.
    Falls back to DB if estimates not yet loaded.
    """
    # Reuse bridged estimates if available
    estimates_cache = cache_get("current_estimates")
    if estimates_cache:
        features_data = estimates_cache.get("features", [])
        if features_data:
            # Sort by bridged level descending, take top 20 critical
            critical_districts = sorted(
                [f for f in features_data if f["properties"]["level_mbgl"] > 20],
                key=lambda f: f["properties"]["level_mbgl"],
                reverse=True
            )[:20]
            critical = [
                {
                    "district":  f["properties"]["district"],
                    "state":     f["properties"]["state"],
                    "level_mbgl": f["properties"]["level_mbgl"],
                    "year":      "2026 (estimated)",
                    "quarter":   2,
                    "estimated": True,
                }
                for f in critical_districts
            ]
            result = {"count": len(critical), "districts": critical,
                      "data_as_of": "Q2 2026 (model-estimated)"}
            cache_set("critical", result)
            return result

    # Fallback cache
    cached = cache_get("critical")
    if cached:
        return cached

    # Last resort: DB query
    rows = db.execute(text("""
        SELECT g.district, g.state, AVG(g.water_level_mbgl) as avg_level,
               g.year, g.quarter
        FROM groundwater_readings g
        INNER JOIN (
            SELECT district, MAX(year * 10 + quarter) AS max_yq
            FROM groundwater_readings GROUP BY district
        ) latest ON g.district = latest.district
               AND (g.year * 10 + g.quarter) = latest.max_yq
        GROUP BY g.district, g.state
        HAVING AVG(g.water_level_mbgl) > 20
        ORDER BY avg_level DESC
        LIMIT 20
    """)).fetchall()

    critical = [{"district": r.district, "state": r.state,
                 "level_mbgl": round(r.avg_level, 1),
                 "year": r.year, "quarter": r.quarter,
                 "estimated": False}
                for r in rows]
    result = {"count": len(critical), "districts": critical,
              "data_as_of": "DB (historical)"}
    cache_set("critical", result)
    return result


@router.get("/geojson")
async def get_geojson(db: Session = Depends(get_db)):
    cached = cache_get("geojson")
    if cached:
        return cached

    rows = db.execute(text("""
        SELECT g.district, g.state,
               AVG(g.water_level_mbgl) as water_level_mbgl,
               AVG(g.latitude) as latitude,
               AVG(g.longitude) as longitude,
               g.year, g.quarter
        FROM groundwater_readings g
        INNER JOIN (
            SELECT district, MAX(year * 10 + quarter) AS max_yq
            FROM groundwater_readings GROUP BY district
        ) latest ON g.district = latest.district
               AND (g.year * 10 + g.quarter) = latest.max_yq
        WHERE g.latitude IS NOT NULL AND g.longitude IS NOT NULL
        GROUP BY g.district, g.state
        ORDER BY water_level_mbgl DESC
    """)).fetchall()

    features = []
    for r in rows:
        level = round(float(r.water_level_mbgl), 2)
        risk = classify_risk(level)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [round(float(r.longitude), 4), round(float(r.latitude), 4)],
            },
            "properties": {
                "district": r.district, "state": r.state,
                "level_mbgl": level, "risk": risk,
                "color": RISK_META[risk]["color"],
                "year": r.year, "quarter": r.quarter,
            },
        })

    result = {"type": "FeatureCollection", "features": features}
    cache_set("geojson", result)
    return result

@router.get("/current-estimates")
async def get_current_estimates(db: Session = Depends(get_db)):
    cached = cache_get("current_estimates")
    if cached:
        return cached

    predictor = get_predictor()
    if not predictor.is_trained():
        raise HTTPException(503, "Model not trained.")

    rows = db.execute(text("""
        SELECT g.district, g.state,
               AVG(g.water_level_mbgl) as water_level_mbgl,
               AVG(g.latitude) as latitude,
               AVG(g.longitude) as longitude,
               g.year, g.quarter
        FROM groundwater_readings g
        INNER JOIN (
            SELECT district, MAX(year * 10 + quarter) AS max_yq
            FROM groundwater_readings GROUP BY district
        ) latest ON g.district = latest.district
               AND (g.year * 10 + g.quarter) = latest.max_yq
        WHERE g.latitude IS NOT NULL AND g.longitude IS NOT NULL
        GROUP BY g.district, g.state
    """)).fetchall()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from app.utils.database import SessionLocal

    def process_district(r):
        thread_db = SessionLocal()
        try:
            features, last_year, last_quarter = build_features_from_history(
                district=r.district, db=thread_db
            )
            if features is None:
                return None
            _, _, _, bridged_level = predictor.bridge_to_today(
                features, last_year, last_quarter
            )
            risk = classify_risk(bridged_level)
            return {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(float(r.longitude), 4), round(float(r.latitude), 4)],
                },
                "properties": {
                    "district":   r.district,
                    "state":      r.state,
                    "level_mbgl": round(bridged_level, 2),
                    "db_level":   round(float(r.water_level_mbgl), 2),
                    "db_year":    r.year,
                    "db_quarter": r.quarter,
                    "risk":       risk,
                    "color":      RISK_META[risk]["color"],
                    "estimated":  True,
                },
            }
        except Exception as e:
            print(f"Error processing {r.district}: {e}")
            return None
        finally:
            thread_db.close()

    features_list = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_district, r): r for r in rows}
        for future in as_completed(futures):
            result = future.result()
            if result:
                features_list.append(result)

    result = {"type": "FeatureCollection", "features": features_list}
    cache_set("current_estimates", result)
    return result