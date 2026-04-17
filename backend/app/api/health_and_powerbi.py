"""Power BI export and health endpoints."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime
import numpy as np

from app.utils.database import get_db, GroundwaterReading, DistrictMeta
from app.utils.config import classify_risk, RISK_META
from app.ml.pipeline import get_predictor

# ── Health router ──────────────────────────────────────────────────────────
health_router = APIRouter()

@health_router.get("/health")
async def health():
    return {"status": "ok", "service": "AquaSense India", "version": "1.0.0",
            "model_trained": get_predictor().is_trained()}

@health_router.get("/model-status")
async def model_status():
    return {
        "trained":    get_predictor().is_trained(),
        "metrics":    get_predictor().get_metrics(),
        "train_cmd":  "python -m app.ml.pipeline",
        "upload_cmd": "POST /api/ingest/upload with your CGWB CSV",
    }


# ── Power BI router ────────────────────────────────────────────────────────
powerbi_router = APIRouter()

@powerbi_router.get("/export")
async def export_for_powerbi(db: Session = Depends(get_db)):
    """
    Flat denormalized table — connect directly from Power BI.
    Power BI: Home → Get Data → Web → http://localhost:8000/api/powerbi/export
    """
    readings = db.query(GroundwaterReading).order_by(
        GroundwaterReading.year, GroundwaterReading.quarter
    ).all()
    meta_map = {m.district: m for m in db.query(DistrictMeta).all()}

    rows = []
    for r in readings:
        meta = meta_map.get(r.district)
        risk = classify_risk(r.water_level_mbgl)
        rows.append({
            "State":             r.state,
            "District":          r.district,
            "Year":              r.year,
            "Quarter":           r.quarter,
            "Quarter_Label":     f"Q{r.quarter} {r.year}",
            "Water_Level_MBGL":  r.water_level_mbgl,
            "Risk_Level":        risk,
            "Risk_Priority":     RISK_META[risk]["priority"],
            "Rainfall_MM":       r.rainfall_mm,
            "Latitude":          r.latitude or (meta.latitude if meta else None),
            "Longitude":         r.longitude or (meta.longitude if meta else None),
            "Population_2021":   meta.population_2021 if meta else None,
            "Agricultural_Pct":  meta.agricultural_area_pct if meta else None,
            "Data_Source":       r.data_source,
        })

    return {
        "data":         rows,
        "row_count":    len(rows),
        "exported_at":  datetime.utcnow().isoformat(),
    }


@powerbi_router.get("/kpis")
async def powerbi_kpis(db: Session = Depends(get_db)):
    """KPI measures for Power BI cards."""
    readings = db.query(GroundwaterReading).all()
    if not readings:
        return {}

    levels   = [r.water_level_mbgl for r in readings]
    districts = set(r.district for r in readings)
    risk_counts = {"CRITICAL": 0, "WARNING": 0, "STABLE": 0, "RECOVERING": 0}
    for r in readings:
        risk_counts[classify_risk(r.water_level_mbgl)] += 1

    return {
        "Total_Districts":     len(districts),
        "Critical_Districts":  risk_counts["CRITICAL"],
        "Warning_Districts":   risk_counts["WARNING"],
        "Stable_Districts":    risk_counts["STABLE"],
        "Recovering_Districts":risk_counts["RECOVERING"],
        "Avg_Level_MBGL":      round(float(np.mean(levels)), 2),
        "Max_Level_MBGL":      round(float(max(levels)), 2),
        "Pct_At_Risk":         round((risk_counts["CRITICAL"] + risk_counts["WARNING"]) / len(districts) * 100, 1),
    }