from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional
import time

from app.utils.database import get_db, GroundwaterReading, DistrictMeta
from app.utils.config import classify_risk, RISK_META

router = APIRouter()

_cache = {}
CACHE_TTL = 600

def cache_get(key):
    if key in _cache:
        val, ts = _cache[key]
        if time.time() - ts < CACHE_TTL: return val
    return None

def cache_set(key, val):
    _cache[key] = (val, time.time())

def cache_clear():
    _cache.clear()

LATEST_SQL = """
    SELECT g.district, g.state, g.water_level_mbgl,
           g.latitude, g.longitude, g.year, g.quarter
    FROM groundwater_readings g
    INNER JOIN (
        SELECT district, MAX(year * 10 + quarter) AS max_yq
        FROM groundwater_readings GROUP BY district
    ) latest ON g.district = latest.district
           AND (g.year * 10 + g.quarter) = latest.max_yq
"""

@router.get("/")
async def list_districts(state: Optional[str] = None, db: Session = Depends(get_db)):
    cache_key = f"districts_{state or 'all'}"
    cached = cache_get(cache_key)
    if cached: return cached

    sql = LATEST_SQL + (" WHERE g.state = :state" if state else "")
    rows = db.execute(text(sql), {"state": state} if state else {}).fetchall()

    result = []
    for r in rows:
        risk = classify_risk(r.water_level_mbgl)
        result.append({"district":r.district,"state":r.state,"latest_level":r.water_level_mbgl,
                       "latest_year":r.year,"latest_quarter":r.quarter,"risk":risk,
                       "risk_color":RISK_META[risk]["color"],"latitude":r.latitude,"longitude":r.longitude})

    result.sort(key=lambda x: x["latest_level"], reverse=True)
    final = {"count": len(result), "districts": result}
    cache_set(cache_key, final)
    return final

@router.get("/states")
async def list_states(db: Session = Depends(get_db)):
    states = db.execute(text("SELECT DISTINCT state FROM groundwater_readings ORDER BY state")).fetchall()
    return {"states": [s[0] for s in states]}

@router.get("/trend-data")
async def trend_data(db: Session = Depends(get_db)):
    cached = cache_get("trend")
    if cached: return cached

    rows = db.execute(text("""
        SELECT year, quarter,
               AVG(water_level_mbgl) AS avg,
               MAX(water_level_mbgl) AS max,
               MIN(water_level_mbgl) AS min,
               COUNT(*) AS cnt
        FROM groundwater_readings
        GROUP BY year, quarter
        ORDER BY year, quarter
    """)).fetchall()

    if not rows: return {"trend": []}

    result = {
        "trend": [{"period":f"Q{r.quarter} {r.year}","year":r.year,"quarter":r.quarter,
                   "avg":round(float(r.avg),3),"max":round(float(r.max),3),
                   "min":round(float(r.min),3),"count":r.cnt} for r in rows],
        "total_periods": len(rows)
    }
    cache_set("trend", result)
    return result

@router.get("/{district}/history")
async def district_history(district: str, db: Session = Depends(get_db)):
    readings = (db.query(GroundwaterReading)
        .filter(GroundwaterReading.district.ilike(district))
        .order_by(GroundwaterReading.year, GroundwaterReading.quarter).all())
    if not readings:
        raise HTTPException(404, f"No data for district: {district}")
    levels = [r.water_level_mbgl for r in readings]
    return {
        "district":district,"state":readings[0].state,"total":len(readings),
        "current_level":levels[-1],"min_level":min(levels),"max_level":max(levels),
        "trend":"worsening" if levels[-1]>levels[0] else "improving",
        "current_risk":classify_risk(levels[-1]),
        "history":[{"year":r.year,"quarter":r.quarter,"label":f"Q{r.quarter} {r.year}",
                    "level":r.water_level_mbgl,"rainfall":r.rainfall_mm,
                    "risk":classify_risk(r.water_level_mbgl)} for r in readings],
    }

@router.get("/map/folium", response_class=HTMLResponse)
async def folium_map(db: Session = Depends(get_db)):
    return HTMLResponse("<p>Use the frontend map instead.</p>")
