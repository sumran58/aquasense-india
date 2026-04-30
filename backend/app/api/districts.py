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
CACHE_TTL = 3600

def cache_get(key):
    if key in _cache:
        val, ts = _cache[key]
        if time.time() - ts < CACHE_TTL: return val
    return None

def cache_set(key, val):
    _cache[key] = (val, time.time())

def cache_clear():
    _cache.clear()

# Get the AVERAGE level across all wells in a district for the latest quarter
# This is consistent with how training aggregates data and how geojson works
LATEST_SQL = """
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
    GROUP BY g.district, g.state
"""

@router.get("/")
async def list_districts(state: Optional[str] = None, db: Session = Depends(get_db)):
    cache_key = f"districts_{state or 'all'}"
    cached = cache_get(cache_key)
    if cached: return cached

    sql = LATEST_SQL
    if state:
        sql += " HAVING g.state = :state"
    sql += " ORDER BY water_level_mbgl DESC"
    
    rows = db.execute(text(sql), {"state": state} if state else {}).fetchall()

    result = []
    for r in rows:
        level = round(float(r.water_level_mbgl), 2)
        risk = classify_risk(level)
        result.append({
            "district": r.district,
            "state": r.state,
            "latest_level": level,
            "latest_year": r.year,
            "latest_quarter": r.quarter,
            "risk": risk,
            "risk_color": RISK_META[risk]["color"],
            "latitude": round(float(r.latitude), 4) if r.latitude else None,
            "longitude": round(float(r.longitude), 4) if r.longitude else None,
        })

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

    # Aggregate to district averages first, then compute national trends
    # This prevents districts with more wells from dominating the average
    rows = db.execute(text("""
        SELECT year, quarter,
               AVG(district_avg) AS avg,
               MAX(district_avg) AS max,
               MIN(district_avg) AS min,
               COUNT(*) AS cnt
        FROM (
            SELECT district, year, quarter, AVG(water_level_mbgl) as district_avg
            FROM groundwater_readings
            GROUP BY district, year, quarter
        )
        GROUP BY year, quarter
        ORDER BY year, quarter
    """)).fetchall()

    if not rows: return {"trend": []}

    result = {
        "trend": [{"period":f"Q{r.quarter} {r.year}","year":r.year,"quarter":r.quarter,
                   "avg":round(float(r.avg),2),"max":round(float(r.max),2),
                   "min":round(float(r.min),2),"count":r.cnt} for r in rows],
        "total_periods": len(rows)
    }
    cache_set("trend", result)
    return result

@router.get("/{district}/history")
async def district_history(district: str, db: Session = Depends(get_db)):
    # Aggregate wells per quarter — return one row per quarter, not per well
    rows = db.execute(text("""
        SELECT district, state, year, quarter,
               AVG(water_level_mbgl) as water_level_mbgl,
               AVG(rainfall_mm) as rainfall_mm,
               COUNT(*) as num_wells
        FROM groundwater_readings
        WHERE district = :district COLLATE NOCASE
        GROUP BY district, state, year, quarter
        ORDER BY year, quarter
    """), {"district": district}).fetchall()
    
    if not rows:
        raise HTTPException(404, f"No data for district: {district}")
    
    levels = [float(r.water_level_mbgl) for r in rows]
    
    return {
        "district": district,
        "state": rows[0].state,
        "total": len(rows),
        "current_level": round(levels[-1], 2),
        "min_level": round(min(levels), 2),
        "max_level": round(max(levels), 2),
        "trend": "worsening" if levels[-1] > levels[0] else "improving",
        "current_risk": classify_risk(levels[-1]),
        "history": [{
            "year": r.year,
            "quarter": r.quarter,
            "label": f"Q{r.quarter} {r.year}",
            "level": round(float(r.water_level_mbgl), 2),
            "rainfall": round(float(r.rainfall_mm), 1) if r.rainfall_mm else None,
            "risk": classify_risk(float(r.water_level_mbgl)),
            "num_wells": r.num_wells,
        } for r in rows],
    }

@router.get("/map/folium", response_class=HTMLResponse)
async def folium_map(db: Session = Depends(get_db)):
    return HTMLResponse("<p>Use the frontend map instead.</p>")