from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func, text
import pandas as pd
import tempfile, os, threading, sqlite3, re

from app.utils.database import get_db, GroundwaterReading, UploadLog, engine, SessionLocal
from app.data.loader import load_cgwb_csv
from app.utils.config import settings

router = APIRouter()
_jobs: dict = {}


def _get_db_path():
    """Extract file path from DATABASE_URL."""
    url = settings.DATABASE_URL  # e.g. sqlite:///./data/aquasense.db
    path = url.replace("sqlite:///./", "").replace("sqlite:///", "")
    return path


def _bulk_insert(df: pd.DataFrame, job_id: str, filename: str):
    """Fastest SQLite bulk insert using raw sqlite3 + WAL mode + PRAGMA optimizations."""
    _jobs[job_id] = {"status": "processing", "message": "Preparing data...", "progress": 5}
    db = SessionLocal()
    try:
        # Prepare columns
        cols = ["state","district","year","quarter","water_level_mbgl",
                "rainfall_mm","latitude","longitude","season","well_id"]
        ins = pd.DataFrame()
        for c in cols:
            ins[c] = df[c] if c in df.columns else None
        ins["data_source"] = "CGWB_UPLOAD"

        before = len(ins)
        ins = ins.dropna(subset=["state","district","year","quarter","water_level_mbgl"])
        skipped = before - len(ins)

        if ins.empty:
            _jobs[job_id] = {"status": "done", "rows_added": 0, "rows_skipped": skipped}
            return

        _jobs[job_id] = {"status": "processing", "message": "Clearing old data...", "progress": 10}

        # Raw sqlite3 connection for max speed
        db_path = _get_db_path()
        conn = sqlite3.connect(db_path, timeout=120)

        # Speed pragmas — turns off safety during bulk insert
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA cache_size=200000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")

        # Clear existing data
        conn.execute("DELETE FROM groundwater_readings")
        conn.execute("DELETE FROM upload_log")
        conn.commit()

        _jobs[job_id] = {"status": "processing", "message": f"Inserting {len(ins):,} rows (fast mode)...", "progress": 20}

        # Bulk insert — fastest method
        ins.to_sql(
            "groundwater_readings", conn,
            if_exists="append", index=False,
            method="multi", chunksize=50000
        )

        added = len(ins)
        conn.commit()

        # Restore safe settings
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA locking_mode=NORMAL")
        conn.close()

        _jobs[job_id] = {"status": "processing", "message": "Saving upload log...", "progress": 95}
        # Clear prediction cache so dashboard shows fresh data
        try:
            from app.api.predictions import cache_clear
            cache_clear()
        except: pass
        try:
            from app.api.districts import cache_clear as dc
            dc()
        except: pass

        db.add(UploadLog(filename=filename, rows_added=added, rows_skipped=skipped, status="success"))
        db.commit()

        _jobs[job_id] = {
            "status": "done",
            "rows_added": added,
            "rows_skipped": skipped,
            "districts": int(df["district"].nunique()),
            "states": int(df["state"].nunique()),
            "year_range": f"{int(df['year'].min())} - {int(df['year'].max())}",
            "message": f"Successfully ingested {added:,} records",
        }

    except Exception as e:
        db.add(UploadLog(filename=filename, status="failed", error_msg=str(e)[:500]))
        db.commit()
        _jobs[job_id] = {"status": "error", "message": str(e)[:300]}
    finally:
        db.close()


@router.post("/upload")
async def upload_cgwb_csv(
    file: UploadFile = File(...),
    source_format: str = "auto",
    db: Session = Depends(get_db),
):
    if not file.filename.endswith((".csv", ".CSV")):
        raise HTTPException(400, "Only .csv files accepted")

    content = await file.read()

    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as tmp:
            tmp.write(content); tmp_path = tmp.name
        df = load_cgwb_csv(tmp_path, source_format=source_format)
        os.unlink(tmp_path)
    except Exception as e:
        db.add(UploadLog(filename=file.filename, status="failed", error_msg=str(e)[:500]))
        db.commit()
        raise HTTPException(400, f"Failed to parse CSV: {e}")

    import uuid
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"status": "queued", "message": "Starting...", "progress": 0}

    thread = threading.Thread(target=_bulk_insert, args=(df, job_id, file.filename), daemon=True)
    thread.start()

    return {
        "message": f"Upload started — {len(df):,} rows queued",
        "job_id": job_id,
        "filename": file.filename,
        "rows_parsed": len(df),
        "districts": int(df["district"].nunique()),
        "states": int(df["state"].nunique()),
        "year_range": f"{int(df['year'].min())} - {int(df['year'].max())}",
    }


@router.get("/job/{job_id}")
async def job_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    return _jobs[job_id]


@router.get("/status")
async def ingest_status(db: Session = Depends(get_db)):
    count     = db.query(func.count(GroundwaterReading.id)).scalar() or 0
    states    = db.query(GroundwaterReading.state).distinct().count()
    districts = db.query(GroundwaterReading.district).distinct().count()
    year_min  = db.query(func.min(GroundwaterReading.year)).scalar()
    year_max  = db.query(func.max(GroundwaterReading.year)).scalar()
    logs = db.query(UploadLog).order_by(UploadLog.uploaded_at.desc()).limit(5).all()
    return {
        "readings_in_db": count, "states": states, "districts": districts,
        "year_range": f"{year_min} - {year_max}" if year_min else "No data yet",
        "recent_uploads": [
            {"file": l.filename, "rows_added": l.rows_added,
             "status": l.status, "uploaded": l.uploaded_at.isoformat() if l.uploaded_at else None}
            for l in logs
        ],
        "instructions": "Upload CSV then run python -m app.ml.pipeline" if count == 0 else "Data loaded.",
    }


@router.post("/seed-demo")
async def seed_demo(db: Session = Depends(get_db)):
    existing = db.query(func.count(GroundwaterReading.id)).scalar()
    if existing > 0:
        return {"message": f"DB already has {existing:,} records. Clear first."}
    import numpy as np; np.random.seed(42)
    districts = [
        ("Rajasthan","Jodhpur",26.30,73.02,0.75),("Rajasthan","Barmer",25.75,71.39,0.80),
        ("Punjab","Ludhiana",30.90,75.85,0.70),("Punjab","Amritsar",31.63,74.87,0.65),
        ("Haryana","Hisar",29.15,75.72,0.72),("Gujarat","Mehsana",23.60,72.38,0.60),
        ("Maharashtra","Solapur",17.69,75.91,0.55),("Karnataka","Tumkur",13.34,77.10,0.40),
        ("Tamil Nadu","Coimbatore",11.00,76.96,0.30),("Bihar","Patna",25.61,85.14,0.15),
        ("Uttarakhand","Dehradun",30.32,78.03,0.08),("Kerala","Palakkad",10.77,76.65,0.05),
    ]
    rows = []
    for state,district,lat,lng,dep in districts:
        level = dep*22+2
        for year in range(2005,2025):
            rain = np.random.normal(700,200)
            for q in range(1,5):
                level = max(0.5, level+(-0.7 if q==3 else dep*0.4+0.05)+np.random.normal(0,0.25))
                rows.append({"state":state,"district":district,"latitude":lat,"longitude":lng,
                             "year":year,"quarter":q,"water_level_mbgl":round(level,2),
                             "rainfall_mm":round(rain/4*(2.5 if q==3 else 0.5),1),"data_source":"DEMO"})
    with engine.begin() as conn:
        pd.DataFrame(rows).to_sql("groundwater_readings", conn, if_exists="append", index=False)
    return {"message": f"Seeded {len(rows):,} demo records"}

@router.post("/retrain")
async def retrain_model(db: Session = Depends(get_db)):
    """Retrain XGBoost with recursive-aware training (fast version)."""
    from app.ml.pipeline import train_xgboost_recursive_fast, FEATURE_COLUMNS, engineer_features
    import app.ml.pipeline as pipeline_module
    
    count = db.query(func.count(GroundwaterReading.id)).scalar() or 0
    if count == 0:
        raise HTTPException(400, "No data in DB. Upload CSV first.")
    
    rows = db.query(GroundwaterReading).all()
    df = pd.DataFrame([{
        "district": r.district,
        "state": r.state,
        "year": r.year,
        "quarter": r.quarter,
        "water_level_mbgl": r.water_level_mbgl,
        "rainfall_mm": r.rainfall_mm if r.rainfall_mm else 0.0,
        "latitude": r.latitude,
        "longitude": r.longitude,
        "population_density": 0.0,
        "agricultural_area_pct": 0.0,
        "irrigation_wells_per_km2": 0.0,
        "ndvi_mean": 0.0,
    } for r in rows])
    
    model = train_xgboost_recursive_fast(df)
    
    pipeline_module._predictor_instance = None
    
    return {
        "message": f"Model retrained on {len(df):,} records",
        "districts": int(df["district"].nunique()),
    }

@router.delete("/clear")
async def clear_data(db: Session = Depends(get_db)):
    if not settings.DEBUG:
        raise HTTPException(403, "Cannot clear in production")
    count = db.query(GroundwaterReading).delete()
    db.commit()
    return {"deleted": count}
