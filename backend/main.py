from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.api.predictions import router as pred_router
from app.api.districts import router as dist_router
from app.api.ingest import router as ingest_router
from app.api.health_and_powerbi import health_router, powerbi_router
from app.utils.config import settings
from app.utils.database import engine, Base
from sqlalchemy import text

from app.utils.database import init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_dist_yq ON groundwater_readings(district, year, quarter)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_yq ON groundwater_readings(year, quarter)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_state ON groundwater_readings(state)"))
            conn.commit()
        logger.info("DB indexes ready")
    except Exception as e:
        logger.warning(f"Index creation: {e}")
    logger.info("AquaSense India API started — DB tables ready")
    yield
    logger.info("API shutdown")


app = FastAPI(
    title="AquaSense India — Groundwater Intelligence API",
    description="""
## India's First District-Level Groundwater Depletion Predictor

### Quick Start
1. **Upload CGWB data**: `POST /api/ingest/upload` with your CSV
2. **Train model**: Hit `/api/ingest/retrain`
3. **Predict**: `POST /api/predictions/district`

### Data Sources
- CGWB: https://ckandev.indiadataportal.com/dataset/groundwater
- India WRIS: https://indiawris.gov.in/wris/#/
""",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router,   prefix="/api",            tags=["Health"])
app.include_router(pred_router,     prefix="/api/predictions", tags=["Predictions"])
app.include_router(dist_router,     prefix="/api/districts",   tags=["Districts"])
app.include_router(ingest_router,   prefix="/api/ingest",      tags=["Data Ingestion"])
app.include_router(powerbi_router,  prefix="/api/powerbi",     tags=["Power BI"])


@app.get("/")
async def root():
    return {"project": "AquaSense India", "docs": "/docs", "status": "running"}

@app.on_event("startup")
def startup():
    init_db()