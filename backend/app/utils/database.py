from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Text, UniqueConstraint
)
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from app.utils.config import settings

# ─────────────────────────────────────────────
# DATABASE ENGINE (Docker-safe + SQLite-safe)
# ─────────────────────────────────────────────

connect_args = {}

# Only SQLite needs this flag
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


# ─────────────────────────────────────────────
# TABLE 1: RAW GROUNDWATER DATA
# ─────────────────────────────────────────────

class GroundwaterReading(Base):
    __tablename__ = "groundwater_readings"

    id = Column(Integer, primary_key=True, index=True)

    state = Column(String(100), nullable=False, index=True)
    district = Column(String(100), nullable=False, index=True)

    well_id = Column(String(80), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    year = Column(Integer, nullable=False)
    quarter = Column(Integer, nullable=False)

    season = Column(String(20), nullable=True)

    water_level_mbgl = Column(Float, nullable=False)
    rainfall_mm = Column(Float, nullable=True)

    data_source = Column(String(50), default="CGWB")
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("well_id", "year", "quarter", name="uq_well_year_quarter"),
    )


# ─────────────────────────────────────────────
# TABLE 2: ML PREDICTIONS
# ─────────────────────────────────────────────

class DistrictPrediction(Base):
    __tablename__ = "district_predictions"

    id = Column(Integer, primary_key=True, index=True)

    state = Column(String(100), nullable=False, index=True)
    district = Column(String(100), nullable=False, index=True)

    prediction_year = Column(Integer, nullable=False)
    prediction_quarter = Column(Integer, nullable=False)

    predicted_level_mbgl = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)

    confidence_interval_low = Column(Float, nullable=True)
    confidence_interval_high = Column(Float, nullable=True)

    shap_top_feature = Column(String(100), nullable=True)
    shap_top_value = Column(Float, nullable=True)

    model_version = Column(String(20), default="1.0")
    created_at = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────
# TABLE 3: DISTRICT META
# ─────────────────────────────────────────────

class DistrictMeta(Base):
    __tablename__ = "district_meta"

    id = Column(Integer, primary_key=True, index=True)

    state = Column(String(100), nullable=False)
    district = Column(String(100), nullable=False, unique=True, index=True)

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    area_sq_km = Column(Float, nullable=True)
    population_2021 = Column(Integer, nullable=True)

    agricultural_area_pct = Column(Float, nullable=True)
    irrigation_wells = Column(Integer, nullable=True)
    aquifer_type = Column(String(50), nullable=True)


# ─────────────────────────────────────────────
# TABLE 4: UPLOAD LOG
# ─────────────────────────────────────────────

class UploadLog(Base):
    __tablename__ = "upload_log"

    id = Column(Integer, primary_key=True, index=True)

    filename = Column(String(255), nullable=False)
    rows_added = Column(Integer, default=0)
    rows_skipped = Column(Integer, default=0)

    status = Column(String(20), default="success")
    error_msg = Column(Text, nullable=True)

    uploaded_at = Column(DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────
# DB DEPENDENCY
# ─────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────
# OPTIONAL: AUTO CREATE TABLES (VERY USEFUL)
# ─────────────────────────────────────────────

def init_db():
    Base.metadata.create_all(bind=engine)