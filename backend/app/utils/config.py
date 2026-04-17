try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

from typing import List
import os

os.makedirs("data", exist_ok=True)

class Settings(BaseSettings):
    APP_NAME: str = "AquaSense India"
    DEBUG: bool = True
    SECRET_KEY: str = "default-secret"
    DATABASE_URL: str = "sqlite:///./data/aquasense.db"

    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:5500",
        "null",
        "https://app.powerbi.com",
    ]

    MODEL_PATH: str = "models/xgb_model.pkl"
    SCALER_PATH: str = "models/scaler.pkl"
    FEATURE_NAMES_PATH: str = "models/feature_names.pkl"
    METRICS_PATH: str = "models/metrics.json"

    CRITICAL_MBGL: float = 20.0
    WARNING_MBGL: float = 12.0
    STABLE_MBGL: float = 5.0

    class Config:
        env_file = ".env"

settings = Settings()

RISK_META = {
    "CRITICAL":   {"label": "Critical",   "color": "#E24B4A", "bg": "#FFF0F0", "priority": 4,
                   "message": "Immediate intervention needed. Wells may go dry within 1-2 seasons."},
    "WARNING":    {"label": "Warning",    "color": "#EF9F27", "bg": "#FFF8ED", "priority": 3,
                   "message": "Depletion accelerating. Conservation measures recommended."},
    "STABLE":     {"label": "Stable",     "color": "#1D9E75", "bg": "#E1F5EE", "priority": 2,
                   "message": "Within acceptable range. Monitor seasonal fluctuations."},
    "RECOVERING": {"label": "Recovering", "color": "#378ADD", "bg": "#EEF4FD", "priority": 1,
                   "message": "Levels improving. Recharge exceeding extraction."},
}

def classify_risk(level_mbgl: float) -> str:
    if level_mbgl > settings.CRITICAL_MBGL:
        return "CRITICAL"
    elif level_mbgl > settings.WARNING_MBGL:
        return "WARNING"
    elif level_mbgl < settings.STABLE_MBGL:
        return "RECOVERING"
    return "STABLE"