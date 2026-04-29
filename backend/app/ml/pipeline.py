"""
AquaSense ML Pipeline (PRODUCTION FIXED VERSION)
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest

import xgboost as xgb

from app.utils.config import classify_risk, RISK_META

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# MODEL PATH
# ─────────────────────────────────────────────

MODEL_DIR = Path("/app/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────

FEATURE_COLUMNS = [
    "level_lag_1q",
    "level_lag_2q",
    "level_lag_4q",
    "level_lag_8q",
    "level_roll_mean_4q",
    "level_roll_mean_8q",
    "level_roll_std_4q",
    "level_roll_min_4q",
    "yoy_change",
    "level_accel",
    "consecutive_depletion",
    "rainfall_mm",
    "rainfall_lag_1q",
    "rainfall_deficit",
    "cum_deficit_4q",
    "population_density",
    "agricultural_area_pct",
    "irrigation_wells_per_km2",
    "ndvi_mean",
    "quarter",
    "year_normalized",
    "is_monsoon",
    "is_rabi",
]

TARGET = "water_level_mbgl"


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["district", "year", "quarter"])
    g = df.groupby("district")["water_level_mbgl"]

    df["level_lag_1q"] = g.shift(1)
    df["level_lag_2q"] = g.shift(2)
    df["level_lag_4q"] = g.shift(4)
    df["level_lag_8q"] = g.shift(8)

    df["level_roll_mean_4q"] = g.transform(lambda x: x.rolling(4, min_periods=2).mean())
    df["level_roll_mean_8q"] = g.transform(lambda x: x.rolling(8, min_periods=4).mean())
    df["level_roll_std_4q"] = g.transform(lambda x: x.rolling(4, min_periods=2).std())
    df["level_roll_min_4q"] = g.transform(lambda x: x.rolling(4, min_periods=2).min())

    df["yoy_change"] = g.diff().fillna(0)
    df["level_accel"] = df.groupby("district")["yoy_change"].diff().fillna(0)

    df["consecutive_depletion"] = 0

    if "rainfall_mm" not in df.columns:
        df["rainfall_mm"] = 0.0

    rg = df.groupby("district")["rainfall_mm"]
    df["rainfall_lag_1q"] = rg.shift(1)

    mean_rain = rg.transform("mean")
    df["rainfall_deficit"] = mean_rain - df["rainfall_mm"]
    df["cum_deficit_4q"] = 0.0

    df["year_normalized"] = (
        df["year"] - df["year"].min()
    ) / (df["year"].max() - df["year"].min() + 1)

    df["is_monsoon"] = (df["quarter"] == 3).astype(int)
    df["is_rabi"] = df["quarter"].isin([1, 4]).astype(int)

    for c in [
        "population_density",
        "agricultural_area_pct",
        "irrigation_wells_per_km2",
        "ndvi_mean",
    ]:
        if c not in df.columns:
            df[c] = 0.0

    return df


# ─────────────────────────────────────────────
# PREDICTOR CLASS (FIXED)
# ─────────────────────────────────────────────

class AquaSensePredictor:

    def __init__(self):
        self.model = joblib.load(MODEL_DIR / "xgb_model.pkl")
        self.features = joblib.load(MODEL_DIR / "feature_names.pkl")
        self._trained = True

    def is_trained(self):
        return self._trained

    def get_metrics(self):
        return {
            "model": "XGBoostRegressor",
            "features": len(self.features),
        }

    def predict(self, X: pd.DataFrame):
        X = X.reindex(columns=self.features, fill_value=0)
        return self.model.predict(X)

    def predict_district(self, features: dict, quarters_ahead=4):

        df = pd.DataFrame([features])
        df = df.reindex(columns=self.features, fill_value=0)

        preds = []
        base = df.copy()

        # Track recent predictions for rolling stat updates
        recent_levels = [
            float(base["level_lag_1q"].iloc[0]),
            float(base["level_lag_2q"].iloc[0]),
            float(base["level_lag_4q"].iloc[0]),
            float(base["level_lag_8q"].iloc[0]),
        ]

        current_quarter = int(base["quarter"].iloc[0])

        for i in range(quarters_ahead):

            pred = float(self.predict(base)[0])

            # Can't be negative
            pred = max(0, pred)

            # Dynamic delta constraint based on district's historical volatility
            prev = float(base["level_lag_1q"].iloc[0])
            std = float(base["level_roll_std_4q"].iloc[0])
            max_delta = max(2.0, std * 3)

            delta = pred - prev
            if abs(delta) > max_delta:
                pred = prev + (max_delta if delta > 0 else -max_delta)

            risk = classify_risk(pred)

            preds.append({
                "quarter": i + 1,
                "predicted_level_mbgl": round(pred, 2),
                "risk_level": risk,
                "risk_label": RISK_META[risk]["label"],
                "risk_message": RISK_META[risk]["message"],
            })

            # ── Recursive feature update for next quarter ──

            # Shift lag chain
            old_lag1 = float(base["level_lag_1q"].iloc[0])
            old_lag2 = float(base["level_lag_2q"].iloc[0])
            old_lag4 = float(base["level_lag_4q"].iloc[0])

            base["level_lag_1q"] = pred
            base["level_lag_2q"] = old_lag1
            base["level_lag_4q"] = old_lag2
            base["level_lag_8q"] = old_lag4

            # Rolling updates
            recent_levels.insert(0, pred)
            window_4 = recent_levels[:4]
            window_8 = recent_levels[:8]

            base["level_roll_mean_4q"] = float(np.mean(window_4))
            base["level_roll_mean_8q"] = float(np.mean(window_8))
            base["level_roll_std_4q"] = float(np.std(window_4)) if len(window_4) > 1 else 0.0
            base["level_roll_min_4q"] = float(min(window_4))

            # Year-over-year change
            if len(recent_levels) > 4:
                base["yoy_change"] = pred - recent_levels[4]
            else:
                base["yoy_change"] = pred - recent_levels[-1]

            # Acceleration
            base["level_accel"] = pred - 2 * old_lag1 + old_lag2

            # Advance quarter (1->2->3->4->1...)
            current_quarter = (current_quarter % 4) + 1
            base["quarter"] = current_quarter
            base["is_monsoon"] = 1 if current_quarter == 3 else 0
            base["is_rabi"] = 1 if current_quarter in [1, 4] else 0

            # Advance year normalization
            base["year_normalized"] = float(base["year_normalized"].iloc[0]) + (0.25 / 10)

            # Consecutive depletion tracker
            if pred > old_lag1:
                base["consecutive_depletion"] = int(base["consecutive_depletion"].iloc[0]) + 1
            else:
                base["consecutive_depletion"] = 0

        return preds


# ─────────────────────────────────────────────
# SINGLETON (USE THIS IN API)
# ─────────────────────────────────────────────

_predictor_instance = None


def get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = AquaSensePredictor()
    return _predictor_instance