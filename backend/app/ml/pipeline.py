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
# TRAIN XGBOOST
# ─────────────────────────────────────────────

def train_xgboost(df: pd.DataFrame):

    df_feat = engineer_features(df)

    X = df_feat.reindex(columns=FEATURE_COLUMNS).fillna(0)
    y = df_feat[TARGET]

    split = int(len(df_feat) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # ── Evaluation (IMPORTANT)
    y_pred = model.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2:", r2_score(y_test, y_pred))

    # Save artifacts
    joblib.dump(model, MODEL_DIR / "xgb_model.pkl")
    joblib.dump(FEATURE_COLUMNS, MODEL_DIR / "feature_names.pkl")

    return model


# ─────────────────────────────────────────────
# ANOMALY DETECTOR
# ─────────────────────────────────────────────

def train_anomaly_detector(df: pd.DataFrame):

    df_feat = engineer_features(df)

    cols = ["level_accel", "yoy_change", "rainfall_deficit", "level_roll_std_4q"]
    X = df_feat[cols].fillna(0)

    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X)

    joblib.dump(iso, MODEL_DIR / "isolation_forest.pkl")


# ─────────────────────────────────────────────
# PREDICTOR CLASS (FIXED + FULL API SUPPORT)
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

        for i in range(quarters_ahead):

            pred = float(self.predict(base)[0])

            risk = classify_risk(pred)

            preds.append({
                "quarter": i + 1,
                "predicted_level_mbgl": pred,
                "risk_level": risk,
                "risk_label": RISK_META[risk]["label"],
                "risk_message": RISK_META[risk]["message"],
            })

            # recursive update
            base["level_lag_1q"] = pred

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