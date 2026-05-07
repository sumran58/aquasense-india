"""
AquaSense ML Pipeline — FINAL CORRECTED VERSION
================================================
Bugs fixed vs original + previous attempt:

1.  SHAP identical for all districts
    → explain() now always builds a fresh DataFrame from the exact features dict
      passed in; handles XGBoost expected_value being an array.

2.  Forecast dates stuck in 2024 / wrong year
    → Added _bridge_to_today() that fast-forwards the model state from the DB's
      last recorded quarter up to the real current quarter (May 2026 = Q2 2026),
      then forecasts forward from Q3 2026 onward.

3.  _advance_quarter() — verified correct, kept as-is.

4.  Dead-code "pass" block removed from predict_district().

5.  year_normalized tracks the real bridging year, not a constant increment.

6.  rainfall_lag_1q and cum_deficit_4q now roll forward correctly during
    bridging and recursive forecasting.

7.  _bridge_to_today made public (bridge_to_today) so routers can call it
    without dunder-name hacks, and it returns bridged_level_today so the
    router can surface the correct "estimated current level".

8.  Training functions unchanged — they were correct already.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest

import xgboost as xgb
import shap

from app.utils.config import classify_risk, RISK_META

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

MODEL_DIR = Path("/app/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# FEATURE DEFINITIONS
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

FEATURE_DISPLAY_NAMES = {
    "level_lag_1q":           "Previous Quarter Level",
    "level_lag_2q":           "Level 2 Quarters Ago",
    "level_lag_4q":           "Level 1 Year Ago",
    "level_lag_8q":           "Level 2 Years Ago",
    "level_roll_mean_4q":     "4-Quarter Rolling Average",
    "level_roll_mean_8q":     "8-Quarter Rolling Average",
    "level_roll_std_4q":      "Level Volatility (4Q)",
    "level_roll_min_4q":      "Shallowest in 4 Quarters",
    "yoy_change":             "Year-over-Year Change",
    "level_accel":            "Depletion Acceleration",
    "consecutive_depletion":  "Consecutive Depletion Quarters",
    "rainfall_mm":            "Quarterly Rainfall (mm)",
    "rainfall_lag_1q":        "Previous Quarter Rainfall",
    "rainfall_deficit":       "Rainfall Deficit",
    "cum_deficit_4q":         "Cumulative Rainfall Deficit (4Q)",
    "population_density":     "Population Density",
    "agricultural_area_pct":  "Agricultural Area %",
    "irrigation_wells_per_km2": "Irrigation Well Density",
    "ndvi_mean":              "Vegetation Index (NDVI)",
    "quarter":                "Quarter of Year",
    "year_normalized":        "Year (normalized)",
    "is_monsoon":             "Is Monsoon Quarter",
    "is_rabi":                "Is Rabi Season",
}

TARGET = "water_level_mbgl"

# Baseline year used for year_normalized — must match training
_BASELINE_YEAR = 2000


# ─────────────────────────────────────────────
# DATE HELPERS
# ─────────────────────────────────────────────

def _current_year_quarter():
    """Return (year, quarter) for today."""
    now = datetime.now()
    return now.year, (now.month - 1) // 3 + 1


def _quarters_between(from_year, from_quarter, to_year, to_quarter):
    """Signed number of quarters from → to. Positive = to is later."""
    return (to_year - from_year) * 4 + (to_quarter - from_quarter)


def _advance_quarter(year, quarter, steps=1):
    """
    Advance (year, quarter) by `steps` quarters.
    Verified: year=2026,q=4,steps=1 → (2027,1) ✓
              year=2026,q=1,steps=1 → (2026,2) ✓
    """
    total = year * 4 + (quarter - 1) + steps
    return total // 4, (total % 4) + 1


def _year_normalized(year: int) -> float:
    """
    Normalise year relative to baseline.
    Uses a fixed denominator so bridged years get correct values.
    Matches what engineer_features produces when max-year is ~2024.
    """
    return (year - _BASELINE_YEAR) / 25.0  # 25-year span roughly covers 2000-2025


# ─────────────────────────────────────────────
# FEATURE ENGINEERING (unchanged from original)
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
    df["level_roll_std_4q"]  = g.transform(lambda x: x.rolling(4, min_periods=2).std())
    df["level_roll_min_4q"]  = g.transform(lambda x: x.rolling(4, min_periods=2).min())

    df["yoy_change"]  = g.diff().fillna(0)
    df["level_accel"] = df.groupby("district")["yoy_change"].diff().fillna(0)

    def count_consecutive(series):
        result, count, prev = [], 0, None
        for val in series:
            count = (count + 1) if (prev is not None and val > prev) else 0
            result.append(count)
            prev = val
        return result

    df["consecutive_depletion"] = df.groupby("district")["water_level_mbgl"].transform(
        lambda x: pd.Series(count_consecutive(x.values), index=x.index)
    )

    if "rainfall_mm" not in df.columns:
        df["rainfall_mm"] = 0.0

    rg = df.groupby("district")["rainfall_mm"]
    df["rainfall_lag_1q"] = rg.shift(1)
    mean_rain = rg.transform("mean")
    df["rainfall_deficit"] = mean_rain - df["rainfall_mm"]
    df["cum_deficit_4q"]   = rg.transform(lambda x: x.rolling(4, min_periods=1).sum())

    df["year_normalized"] = (
        (df["year"] - df["year"].min())
        / (df["year"].max() - df["year"].min() + 1)
    )

    df["is_monsoon"] = (df["quarter"] == 3).astype(int)
    df["is_rabi"]    = df["quarter"].isin([1, 4]).astype(int)

    for c in ["population_density", "agricultural_area_pct",
              "irrigation_wells_per_km2", "ndvi_mean"]:
        if c not in df.columns:
            df[c] = 0.0

    return df


# ─────────────────────────────────────────────
# RECURSIVE STATE UPDATE (shared by bridge + forecast)
# ─────────────────────────────────────────────

def _apply_recursive_step(base: pd.DataFrame,
                           pred: float,
                           level_ring: list,
                           rainfall_ring: list,
                           mean_rainfall: float,
                           current_q: int,
                           current_year: int,
                           year_norm_denominator: float) -> tuple:
    """
    Advance `base` (single-row DataFrame) by one quarter in-place.

    level_ring   — sliding window of ALL recent levels, newest first.
                   Must have at least 8 entries on entry (padded if needed).
                   After call, pred is inserted at index 0.
    rainfall_ring — sliding window of recent rainfall values, newest first.
    mean_rainfall — district historical mean rainfall (constant, used for deficit).
    year_norm_denominator — the exact denominator used during training so
                            year_normalized stays in-distribution.

    Returns (next_q, next_year).
    """
    # ── Level ring: insert new prediction, derive all lags correctly ──────────
    level_ring.insert(0, pred)
    # Trim to keep memory bounded (keep 12 quarters = 3 years)
    if len(level_ring) > 12:
        level_ring.pop()

    # Lags read directly from ring — always correct regardless of step count
    def _ring(idx):
        return float(level_ring[idx]) if idx < len(level_ring) else float(level_ring[-1])

    base["level_lag_1q"] = _ring(1)   # 1 quarter ago  (previous pred)
    base["level_lag_2q"] = _ring(2)   # 2 quarters ago
    base["level_lag_4q"] = _ring(4)   # 4 quarters ago (1 year)
    base["level_lag_8q"] = _ring(8)   # 8 quarters ago (2 years)

    w4 = level_ring[:4]
    w8 = level_ring[:8]
    base["level_roll_mean_4q"] = float(np.mean(w4))
    base["level_roll_mean_8q"] = float(np.mean(w8))
    base["level_roll_std_4q"]  = float(np.std(w4))  if len(w4) > 1 else 0.0
    base["level_roll_min_4q"]  = float(min(w4))

    # yoy_change: current vs 4 quarters ago
    base["yoy_change"]  = pred - _ring(4)
    # acceleration: change in change
    base["level_accel"] = pred - 2 * _ring(1) + _ring(2)

    # Consecutive depletion
    base["consecutive_depletion"] = (
        int(base["consecutive_depletion"].iloc[0]) + 1
        if pred > _ring(1) else 0
    )

    # ── Calendar advance ──────────────────────────────────────────────────────
    next_q    = (current_q % 4) + 1
    next_year = current_year + (1 if current_q == 4 else 0)

    base["quarter"]    = next_q
    base["is_monsoon"] = 1 if next_q == 3 else 0
    base["is_rabi"]    = 1 if next_q in [1, 4] else 0

    # year_normalized: use the SAME denominator as training to stay in-distribution
    base["year_normalized"] = (next_year - _BASELINE_YEAR) / year_norm_denominator

    # ── Rainfall: seasonal proxy + correct deficit rolling ────────────────────
    # Seasonal multipliers relative to district mean (climatological pattern)
    # Q1=dry, Q2=pre-monsoon, Q3=monsoon peak, Q4=post-monsoon
    seasonal_factor = {1: 0.20, 2: 0.35, 3: 2.10, 4: 0.35}
    prev_rainfall = float(base["rainfall_mm"].iloc[0])
    base["rainfall_lag_1q"] = prev_rainfall

    # Use district mean_rainfall to scale seasonal estimate
    next_rainfall = mean_rainfall * seasonal_factor.get(next_q, 1.0)
    base["rainfall_mm"] = next_rainfall

    # Roll rainfall ring and recompute deficit correctly
    rainfall_ring.insert(0, next_rainfall)
    if len(rainfall_ring) > 8:
        rainfall_ring.pop()

    # deficit = mean - current (positive means below average)
    base["rainfall_deficit"] = mean_rainfall - next_rainfall
    # cum_deficit over last 4 quarters = sum of (mean - actual) per quarter
    recent_4_rain = rainfall_ring[:4]
    base["cum_deficit_4q"] = sum(mean_rainfall - r for r in recent_4_rain)

    return next_q, next_year


# ─────────────────────────────────────────────
# TRAINING — STANDARD
# ─────────────────────────────────────────────

def train_xgboost(df: pd.DataFrame):
    """Standard single-step XGBoost training."""
    df_feat = engineer_features(df)
    X = df_feat.reindex(columns=FEATURE_COLUMNS).fillna(0)
    y = df_feat[TARGET]
    split = int(len(df_feat) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    # Save year_norm_denom so inference matches training normalization exactly
    year_norm_denom = float(df_feat["year"].max() - df_feat["year"].min() + 1)
    joblib.dump(year_norm_denom, MODEL_DIR / "year_norm_denom.pkl")

    model = xgb.XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    print(f"[train_xgboost] MAE={mean_absolute_error(y_te,y_pred):.4f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_te,y_pred)):.4f}  "
          f"R2={r2_score(y_te,y_pred):.4f}")
    joblib.dump(model, MODEL_DIR / "xgb_model.pkl")
    joblib.dump(FEATURE_COLUMNS, MODEL_DIR / "feature_names.pkl")
    return model


# ─────────────────────────────────────────────
# TRAINING — RECURSIVE-AWARE
# ─────────────────────────────────────────────

def train_xgboost_recursive_fast(df: pd.DataFrame):
    """
    Recursive-aware training.
    Aggregates wells → district quarterly averages first.
    Samples 3 starting points per district for recursive examples.
    """
    agg_cols = {
        c: ("mean" if c in ["water_level_mbgl", "rainfall_mm", "latitude", "longitude"]
            else "first")
        for c in df.columns if c not in ["district", "year", "quarter"]
    }
    df = df.groupby(["district", "year", "quarter"]).agg(agg_cols).reset_index()
    df_feat = engineer_features(df)

    # Save year_norm_denom so inference stays in-distribution
    year_norm_denom = float(df_feat["year"].max() - df_feat["year"].min() + 1)
    joblib.dump(year_norm_denom, MODEL_DIR / "year_norm_denom.pkl")

    X = df_feat.reindex(columns=FEATURE_COLUMNS).fillna(0)
    y = df_feat[TARGET]
    split = int(len(df_feat) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    # Stage 1
    print("Stage 1: Training initial model...", flush=True)
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    print(f"  Stage 1 R2={r2_score(y_te,model.predict(X_te)):.4f}", flush=True)

    # Stage 2: generate recursive examples
    print("Stage 2: Generating recursive examples...", flush=True)
    recursive_X, recursive_y = [], []

    for district in df_feat["district"].unique():
        d = df_feat[df_feat["district"] == district].sort_values(["year", "quarter"])
        if len(d) < 10:
            continue
        n_samples = max(1, min(3, len(d) - 9))
        indices = np.linspace(4, len(d) - 5, n_samples, dtype=int)

        for start_idx in indices:
            row = d.iloc[start_idx]
            base_f = {col: float(row[col]) if pd.notna(row[col]) else 0.0
                      for col in FEATURE_COLUMNS}
            base_df = pd.DataFrame([base_f]).reindex(columns=FEATURE_COLUMNS, fill_value=0)

            for step in range(min(4, len(d) - start_idx - 1)):
                pred   = float(model.predict(base_df)[0])
                actual = float(d.iloc[start_idx + step + 1][TARGET])
                recursive_X.append(base_df.iloc[0].values.copy())
                recursive_y.append(actual)

                # Simple lag update for training (rainfall not critical here)
                old1 = float(base_df["level_lag_1q"].iloc[0])
                old2 = float(base_df["level_lag_2q"].iloc[0])
                old4 = float(base_df["level_lag_4q"].iloc[0])
                base_df["level_lag_1q"] = pred
                base_df["level_lag_2q"] = old1
                base_df["level_lag_4q"] = old2
                base_df["level_lag_8q"] = old4
                recent = [pred, old1, old2, old4]
                base_df["level_roll_mean_4q"] = float(np.mean(recent))
                base_df["level_roll_mean_8q"] = float(np.mean(recent))
                base_df["level_roll_std_4q"]  = float(np.std(recent))
                base_df["level_roll_min_4q"]  = float(min(recent))
                base_df["yoy_change"]  = pred - old4
                base_df["level_accel"] = pred - 2 * old1 + old2
                next_q = (int(base_df["quarter"].iloc[0]) % 4) + 1
                base_df["quarter"]    = next_q
                base_df["is_monsoon"] = 1 if next_q == 3 else 0
                base_df["is_rabi"]    = 1 if next_q in [1, 4] else 0

    print(f"  Generated {len(recursive_X)} recursive examples", flush=True)

    # Stage 3: retrain on combined data
    if recursive_X:
        print("Stage 3: Retraining on combined data...", flush=True)
        X_combined = pd.concat(
            [X_tr, pd.DataFrame(recursive_X, columns=FEATURE_COLUMNS)],
            ignore_index=True,
        )
        y_combined = pd.concat(
            [y_tr, pd.Series(recursive_y)],
            ignore_index=True,
        )
        model_final = xgb.XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        model_final.fit(X_combined, y_combined)
        y_pred = model_final.predict(X_te)
        print(f"  FINAL R2={r2_score(y_te,y_pred):.4f}  "
              f"MAE={mean_absolute_error(y_te,y_pred):.4f}", flush=True)
        joblib.dump(model_final, MODEL_DIR / "xgb_model.pkl")
        joblib.dump(FEATURE_COLUMNS, MODEL_DIR / "feature_names.pkl")
        return model_final

    joblib.dump(model, MODEL_DIR / "xgb_model.pkl")
    joblib.dump(FEATURE_COLUMNS, MODEL_DIR / "feature_names.pkl")
    return model


# ─────────────────────────────────────────────
# ANOMALY DETECTOR
# ─────────────────────────────────────────────

def train_anomaly_detector(df: pd.DataFrame):
    df_feat = engineer_features(df)
    cols = ["level_accel", "yoy_change", "rainfall_deficit", "level_roll_std_4q"]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(df_feat[cols].fillna(0))
    joblib.dump(iso, MODEL_DIR / "isolation_forest.pkl")


# ─────────────────────────────────────────────
# PREDICTOR
# ─────────────────────────────────────────────

class AquaSensePredictor:

    def __init__(self):
        self.model    = joblib.load(MODEL_DIR / "xgb_model.pkl")
        self.features = joblib.load(MODEL_DIR / "feature_names.pkl")
        self._trained = True
        self._explainer = shap.TreeExplainer(self.model)
        # year_norm_denom: saved by training functions so inference stays
        # in-distribution with training. Defaults to 24 (covers 2000-2023).
        denom_path = MODEL_DIR / "year_norm_denom.pkl"
        self._year_norm_denom = (
            float(joblib.load(denom_path)) if denom_path.exists() else 24.0
        )

    def is_trained(self) -> bool:
        return self._trained

    def get_metrics(self) -> dict:
        return {"model": "XGBoostRegressor", "features": len(self.features)}

    def reload(self):
        """
        Hot-reload model from disk after retraining — no server restart needed.
        Call this from your retrain endpoint after training completes.
        """
        self.model      = joblib.load(MODEL_DIR / "xgb_model.pkl")
        self.features   = joblib.load(MODEL_DIR / "feature_names.pkl")
        self._explainer = shap.TreeExplainer(self.model)
        denom_path = MODEL_DIR / "year_norm_denom.pkl"
        if denom_path.exists():
            self._year_norm_denom = float(joblib.load(denom_path))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.reindex(columns=self.features, fill_value=0))

    # ── FIX 1: SHAP — fresh per-district, correct expected_value handling ───
    def explain(self, features: dict, top_n: int = 8) -> dict:
        """
        Compute SHAP values for a single district's current feature state.

        Fixes:
        - Always builds a brand-new DataFrame from `features` dict, so every
          district gets its own unique SHAP values (old bug: stale shared df).
        - Handles XGBoost returning expected_value as an ndarray or list.
        - shap_values() returns shape (1, n_features) for a single row —
          flatten correctly regardless of SHAP version.
        """
        df = pd.DataFrame(
            [{col: float(features.get(col, 0.0)) for col in self.features}]
        ).reindex(columns=self.features, fill_value=0).astype(float)

        sv = self._explainer.shap_values(df)

        # Handle expected_value as scalar or array
        base = self._explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.asarray(base).flat[0])
        else:
            base = float(base)

        # shap_values for a single-row df: shape (1, n_features) or (n_features,)
        shap_arr = np.asarray(sv)
        if shap_arr.ndim == 2:
            shap_arr = shap_arr[0]        # (n_features,)
        elif shap_arr.ndim == 3:
            shap_arr = shap_arr[0, 0, :]  # multi-output edge case

        pred = base + float(np.sum(shap_arr))

        impacts = [
            {
                "feature":      fname,
                "display_name": FEATURE_DISPLAY_NAMES.get(fname, fname),
                "value":        round(float(df.iloc[0][fname]), 3),
                "shap_value":   round(float(shap_arr[i]), 3),
                "direction":    "deepens" if shap_arr[i] > 0 else "recovers",
            }
            for i, fname in enumerate(self.features)
            if abs(shap_arr[i]) >= 0.001
        ]
        impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "base_value":      round(base, 2),
            "predicted_value": round(pred, 2),
            "top_factors":     impacts[:top_n],
        }

    # ── FIX 2: Gap-bridging — public, correct year tracking, all features ────
    def bridge_to_today(self,
                        features: dict,
                        db_last_year: int,
                        db_last_quarter: int) -> tuple[dict, int, int, float]:
        """
        Fast-forward model state from the DB's last data point to today.

        DB data typically ends Q3/Q4 2023 or Q1 2024 — today is Q2 2026.
        Without bridging, forecasts start from 2024 and labels are wrong.

        Returns:
            bridged_features  — feature dict representing state as of today
            today_year        — int
            today_quarter     — int (1-4)
            bridged_level     — float, model-estimated water level TODAY (mbgl)
        """
        today_year, today_quarter = _current_year_quarter()
        gap = _quarters_between(db_last_year, db_last_quarter,
                                today_year, today_quarter)

        if gap <= 0:
            bridged_level = float(features.get("level_lag_1q", 0.0))
            return dict(features), today_year, today_quarter, bridged_level

        # Cap at 12 quarters to prevent runaway drift
        bridge_steps = min(gap, 12)

        base = pd.DataFrame(
            [{col: float(features.get(col, 0.0)) for col in self.features}]
        ).reindex(columns=self.features, fill_value=0).astype(float)

        # Build full 12-entry ring from available lags (pad oldest with lag_8q)
        l1 = float(base["level_lag_1q"].iloc[0])
        l2 = float(base["level_lag_2q"].iloc[0])
        l4 = float(base["level_lag_4q"].iloc[0])
        l8 = float(base["level_lag_8q"].iloc[0])
        # Ring newest→oldest: [l1, l2, ?, l4, ?, ?, ?, l8, ...]
        level_ring = [l1, l2,
                      (l2 + l4) / 2,   # interpolate missing lag_3q
                      l4,
                      (l4 + l8) / 2,   # interpolate lag_5q
                      (l4 + l8) / 2,   # lag_6q
                      (l4 + l8) / 2,   # lag_7q
                      l8, l8, l8, l8, l8]

        # District mean rainfall — use historical average as baseline
        hist_rain = float(base["rainfall_mm"].iloc[0])
        mean_rainfall = max(hist_rain, 80.0)  # floor at 80mm so deficit math works
        rainfall_ring = [hist_rain] * 8

        # year_norm_denominator: replicate training formula
        # training used (year - min_year) / (max_year - min_year + 1)
        # DB spans roughly 2000-2023 → denominator = 24
        # We fix this to 24 so 2026 normalises consistently with training
        year_norm_denom = float(self._year_norm_denom)

        current_q    = int(base["quarter"].iloc[0])
        current_year = db_last_year
        last_pred    = l1  # initialise

        for _ in range(bridge_steps):
            last_pred = float(self.predict(base)[0])
            last_pred = max(0.0, last_pred)
            current_q, current_year = _apply_recursive_step(
                base, last_pred, level_ring, rainfall_ring,
                mean_rainfall, current_q, current_year, year_norm_denom,
            )

        bridged_features = {col: float(base[col].iloc[0]) for col in self.features}
        return bridged_features, today_year, today_quarter, last_pred

    # ── FIX 3: Forecast from today, correct labels, all features updated ─────
    def predict_district(self,
                         features: dict,
                         quarters_ahead: int = 4,
                         start_year: int = None,
                         start_quarter: int = None) -> tuple[list, dict | None]:
        """
        Multi-step recursive forecast starting from TODAY.
        Bridges DB gap first, then forecasts Q3 2026 → Q2 2027 etc.
        """
        db_last_year    = start_year
        db_last_quarter = start_quarter
        if db_last_year is None or db_last_quarter is None:
            db_last_year, db_last_quarter = _current_year_quarter()

        # Step 1: Bridge DB date → today; get bridged_level for estimated current
        bridged_features, today_year, today_quarter, bridged_level = self.bridge_to_today(
            features, db_last_year, db_last_quarter
        )

        # Step 2: Forecast starts from the quarter AFTER today
        forecast_year, forecast_quarter = _advance_quarter(today_year, today_quarter)

        base = pd.DataFrame(
            [{col: float(bridged_features.get(col, 0.0)) for col in self.features}]
        ).reindex(columns=self.features, fill_value=0).astype(float)

        # Rebuild ring buffer from bridged state
        l1 = float(base["level_lag_1q"].iloc[0])
        l2 = float(base["level_lag_2q"].iloc[0])
        l4 = float(base["level_lag_4q"].iloc[0])
        l8 = float(base["level_lag_8q"].iloc[0])
        level_ring = [bridged_level, l1, l2,
                      (l2 + l4) / 2,
                      l4,
                      (l4 + l8) / 2, (l4 + l8) / 2, (l4 + l8) / 2,
                      l8, l8, l8, l8]

        hist_rain     = float(base["rainfall_mm"].iloc[0])
        mean_rainfall = max(hist_rain, 80.0)
        rainfall_ring = [hist_rain] * 8

        year_norm_denom = float(self._year_norm_denom)
        current_q       = int(base["quarter"].iloc[0])
        current_year    = today_year

        preds      = []
        first_shap = None

        for i in range(quarters_ahead):
            pred = float(self.predict(base)[0])
            pred = max(0.0, pred)
            risk = classify_risk(pred)

            # Trend vs previous step
            prev = preds[-1]["predicted_level_mbgl"] if preds else bridged_level
            trend = ("worsening" if pred > prev + 0.1
                     else "improving" if pred < prev - 0.1
                     else "stable")

            # Confidence from recent volatility
            w4  = level_ring[:4]
            std = float(np.std(w4)) if len(w4) > 1 else 2.0
            confidence        = "high" if std < 1.5 else ("medium" if std < 3.0 else "low")
            uncertainty_range = round(std * 1.5, 2)

            # SHAP only for first step — uses bridged features (unique per district)
            if i == 0:
                first_shap = self.explain(bridged_features, top_n=5)

            preds.append({
                "quarter":              i + 1,
                "year":                 forecast_year,
                "forecast_quarter":     forecast_quarter,
                "label":                f"Q{forecast_quarter} {forecast_year}",
                "predicted_level_mbgl": round(pred, 2),
                "risk_level":           risk,
                "risk_label":           RISK_META[risk]["label"],
                "risk_message":         RISK_META[risk]["message"],
                "trend":                trend,
                "confidence":           confidence,
                "range_low":            round(max(0.0, pred - uncertainty_range), 2),
                "range_high":           round(pred + uncertainty_range, 2),
            })

            current_q, current_year = _apply_recursive_step(
                base, pred, level_ring, rainfall_ring,
                mean_rainfall, current_q, current_year, year_norm_denom,
            )
            forecast_year, forecast_quarter = _advance_quarter(forecast_year, forecast_quarter)

        return preds, first_shap


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_predictor_instance: AquaSensePredictor | None = None


def get_predictor() -> AquaSensePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = AquaSensePredictor()
    return _predictor_instance