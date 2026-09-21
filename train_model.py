"""
train_model.py
---------------
End-to-end training pipeline for the BigMart Sales Prediction app.

Loads the raw dataset, cleans it, engineers features, trains and compares
three regression models, then saves everything the Flask backend needs to
serve live predictions:

    model/model.pkl        <- best trained model (Random Forest)
    model/encoders.pkl     <- LabelEncoders for each categorical column
    model/metadata.json    <- feature list, categorical options, metrics
    model/diagnostics.png  <- actual-vs-predicted + residuals plot

Run:
    python train_model.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "BigMart_Sales_Dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Features used by the deployed app. Outlet_Identifier is deliberately
# excluded (it only identifies 10 specific existing stores and wouldn't
# generalize to a hypothetical new store); Item_MRP already captures price,
# so the redundant MRP_Segment bucket is dropped too.
CATEGORICAL_COLS = ["Item_Fat_Content", "Item_Type", "Outlet_Size",
                     "Outlet_Location_Type", "Outlet_Type", "Item_Category"]
NUMERIC_COLS = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Age"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
TARGET_COL = "Item_Outlet_Sales"

FOOD_TYPES = ["Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Seafood",
              "Starchy Foods", "Breads", "Frozen Foods", "Snack Foods",
              "Canned", "Baking Goods"]
DRINK_TYPES = ["Soft Drinks", "Hard Drinks"]


def categorize(item_type):
    if item_type in FOOD_TYPES:
        return "Food"
    if item_type in DRINK_TYPES:
        return "Drinks"
    return "Non-Consumable"


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix inconsistent Item_Fat_Content labels
    df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
        {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
    )

    # Impute Item_Weight using the mean weight of that specific product
    weight_map = df.groupby("Item_Identifier")["Item_Weight"].mean()
    df["Item_Weight"] = df.apply(
        lambda r: weight_map.get(r["Item_Identifier"], np.nan)
        if pd.isna(r["Item_Weight"]) else r["Item_Weight"],
        axis=1,
    )
    df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Weight"].mean())

    # Impute Outlet_Size using the most common size for that Outlet_Type
    mode_map = (df.dropna(subset=["Outlet_Size"])
                  .groupby("Outlet_Type")["Outlet_Size"]
                  .agg(lambda x: x.mode()[0]))
    df["Outlet_Size"] = df.apply(
        lambda r: mode_map.get(r["Outlet_Type"], "Medium")
        if pd.isna(r["Outlet_Size"]) else r["Outlet_Size"],
        axis=1,
    )

    # Feature engineering
    df["Outlet_Age"] = 2013 - df["Outlet_Establishment_Year"]
    df["Item_Category"] = df["Item_Type"].apply(categorize)

    return df


def main():
    print(f"Loading raw data from {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    df = clean_and_engineer(df)
    print("Cleaning + feature engineering complete. Missing values remaining:",
          df[FEATURE_COLS + [TARGET_COL]].isnull().sum().sum())

    # Save categorical options for the frontend dropdowns, before encoding
    categorical_options = {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_COLS}

    # Encode categoricals
    model_df = df.copy()
    encoders = {}
    for c in CATEGORICAL_COLS:
        le = LabelEncoder()
        model_df[c] = le.fit_transform(model_df[c].astype(str))
        encoders[c] = le

    X = model_df[FEATURE_COLS]
    y = model_df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    def evaluate(name, model):
        pred = model.predict(X_test)
        return {
            "Model": name,
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
            "MAE": round(float(mean_absolute_error(y_test, pred)), 2),
            "R2 Score": round(float(r2_score(y_test, pred)), 4),
        }

    results = []
    models = {}

    lr = LinearRegression().fit(X_train, y_train)
    models["Linear Regression"] = lr
    results.append(evaluate("Linear Regression", lr))

    rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=25, random_state=42)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf
    results.append(evaluate("Random Forest", rf))

    if HAS_XGB:
        xgbm = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42)
        xgbm.fit(X_train, y_train)
        models["XGBoost"] = xgbm
        results.append(evaluate("XGBoost", xgbm))

    results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False).reset_index(drop=True)
    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = models[best_name]
    best_metrics = results_df.iloc[0].to_dict()
    print(f"\nBest model: {best_name}")

    # Feature importance (tree models only)
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS) \
                        .sort_values(ascending=False)
    else:
        importances = pd.Series(np.abs(best_model.coef_), index=FEATURE_COLS) \
                        .sort_values(ascending=False)

    # --- Save artifacts ---
    joblib.dump(best_model, MODEL_DIR / "model.pkl")
    joblib.dump(encoders, MODEL_DIR / "encoders.pkl")

    metadata = {
        "model_type": best_name,
        "feature_cols": FEATURE_COLS,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "categorical_options": categorical_options,
        "metrics": {
            "RMSE": best_metrics["RMSE"],
            "MAE": best_metrics["MAE"],
            "R2": best_metrics["R2 Score"],
        },
        "all_model_results": results_df.to_dict(orient="records"),
        "feature_importance": importances.round(4).to_dict(),
        "reference_year": 2013,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    results_df.to_csv(MODEL_DIR / "model_results.csv", index=False)

    # --- Diagnostics plot: actual vs predicted + residuals ---
    pred_test = best_model.predict(X_test)
    residuals = y_test - pred_test

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_test, pred_test, alpha=0.4, color="steelblue", edgecolor="none")
    lims = [0, max(y_test.max(), pred_test.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    axes[0].set_xlabel("Actual Sales")
    axes[0].set_ylabel("Predicted Sales")
    axes[0].set_title(f"Actual vs. Predicted ({best_name})")
    axes[0].legend()

    axes[1].hist(residuals, bins=40, color="darkorange", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual (Actual − Predicted)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "diagnostics.png", dpi=120)
    plt.close()

    print(f"\n✅ Saved model.pkl, encoders.pkl, metadata.json, model_results.csv, diagnostics.png to {MODEL_DIR}")


if __name__ == "__main__":
    main()
