"""
backend/app.py
---------------
Flask REST API for the BigMart Sales Prediction project.

Endpoints:
    GET  /health       -> health check
    POST /predict       -> predict Item_Outlet_Sales for a product/store combination
    GET  /dataset        -> return the (cleaned) dataset as JSON
    GET  /model_info     -> model type, metrics, feature importance, valid category options

Run:
    python backend/app.py
API runs at http://localhost:5000
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_PATH = BASE_DIR / "data" / "BigMart_Sales_Dataset.csv"

app = Flask(__name__)
CORS(app)  # allow the Streamlit frontend (different port) to call this API

# --- Load model artifacts once at startup ---
model = joblib.load(MODEL_DIR / "model.pkl")
encoders = joblib.load(MODEL_DIR / "encoders.pkl")
with open(MODEL_DIR / "metadata.json") as f:
    metadata = json.load(f)

FEATURE_COLS = metadata["feature_cols"]
NUMERIC_COLS = metadata["numeric_cols"]
CATEGORICAL_COLS = metadata["categorical_cols"]
CATEGORICAL_OPTIONS = metadata["categorical_options"]
REFERENCE_YEAR = metadata["reference_year"]

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


# Cache the cleaned dataset for the /dataset endpoint (same cleaning as training)
_dataset_cache = None


def get_clean_dataset():
    global _dataset_cache
    if _dataset_cache is None:
        df = pd.read_csv(DATA_PATH)
        df["Item_Fat_Content"] = df["Item_Fat_Content"].replace(
            {"LF": "Low Fat", "low fat": "Low Fat", "reg": "Regular"}
        )
        weight_map = df.groupby("Item_Identifier")["Item_Weight"].mean()
        df["Item_Weight"] = df.apply(
            lambda r: weight_map.get(r["Item_Identifier"], np.nan)
            if pd.isna(r["Item_Weight"]) else r["Item_Weight"], axis=1)
        df["Item_Weight"] = df["Item_Weight"].fillna(df["Item_Weight"].mean())
        mode_map = (df.dropna(subset=["Outlet_Size"])
                      .groupby("Outlet_Type")["Outlet_Size"]
                      .agg(lambda x: x.mode()[0]))
        df["Outlet_Size"] = df.apply(
            lambda r: mode_map.get(r["Outlet_Type"], "Medium")
            if pd.isna(r["Outlet_Size"]) else r["Outlet_Size"], axis=1)
        df["Outlet_Age"] = REFERENCE_YEAR - df["Outlet_Establishment_Year"]
        df["Item_Category"] = df["Item_Type"].apply(categorize)
        _dataset_cache = df
    return _dataset_cache


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    required = ["Item_Weight", "Item_Visibility", "Item_MRP",
                "Outlet_Establishment_Year", "Item_Fat_Content", "Item_Type",
                "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"]
    missing = [f for f in required if f not in payload]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    # Validate categorical values against what the model was trained on
    for col in ["Item_Fat_Content", "Item_Type", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"]:
        if payload[col] not in CATEGORICAL_OPTIONS[col]:
            return jsonify({
                "error": f"Invalid value '{payload[col]}' for {col}. "
                         f"Valid options: {CATEGORICAL_OPTIONS[col]}"
            }), 400

    try:
        item_weight = float(payload["Item_Weight"])
        item_visibility = float(payload["Item_Visibility"])
        item_mrp = float(payload["Item_MRP"])
        outlet_year = int(payload["Outlet_Establishment_Year"])
    except (TypeError, ValueError):
        return jsonify({"error": "Item_Weight, Item_Visibility, Item_MRP and "
                                  "Outlet_Establishment_Year must be numeric"}), 400

    outlet_age = REFERENCE_YEAR - outlet_year
    item_category = categorize(payload["Item_Type"])

    row = {
        "Item_Weight": item_weight,
        "Item_Visibility": item_visibility,
        "Item_MRP": item_mrp,
        "Outlet_Age": outlet_age,
        "Item_Fat_Content": encoders["Item_Fat_Content"].transform([payload["Item_Fat_Content"]])[0],
        "Item_Type": encoders["Item_Type"].transform([payload["Item_Type"]])[0],
        "Outlet_Size": encoders["Outlet_Size"].transform([payload["Outlet_Size"]])[0],
        "Outlet_Location_Type": encoders["Outlet_Location_Type"].transform([payload["Outlet_Location_Type"]])[0],
        "Outlet_Type": encoders["Outlet_Type"].transform([payload["Outlet_Type"]])[0],
        "Item_Category": encoders["Item_Category"].transform([item_category])[0],
    }

    X = pd.DataFrame([row])[FEATURE_COLS]
    prediction = float(model.predict(X)[0])
    prediction = max(prediction, 0.0)  # sales can't be negative

    return jsonify({
        "predicted_sales": round(prediction, 2),
        "currency": "INR",
        "derived_features": {
            "Outlet_Age": outlet_age,
            "Item_Category": item_category,
        },
        "model_used": metadata["model_type"],
    })


@app.route("/dataset", methods=["GET"])
def dataset():
    df = get_clean_dataset()
    limit = request.args.get("limit", default=None, type=int)
    offset = request.args.get("offset", default=0, type=int)

    total = len(df)
    subset = df.iloc[offset: offset + limit] if limit else df.iloc[offset:]

  




