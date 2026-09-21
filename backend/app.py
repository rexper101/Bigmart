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

@app.route("/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "model_type": metadata["model_type"],
        "metrics": metadata["metrics"],
        "all_model_results": metadata["all_model_results"],
        "feature_importance": metadata["feature_importance"],
        "feature_cols": FEATURE_COLS,
        "categorical_options": CATEGORICAL_OPTIONS,
        "reference_year": REFERENCE_YEAR,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
