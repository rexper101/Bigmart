"""
model_training.py
-------------------
Loads the cleaned dataset, trains and compares Linear Regression,
Random Forest, and XGBoost models to predict Item_Outlet_Sales,
saves the evaluation results, a feature-importance chart, and the
best model to disk.

Run:
    python src/model_training.py
(run data_cleaning.py first so data/processed/cleaned_data.csv exists)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "cleaned_data.csv"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODEL_PATH = OUTPUTS_DIR / "best_model.pkl"
RESULTS_PATH = OUTPUTS_DIR / "model_results.csv"

FEATURE_COLS = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP',
                'Outlet_Identifier', 'Outlet_Age', 'Outlet_Size', 'Outlet_Location_Type',
                'Outlet_Type', 'Item_Category', 'MRP_Segment']
CAT_COLS = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size',
            'Outlet_Location_Type', 'Outlet_Type', 'Item_Category', 'MRP_Segment']
TARGET = 'Item_Outlet_Sales'


def encode_features(df: pd.DataFrame):
    df = df.copy()
    for c in CAT_COLS:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
    return df


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"{name:20s} RMSE: {rmse:8.2f}  MAE: {mae:8.2f}  R2: {r2:.4f}")
    return {'Model': name, 'RMSE': round(rmse, 2), 'MAE': round(mae, 2), 'R2 Score': round(r2, 4)}


def main():
    df = pd.read_csv(PROCESSED_PATH)
    print(f"Loaded cleaned data: {df.shape[0]} rows")

    model_df = encode_features(df)
    X = model_df[FEATURE_COLS]
    y = model_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape} | Test size: {X_test.shape}\n")

    results = []

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results.append(evaluate("Linear Regression", lr, X_test, y_test))

    rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=25, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    xgbm = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgbm.fit(X_train, y_train)
    results.append(evaluate("XGBoost", xgbm, X_test, y_test))

    results_df = pd.DataFrame(results).sort_values('R2 Score', ascending=False).reset_index(drop=True)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n✅ Results saved to: {RESULTS_PATH}")
    print(results_df)

    # Model comparison chart
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=results_df, x='Model', y='R2 Score', hue='Model', palette='viridis', legend=False, ax=ax)
    ax.set_title("Model Comparison — R² Score (higher is better)")
    ax.set_ylim(0, max(results_df['R2 Score']) + 0.1)
    for i, v in enumerate(results_df['R2 Score']):
        ax.text(i, v + 0.01, str(v), ha='center', fontweight='bold')
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "06_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Feature importance chart (from Random Forest, the best model)
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette='crest', legend=False, ax=ax)
    ax.set_title("Feature Importance (Random Forest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Charts saved to: {FIGURES_DIR}")

    # Save the best model (Random Forest)
    joblib.dump(rf, MODEL_PATH)
    print(f"✅ Best model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
