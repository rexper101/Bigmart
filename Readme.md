# 🛒 BigMart Sales Prediction

A full-stack machine learning web application that predicts retail product sales based on
product and store attributes.

- **Backend**: Flask REST API
- **Frontend**: Streamlit + Plotly
- **ML Model**: Random Forest Regressor (scikit-learn), compared against Linear Regression and XGBoost
- **Dataset**: `data/BigMart_Sales_Dataset.csv` — [Big Mart Sales Data (Kaggle)](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data), 8,523 rows across 10 stores

> **Note:** The CSV was obtained from a verified GitHub mirror of the Kaggle dataset (Kaggle's
> own download requires an authenticated session, not reachable from a script). If your
> internship requires the file pulled directly from Kaggle, download it from the link above
> and swap it into `data/` — the columns are identical, so nothing else needs to change.

---

## Project Structure

```
BigMart_Sales_Project/
├── data/
│   └── BigMart_Sales_Dataset.csv     # Dataset
├── model/
│   ├── model.pkl                     # Trained Random Forest model (generated)
│   ├── encoders.pkl                  # LabelEncoders for categorical features (generated)
│   ├── metadata.json                 # Feature list, metrics, category options (generated)
│   ├── model_results.csv             # Model comparison table (generated)
│   ├── diagnostics.png               # Actual-vs-predicted + residuals plot (generated)
│   ├── model_comparison.png          # R² bar chart (generated)
│   └── feature_importance.png        # Feature importance chart (generated)
├── backend/
│   └── app.py                        # Flask REST API
├── frontend/
│   └── ui.py                         # Streamlit UI
├── train_model.py                    # Model training script
├── requirements.txt
├── README.md
├── BigMart_Sales_Prediction_Report.docx   # Formal written report
└── extras/                           # Bonus: the original in-depth EDA notebook
    ├── BigMart_Sales_Analysis.ipynb
    ├── BigMart_Sales_Analysis.html
    └── src/                          # same analysis as standalone cleaning/viz/training scripts
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```
This cleans the raw data, engineers features, trains & compares 3 models, and saves
`model/model.pkl`, `model/encoders.pkl`, `model/metadata.json`, plus 3 chart images.

### 3. Start the Flask backend
```bash
python backend/app.py
```
API runs at **http://localhost:5000**

### 4. Launch the Streamlit frontend
*(open a second terminal)*
```bash
streamlit run frontend/ui.py
```
UI opens at **http://localhost:8501**

> On Windows Command Prompt, use `python -m venv venv`, `venv\Scripts\activate.bat`, then the
> same `pip install` / `python` / `streamlit` commands above.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/predict` | Predict sales for a product/store combination |
| GET | `/dataset` | Return the cleaned dataset as JSON (`?limit=` & `?offset=` supported) |
| GET | `/model_info` | Model type, metrics, feature importance, valid category options |

### POST `/predict` — example
```json
// Request
{
  "Item_Weight": 9.3,
  "Item_Visibility": 0.016,
  "Item_MRP": 249.8,
  "Outlet_Establishment_Year": 1999,
  "Item_Fat_Content": "Low Fat",
  "Item_Type": "Dairy",
  "Outlet_Size": "Medium",
  "Outlet_Location_Type": "Tier 1",
  "Outlet_Type": "Supermarket Type1"
}

// Response
{
  "predicted_sales": 4394.23,
  "currency": "INR",
  "derived_features": { "Outlet_Age": 14, "Item_Category": "Food" },
  "model_used": "Random Forest"
}
```

---

## Frontend Pages

| Page | Description |
|---|---|
| **Predict Sales** | Input product & store attributes, get an instant sales prediction + gauge chart |
| **Dataset Explorer** | Browse the data; distribution, boxplot, scatter, and correlation-heatmap charts |
| **Model Insights** | R², MAE, RMSE, model comparison, feature importance, and diagnostics plot |

---

## Model Performance

| Model | RMSE | MAE | R² Score |
|---|---|---|---|
| **Random Forest (deployed)** | 1022.68 | 716.03 | **0.6152** |
| XGBoost | 1033.55 | 723.03 | 0.6070 |
| Linear Regression | 1142.93 | 856.35 | 0.5194 |

**Top predictive features:** `Item_MRP` and `Outlet_Type` dominate — together they explain the
large majority of the model's predictive power. Note that `Outlet_Identifier` and a
price-quartile bucket used in the exploratory notebook were deliberately excluded from the
deployed model so predictions generalize to hypothetical new stores, not just the 10 seen
during training. Dropping them cost virtually no accuracy (0.6154 → 0.6152 R²).

---

## Business Insights

1. Store format (`Outlet_Type`) and price (`Item_MRP`) are the two biggest sales drivers.
2. Grocery Stores dramatically underperform Supermarkets — upgrading store format likely has
   more impact than product-level tweaks.
3. Store age has little effect on sales; the oldest outlet is actually the weakest performer.
4. Item weight, fat content, and shelf visibility matter little — focus on pricing and format.
5. The model can flag under-performing product-store combinations for manual review.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML | scikit-learn, XGBoost, numpy |
| Backend | Flask, Flask-CORS |
| Frontend | Streamlit, Plotly |
| Data | pandas, matplotlib, seaborn |
| Report | python-docx (docx-js) |

## Documents

- **`BigMart_Sales_Prediction_Report.docx`** — formal written report covering objective,
  dataset, methodology, results, business insights, and system architecture.
- **`extras/BigMart_Sales_Analysis.ipynb` / `.html`** — the original in-depth exploratory
  notebook, kept as supplementary material with additional charts and narrative.

## Possible Extensions

- Hyperparameter tuning (GridSearchCV / Optuna) to push R² higher
- SHAP values for per-prediction explainability
- Deploy the Flask + Streamlit app to a cloud host (Render, Streamlit Community Cloud, etc.)
- Incorporate external data (seasonality, promotions) — this dataset is a single-year snapshot
