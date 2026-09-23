"""
frontend/ui.py
---------------
Streamlit UI for the BigMart Sales Prediction project.
Talks to the Flask backend (backend/app.py) running at BACKEND_URL.

Run (after starting the backend):
    streamlit run frontend/ui.py
UI opens at http://localhost:8501
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

BACKEND_URL = "http://localhost:5000"
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"

st.set_page_config(page_title="BigMart Sales Prediction", page_icon="🛒", layout="wide")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_model_info():
    r = requests.get(f"{BACKEND_URL}/model_info", timeout=10)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def fetch_dataset():
    r = requests.get(f"{BACKEND_URL}/dataset", timeout=30)
    r.raise_for_status()
    payload = r.json()
    return pd.DataFrame(payload["data"])


def backend_alive():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.title("🛒 BigMart Sales")
page = st.sidebar.radio("Navigate", ["Predict Sales", "Dataset Explorer", "Model Insights"])

if not backend_alive():
    st.error(
        f"⚠️ Can't reach the backend API at {BACKEND_URL}. "
        "Start it first with:  `python backend/app.py`"
    )
    st.stop()

info = fetch_model_info()

# ---------------------------------------------------------------------
# Page 1: Predict Sales
# ---------------------------------------------------------------------
if page == "Predict Sales":
    st.title("🔮 Predict Item Outlet Sales")
    st.write(
        "Enter a product's and store's attributes to predict how much that "
        f"item would sell in that store (model: **{info['model_type']}**, "
        f"R² ≈ {info['metrics']['R2']})."
    )

    options = info["categorical_options"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Product")
        item_weight = st.number_input("Item Weight (kg)", min_value=0.0, max_value=30.0, value=12.5, step=0.1)
        item_mrp = st.number_input("Item MRP (₹)", min_value=1.0, max_value=300.0, value=140.0, step=1.0)
        item_visibility = st.slider("Item Visibility (share of display area)", 0.0, 0.35, 0.05, 0.01)
        item_fat_content = st.selectbox("Item Fat Content", options["Item_Fat_Content"])
        item_type = st.selectbox("Item Type", options["Item_Type"])

    with col2:
        st.subheader("Store")
        outlet_year = st.slider("Outlet Establishment Year", 1985, 2013, 2005)
        outlet_size = st.selectbox("Outlet Size", options["Outlet_Size"])
        outlet_location = st.selectbox("Outlet Location Type", options["Outlet_Location_Type"])
        outlet_type = st.selectbox("Outlet Type", options["Outlet_Type"])

    if st.button("Predict Sales", type="primary"):
        payload = {
            "Item_Weight": item_weight,
            "Item_Visibility": item_visibility,
            "Item_MRP": item_mrp,
            "Outlet_Establishment_Year": outlet_year,
            "Item_Fat_Content": item_fat_content,
            "Item_Type": item_type,
            "Outlet_Size": outlet_size,
            "Outlet_Location_Type": outlet_location,
            "Outlet_Type": outlet_type,
        }
        try:
            r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
            r.raise_for_status()
            result = r.json()

            predicted = result["predicted_sales"]

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=predicted,
                title={"text": "Predicted Item Outlet Sales (₹)"},
                gauge={
                    "axis": {"range": [0, max(8000, predicted * 1.3)]},
                    "bar": {"color": "#2E7D32"},
                    "steps": [
                        {"range": [0, 2000], "color": "#FFE0E0"},
                        {"range": [2000, 5000], "color": "#FFF3CD"},
                        {"range": [5000, max(8000, predicted * 1.3)], "color": "#D4EDDA"},
                    ],
                },
            ))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, width='stretch')

            st.success(f"Predicted sales: **₹{predicted:,.2f}**")
            st.caption(
                f"Derived features used: Outlet_Age = {result['derived_features']['Outlet_Age']} years, "
                f"Item_Category = {result['derived_features']['Item_Category']}"
            )
        except requests.exceptions.HTTPError:
            st.error(f"Prediction failed: {r.json().get('error', r.text)}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach backend: {e}")

# ---------------------------------------------------------------------
# Page 2: Dataset Explorer
# ---------------------------------------------------------------------
elif page == "Dataset Explorer":
    st.title("📊 Dataset Explorer")

    with st.spinner("Loading dataset from backend..."):
        df = fetch_dataset()

    st.write(f"**{len(df):,} rows** — 2013 sales data for BigMart products and stores.")
    st.dataframe(df.head(50), width='stretch')

    col1, col2 = st.columns(2)

    with col1:
        fig_hist = px.histogram(df, x="Item_Outlet_Sales", nbins=40,
                                 title="Distribution of Item Outlet Sales")
        st.plotly_chart(fig_hist, width='stretch')

        fig_scatter = px.scatter(df, x="Item_MRP", y="Item_Outlet_Sales", color="Outlet_Type",
                                  opacity=0.5, title="Item MRP vs. Sales")
        st.plotly_chart(fig_scatter, width='stretch')

    with col2:
        fig_box = px.box(df, x="Outlet_Type", y="Item_Outlet_Sales", color="Outlet_Type",
                          title="Sales by Outlet Type")
        st.plotly_chart(fig_box, width='stretch')

        numeric_cols = ["Item_Weight", "Item_Visibility", "Item_MRP", "Outlet_Age", "Item_Outlet_Sales"]
        corr = df[numeric_cols].corr()
        fig_heatmap = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                 title="Correlation Heatmap")
        st.plotly_chart(fig_heatmap, width='stretch')

   