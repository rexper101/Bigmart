# BigMart Sales Analysis Project

## Project Description
This project predicts BigMart outlet sales using product and store attributes. It combines data cleaning, feature engineering, machine learning, and a lightweight web application to provide sales predictions through a Flask API and Streamlit dashboard.

## Dataset
Dataset source: https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data

The project uses the local file in `data/BigMart_Sales_Dataset.csv`.

## Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- XGBoost
- Flask
- Streamlit
- Plotly
- Joblib

## Setup and Run Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model:
   ```bash
   python train_model.py
   ```
3. Start backend:
   ```bash
   python backend/app.py
   ```
4. Launch frontend:
   ```bash
   streamlit run frontend/ui.py
   ```
5. Open the app in a browser:
   - Backend: http://localhost:5000
   - Frontend: http://localhost:8501

## Key Information
- Best model: Random Forest Regressor
- Performance metric: R² score around 0.6152
- Project includes data cleaning, model comparison, feature importance, and prediction API
- Model artifacts are saved in the `model/` folder

## Project Structure
- `data/` — dataset files
- `backend/` — Flask API
- `frontend/` — Streamlit UI
- `model/` — trained model and metadata
- `train_model.py` — training script
- `requirements.txt` — dependencies
- `README.md` — project overview
