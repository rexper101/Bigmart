Market Sales Analysis using AI/ML — Big Mart Sales Dataset

An internship-style data analysis project structured around three clear stages — Data Cleaning → Data Visualization → Data Training (AI/ML) — that together uncover what drives product sales across stores and predict sales using machine learning.

📌 Objective

BigMart operates supermarkets and grocery stores across multiple cities. Using 2013 sales data for 1,559 products across 10 stores, this project:

Cleans messy real-world retail data (missing values, inconsistent labels)
Visualizes the key patterns and relationships behind product sales
Trains and compares AI/ML models to predict Item_Outlet_Sales
Turns the findings into actionable business recommendations
📁 Folder Structure
BigMart_Sales_Analysis_Project/
├── README.md                              <- you are here
├── requirements.txt                       <- Python dependencies
├── BigMart_Sales_Analysis.ipynb           <- all-in-one notebook version (fully executed)
├── BigMart_Sales_Analysis.html            <- read-only HTML export (view without Jupyter)
│
├── src/                                   <- the three pipeline stages, as standalone scripts
│   ├── data_cleaning.py                   <- Stage 1: clean + feature-engineer the raw data
│   ├── data_visualization.py              <- Stage 2: generate all EDA charts
│   └── model_training.py                  <- Stage 3: train, evaluate & save the ML model
│
├── data/
│   ├── raw/
│   │   └── BigMart_Sales_Dataset.csv      <- original dataset (input)
│   └── processed/
│       └── cleaned_data.csv               <- output of data_cleaning.py (created when run)
│
└── outputs/                                <- created when you run the scripts
    ├── figures/                            <- all PNG charts from visualization + training
    ├── model_results.csv                   <- RMSE / MAE / R² for each model
    └── best_model.pkl                      <- trained Random Forest model, ready to reload
📊 Dataset
Source: Big Mart Sales Data — Kaggle
Rows: 8,523 | Columns: 12 (raw) → 15 (after feature engineering)
Target variable: Item_Outlet_Sales
Column	Description
Item_Identifier	Unique product ID
Item_Weight	Weight of the product
Item_Fat_Content	Low Fat / Regular
Item_Visibility	% of total display area allocated to the item in a store
Item_Type	Product category (Dairy, Snacks, Household, etc.)
Item_MRP	Maximum Retail Price
Outlet_Identifier	Unique store ID
Outlet_Establishment_Year	Year the store was opened
Outlet_Size	Small / Medium / High
Outlet_Location_Type	Tier 1 / Tier 2 / Tier 3 city
Outlet_Type	Grocery Store / Supermarket Type1-3
Item_Outlet_Sales	Target — sales of the product in that store
🧭 Stage 1 — Data Cleaning

Standardizes Item_Fat_Content labels, imputes missing Item_Weight/Outlet_Size, engineers Outlet_Age, Item_Category, MRP_Segment.

📈 Stage 2 — Data Visualization

Saves 5 chart panels to outputs/figures/: sales distribution, sales by outlet type/size/location, category sales, price-vs-sales + correlation heatmap, sales by store age.

🤖 Stage 3 — Data Training (AI/ML)

Trains & compares Linear Regression, Random Forest, and XGBoost:

Model	RMSE	MAE	R² Score
Random Forest	1022.40	715.98	0.6154
XGBoost	1029.84	720.38	0.6098
Linear Regression	1142.86	855.96	0.5194
💡 Key Business Insights
Store format (Outlet_Type) and price (Item_MRP) are the two biggest sales drivers.
Grocery Stores dramatically underperform Supermarkets.
Store age has little effect on sales.
Item weight, fat content, and shelf visibility matter little.
The model can flag under-performing product-store combos for manual review.
🛠️ Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · XGBoost · joblib · Jupyter