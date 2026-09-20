"""
data_cleaning.py
-----------------
Loads the raw BigMart sales data, fixes inconsistent category labels,
imputes missing values, engineers a few extra features, and saves a
clean, model-ready CSV to data/processed/.

Run:
    python src/data_cleaning.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "BigMart_Sales_Dataset.csv"
PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "cleaned_data.csv"

def impute_outlet_size(df: pd.DataFrame) -> pd.DataFrame:
    mode_map = (df.dropna(subset=['Outlet_Size'])
                  .groupby('Outlet_Type')['Outlet_Size']
                  .agg(lambda x: x.mode()[0]))
    print("\nMost common Outlet_Size per Outlet_Type:")
    print(mode_map)

    df['Outlet_Size'] = df.apply(
        lambda r: mode_map.get(r['Outlet_Type'], 'Medium') if pd.isna(r['Outlet_Size']) else r['Outlet_Size'],
        axis=1
    )
    print(f"\nOutlet_Size missing after imputation: {df['Outlet_Size'].isnull().sum()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['Outlet_Age'] = 2013 - df['Outlet_Establishment_Year']

    food_types = ['Dairy', 'Meat', 'Fruits and Vegetables', 'Breakfast', 'Seafood',
                  'Starchy Foods', 'Breads', 'Frozen Foods', 'Snack Foods', 'Canned', 'Baking Goods']
    drink_types = ['Soft Drinks', 'Hard Drinks']

    def categorize(item_type):
        if item_type in food_types:
            return 'Food'
        elif item_type in drink_types:
            return 'Drinks'
        return 'Non-Consumable'

    df['Item_Category'] = df['Item_Type'].apply(categorize)
    df['MRP_Segment'] = pd.qcut(df['Item_MRP'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])

    print("\nEngineered features added: Outlet_Age, Item_Category, MRP_Segment")
    return df


def main():
    df = load_data(RAW_PATH)
    df = clean_item_fat_content(df)
    df = impute_item_weight(df)
    df = impute_outlet_size(df)
    df = engineer_features(df)

    assert df.isnull().sum().sum() == 0, "There are still missing values after cleaning!"

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\n✅ Cleaned data saved to: {PROCESSED_PATH}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()
