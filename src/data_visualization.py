"""
data_visualization.py
----------------------
Loads the cleaned dataset and generates all EDA charts, saving each
one as a PNG into outputs/figures/.

Run:
    python src/data_visualization.py
(run data_cleaning.py first so data/processed/cleaned_data.csv exists)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "cleaned_data.csv"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (9, 5)


def save_fig(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


def plot_sales_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df['Item_Outlet_Sales'], bins=40, kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title("Distribution of Item_Outlet_Sales")
    sns.boxplot(x=df['Item_Outlet_Sales'], ax=axes[1], color='steelblue')
    axes[1].set_title("Boxplot of Item_Outlet_Sales")
    fig.tight_layout()
    save_fig(fig, "01_sales_distribution.png")


def plot_sales_by_outlet(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    order1 = df.groupby('Outlet_Type')['Item_Outlet_Sales'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='Outlet_Type', y='Item_Outlet_Sales', order=order1, ax=axes[0], hue='Outlet_Type', palette='viridis', legend=False)
    axes[0].set_title("Sales by Outlet Type")
    axes[0].tick_params(axis='x', rotation=30)

    order2 = df.groupby('Outlet_Size')['Item_Outlet_Sales'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='Outlet_Size', y='Item_Outlet_Sales', order=order2, ax=axes[1], hue='Outlet_Size', palette='mako', legend=False)
    axes[1].set_title("Sales by Outlet Size")

    order3 = df.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x='Outlet_Location_Type', y='Item_Outlet_Sales', order=order3, ax=axes[2], hue='Outlet_Location_Type', palette='flare', legend=False)
    axes[2].set_title("Sales by Location Tier")

    fig.tight_layout()
    save_fig(fig, "02_sales_by_outlet.png")


def plot_category_sales(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    cat_sales = df.groupby('Item_Category')['Item_Outlet_Sales'].mean().sort_values(ascending=False)
    sns.barplot(x=cat_sales.values, y=cat_sales.index, ax=axes[0], hue=cat_sales.index, palette='crest', legend=False)
    axes[0].set_title("Average Sales by Item Category")
    axes[0].set_xlabel("Average Item_Outlet_Sales")

    type_sales = df.groupby('Item_Type')['Item_Outlet_Sales'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=type_sales.values, y=type_sales.index, ax=axes[1], hue=type_sales.index, palette='crest', legend=False)
    axes[1].set_title("Top 10 Item Types by Average Sales")
    axes[1].set_xlabel("Average Item_Outlet_Sales")

    fig.tight_layout()
    save_fig(fig, "03_category_sales.png")


def plot_price_vs_sales(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(data=df, x='Item_MRP', y='Item_Outlet_Sales', hue='Outlet_Type',
                     alpha=0.5, ax=axes[0], palette='tab10')
    axes[0].set_title("Item MRP vs. Sales (colored by Outlet Type)")

    numeric_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age', 'Item_Outlet_Sales']
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[1])
    axes[1].set_title("Correlation Heatmap")

    fig.tight_layout()
    save_fig(fig, "04_price_vs_sales_correlation.png")


def plot_sales_by_store_age(df):
    age_sales = df.groupby('Outlet_Establishment_Year')['Item_Outlet_Sales'].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(x=age_sales.index, y=age_sales.values, marker='o', color='darkorange', ax=ax)
    ax.set_title("Average Sales by Outlet Establishment Year")
    ax.set_ylabel("Average Item_Outlet_Sales")
    ax.set_xlabel("Outlet Establishment Year")
    fig.tight_layout()
    save_fig(fig, "05_sales_by_store_age.png")


def main():
    df = pd.read_csv(PROCESSED_PATH)
    print(f"Loaded cleaned data: {df.shape[0]} rows")

    plot_sales_distribution(df)
    plot_sales_by_outlet(df)
    plot_category_sales(df)
    plot_price_vs_sales(df)
    plot_sales_by_store_age(df)

    print(f"\n✅ All charts saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
