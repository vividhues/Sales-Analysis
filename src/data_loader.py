# loading, validating, and pre-processing the raw CSV.
import pandas as pd
from config.settings import Settings


def load_sales_data(filepath: str | None = None) -> pd.DataFrame:
    path = filepath or str(Settings.sales_csv)
    df = pd.read_csv(path, parse_dates=[Settings.date_column])

    required = {"date", "region", "product", "channel", "sales", "profit", "units"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    df["month"] = df[Settings.date_column].dt.to_period("M")
    df["profit_margin"] = (df["profit"] / df["sales"]).round(4)
    df["revenue_per_unit"] = (df["sales"] / df["units"]).round(2)

    df.sort_values(Settings.date_column, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_summary(df: pd.DataFrame) -> dict:
    return {
        "total_records": len(df),
        "date_range": f"{df['date'].min().date()} → {df['date'].max().date()}",
        "total_sales": df["sales"].sum(),
        "total_profit": df["profit"].sum(),
        "avg_profit_margin": round(df["profit_margin"].mean(), 4),
        "products": df["product"].unique().tolist(),
        "regions": df["region"].unique().tolist(),
        "channels": df["channel"].unique().tolist(),
    }