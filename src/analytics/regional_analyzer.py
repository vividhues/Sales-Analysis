import pandas as pd

class RegionalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def sales_by_region(self) -> pd.DataFrame:
        return (
            self.df.groupby("region")
            .agg(total_sales=("sales", "sum"),
                 total_profit=("profit", "sum"),
                 total_units=("units", "sum"),
                 avg_profit_margin=("profit_margin", "mean"))
            .sort_values("total_sales", ascending=False)
            .reset_index()
        )

    def best_region_per_product(self) -> pd.DataFrame:
        return (
            self.df.groupby(["product", "region"])["sales"]
            .sum()
            .reset_index()
            .sort_values(["product", "sales"], ascending=[True, False])
            .groupby("product")
            .first()
            .reset_index()
            .rename(columns={"region": "best_region", "sales": "region_sales"})
        )

    def regional_product_mix(self) -> pd.DataFrame:
        return self.df.pivot_table(
            index="region", columns="product", values="sales", aggfunc="sum", fill_value=0
        )

    def monthly_regional_trend(self) -> pd.DataFrame:
        return (
            self.df.groupby(["month", "region"])["sales"]
            .sum()
            .reset_index()
            .sort_values(["month", "sales"], ascending=[True, False])
        )