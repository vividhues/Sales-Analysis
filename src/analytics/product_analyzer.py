import pandas as pd

class ProductAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def sales_by_product(self) -> pd.DataFrame:
        return (
            self.df.groupby("product")
            .agg(total_sales=("sales", "sum"),
                 total_profit=("profit", "sum"),
                 total_units=("units", "sum"),
                 avg_profit_margin=("profit_margin", "mean"),
                 avg_revenue_per_unit=("revenue_per_unit", "mean"))
            .sort_values("total_sales", ascending=False)
            .reset_index()
        )

    def monthly_product_sales(self) -> pd.DataFrame:
        return (
            self.df.groupby(["month", "product"])["sales"]
            .sum()
            .reset_index()
            .sort_values(["month", "sales"], ascending=[True, False])
        )

    def best_performing_product(self) -> dict:
        summary = self.sales_by_product()
        top = summary.iloc[0]
        return {
            "product": top["product"],
            "total_sales": int(top["total_sales"]),
            "total_profit": int(top["total_profit"]),
            "total_units": int(top["total_units"]),
            "avg_profit_margin": round(top["avg_profit_margin"], 4),
        }

    def product_ranking(self) -> pd.DataFrame:
        summary = self.sales_by_product()
        summary["sales_rank"] = summary["total_sales"].rank(ascending=False).astype(int)
        summary["profit_rank"] = summary["total_profit"].rank(ascending=False).astype(int)
        return summary

    def product_growth_rate(self) -> pd.DataFrame:
        monthly = self.monthly_product_sales()
        monthly = monthly.sort_values(["product", "month"])
        monthly["growth_pct"] = (
            monthly.groupby("product")["sales"]
            .pct_change()
            .round(4) * 100
        )
        return monthly