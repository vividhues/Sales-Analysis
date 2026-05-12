import pandas as pd

class ChannelAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def sales_by_channel(self) -> pd.DataFrame:
        return (
            self.df.groupby("channel")
            .agg(total_sales=("sales", "sum"),
                 total_profit=("profit", "sum"),
                 total_units=("units", "sum"),
                 avg_profit_margin=("profit_margin", "mean"),
                 avg_revenue_per_unit=("revenue_per_unit", "mean"),
                 transaction_count=("sales", "count"))
            .sort_values("total_sales", ascending=False)
            .reset_index()
        )

    def best_channel_per_product(self) -> pd.DataFrame:
        return (
            self.df.groupby(["product", "channel"])["sales"]
            .sum()
            .reset_index()
            .sort_values(["product", "sales"], ascending=[True, False])
            .groupby("product")
            .first()
            .reset_index()
            .rename(columns={"channel": "best_channel", "sales": "channel_sales"})
        )

    def channel_efficiency(self) -> pd.DataFrame:
        """Sales per transaction for each channel."""
        summary = self.sales_by_channel()
        summary["sales_per_transaction"] = (
            summary["total_sales"] / summary["transaction_count"]
        ).round(2)
        return summary

    def channel_product_breakdown(self) -> pd.DataFrame:
        return self.df.pivot_table(
            index="channel", columns="product", values="sales", aggfunc="sum", fill_value=0
        )