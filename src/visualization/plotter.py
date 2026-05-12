import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import Settings


class Plotter:
    def __init__(self):
        Settings.report_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette=Settings.color_palette)
        self.dpi = Settings.figure_dpi

    def _save(self, fig, name: str):
        path = Settings.report_dir / name
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")

    def plot_product_sales(self, product_summary: pd.DataFrame) -> plt.Figure:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        sns.barplot(data=product_summary, x="product", y="total_sales", ax=axes[0])
        axes[0].set_title("Total Sales by Product")

        sns.barplot(data=product_summary, x="product", y="total_profit", ax=axes[1])
        axes[1].set_title("Total Profit by Product")

        sns.barplot(data=product_summary, x="product", y="total_units", ax=axes[2])
        axes[2].set_title("Total Units by Product")

        fig.suptitle("Product Performance Overview", fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "product_performance.png")
        return fig

    def plot_regional_sales(self, regional_summary: pd.DataFrame) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sns.barplot(data=regional_summary, x="region", y="total_sales", ax=axes[0])
        axes[0].set_title("Total Sales by Region")

        sns.barplot(data=regional_summary, x="region", y="avg_profit_margin", ax=axes[1])
        axes[1].set_title("Avg Profit Margin by Region")

        fig.suptitle("Regional Performance", fontsize=14, fontweight="bold")
        fig.tight_layout()
        self._save(fig, "regional_performance.png")
        return fig

    def plot_channel_sales(self, channel_summary: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=channel_summary, x="channel", y="total_sales", ax=ax)
        ax.set_title("Total Sales by Channel")
        fig.tight_layout()
        self._save(fig, "channel_performance.png")
        return fig

    def plot_monthly_trend(self, monthly_df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 5))
        monthly_df["month_str"] = monthly_df["month"].astype(str)
        sns.lineplot(data=monthly_df, x="month_str", y="sales", hue="product",
                     marker="o", ax=ax)
        ax.set_title("Monthly Sales Trend by Product")
        ax.set_xlabel("Month")
        ax.set_ylabel("Sales")
        fig.tight_layout()
        self._save(fig, "monthly_trend.png")
        return fig

    def plot_predictions(self, pred_df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        pred_sorted = pred_df.sort_values("predicted_sales", ascending=True)
        colors = sns.color_palette(Settings.color_palette, n_colors=len(pred_sorted))
        bars = ax.barh(
            pred_sorted["product"] + " | " + pred_sorted["region"] + " | " + pred_sorted["channel"],
            pred_sorted["predicted_sales"],
            color=colors,
        )
        ax.set_title("Predicted Sales for next month Ranking", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Sales")
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 100, bar.get_y() + bar.get_height() / 2,
                    f"${width:,.0f}", va="center", fontsize=9)
        fig.tight_layout()
        self._save(fig, "predicted_sales_ranking.png")
        return fig

    def plot_feature_importance(self, importance_df: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x="importance", y="feature", ax=ax)
        ax.set_title("Feature Importance (Random Forest)")
        fig.tight_layout()
        self._save(fig, "feature_importance.png")
        return fig