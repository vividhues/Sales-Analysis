import sys
from pathlib import Path

# fix for src not being available
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
from src.data_loader import load_sales_data, get_summary
from src.analytics.product_analyzer import ProductAnalyzer
from src.analytics.regional_analyzer import RegionalAnalyzer
from src.analytics.channel_analyzer import ChannelAnalyzer
from src.ml.predictor import SalesPredictor
from src.visualization.plotter import Plotter

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)


def print(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def main():
    print("\n Load Data \n")
    df = load_sales_data()
    summary = get_summary(df)

    print(f"Records loaded : {summary['total_records']}")
    print(f"Date range     : {summary['date_range']}")
    print(f"Total Sales    : ${summary['total_sales']:,}")
    print(f"Total Profit   : ${summary['total_profit']:,}")
    print(f"Avg Margin     : {summary['avg_profit_margin']:.2%}")
    print(f"Products       : {', '.join(summary['products'])}")
    print(f"Regions        : {', '.join(summary['regions'])}")
    print(f"Channels       : {', '.join(summary['channels'])}")


    print("\n Product Analytics \n")
    pa = ProductAnalyzer(df)
    product_summary = pa.sales_by_product()
    print(product_summary.to_string(index=False))

    best = pa.best_performing_product()
    print(f"\nBest Product: {best['product']}")
    print(f"Sales  : ${best['total_sales']:,}")
    print(f"Profit : ${best['total_profit']:,}")
    print(f"Units  : {best['total_units']:,}")
    print(f"Margin : {best['avg_profit_margin']:.2%}")

    ranking = pa.product_ranking()
    print("\n Product Ranking:")
    print(ranking[["product", "sales_rank", "profit_rank"]].to_string(index=False))

    growth = pa.product_growth_rate()
    print("\n Product Growth Rates:")
    print(growth.to_string(index=False))

    print("\n Regional Analysis \n")
    ra = RegionalAnalyzer(df)
    regional_summary = ra.sales_by_region()
    print(regional_summary.to_string(index=False))

    best_region = ra.best_region_per_product()
    print("\n Best Region per Product:")
    print(best_region.to_string(index=False))

    print("\n Channel Analytics \n")
    ca = ChannelAnalyzer(df)
    channel_summary = ca.sales_by_channel()
    print(channel_summary.to_string(index=False))

    best_channel = ca.best_channel_per_product()
    print("\n Best Channel per Product:")
    print(best_channel.to_string(index=False))

    efficiency = ca.channel_efficiency()
    print("\n Channel Efficiency:")
    print(efficiency.to_string(index=False))

    print("\n Predictive Analytics \n")
    predictor = SalesPredictor()
    metrics = predictor.train(df)
    print(f" Model Metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    importance = predictor.feature_importance()
    print("\n Feature Importance:")
    print(importance.to_string(index=False))

    predictions = predictor.predict_best_product_next_month(df)
    print("\n Predicted Best-Selling Combos Next Month:")
    print(predictions.to_string(index=False))

    top = predictions.iloc[0]
    print(f"\n Predicted top product: {top['product']} from {top['channel']} in {top['region']}")
    print(f" Expected Sales: ${top['predicted_sales']:,.0f}")

    print("\n Generating Charts")
    plotter = Plotter()

    plotter.plot_product_sales(product_summary)
    plotter.plot_regional_sales(regional_summary)
    plotter.plot_channel_sales(channel_summary)

    monthly = pa.monthly_product_sales()
    plotter.plot_monthly_trend(monthly)
    plotter.plot_predictions(predictions)
    plotter.plot_feature_importance(importance)

    print(f"Best Product: {best['product']} (${best['total_sales']:,} sales)")
    print(f"Best Region: {regional_summary.iloc[0]['region']} (${int(regional_summary.iloc[0]['total_sales']):,} sales)")
    print(f"Best Channel: {channel_summary.iloc[0]['channel']} (${int(channel_summary.iloc[0]['total_sales']):,} sales)")
    print(f"Predicted #1 Product: {top['product']} | {top['region']} | {top['channel']}")
    print(f"Model R² Score: {metrics['R2']}")
    print(f"\n  Charts saved to reports folder")
    print()


if __name__ == "__main__":
    main()