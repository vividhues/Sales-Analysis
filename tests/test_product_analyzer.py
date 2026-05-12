import pytest
import pandas as pd
from src.data_loader import load_sales_data
from src.analytics.product_analyzer import ProductAnalyzer


@pytest.fixture
def df():
    return load_sales_data()


@pytest.fixture
def analyzer(df):
    return ProductAnalyzer(df)


class TestProductAnalyzer:
    def test_sales_by_product_returns_all_products(self, analyzer):
        result = analyzer.sales_by_product()
        assert len(result) == 3  # Laptop, Monitor, Keyboard

    def test_best_product_is_laptop(self, analyzer):
        best = analyzer.best_performing_product()
        assert best["product"] == "Laptop"

    def test_total_sales_positive(self, analyzer):
        result = analyzer.sales_by_product()
        assert (result["total_sales"] > 0).all()

    def test_ranking_has_ranks(self, analyzer):
        ranking = analyzer.product_ranking()
        assert "sales_rank" in ranking.columns
        assert "profit_rank" in ranking.columns
        assert ranking["sales_rank"].tolist() == [1, 2, 3] or set(ranking["sales_rank"]) == {1, 2, 3}

    def test_monthly_product_sales_has_months(self, analyzer):
        monthly = analyzer.monthly_product_sales()
        assert len(monthly["month"].unique()) >= 1

    def test_growth_rate_calculation(self, analyzer):
        growth = analyzer.product_growth_rate()
        assert "growth_pct" in growth.columns