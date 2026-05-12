import pytest
import pandas as pd
from src.data_loader import load_sales_data, get_summary


class TestDataLoader:
    def test_load_returns_dataframe(self):
        df = load_sales_data()
        assert isinstance(df, pd.DataFrame)

    def test_correct_row_count(self):
        df = load_sales_data()
        assert len(df) == 24

    def test_required_columns_present(self):
        df = load_sales_data()
        for col in ["date", "region", "product", "channel", "sales", "profit", "units"]:
            assert col in df.columns

    def test_derived_columns(self):
        df = load_sales_data()
        assert "month" in df.columns
        assert "profit_margin" in df.columns
        assert "revenue_per_unit" in df.columns

    def test_profit_margin_range(self):
        df = load_sales_data()
        assert (df["profit_margin"] >= 0).all()
        assert (df["profit_margin"] <= 1).all()

    def test_get_summary_keys(self):
        df = load_sales_data()
        summary = get_summary(df)
        expected_keys = {"total_records", "date_range", "total_sales",
                         "total_profit", "avg_profit_margin", "products",
                         "regions", "channels"}
        assert set(summary.keys()) == expected_keys

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_sales_data("/nonexistent/path.csv")