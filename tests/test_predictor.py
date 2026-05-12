import pytest
import pandas as pd
from src.data_loader import load_sales_data
from src.ml.predictor import SalesPredictor


@pytest.fixture
def df():
    return load_sales_data()


class TestSalesPredictor:
    def test_train_returns_metrics(self, df):
        predictor = SalesPredictor()
        metrics = predictor.train(df)
        assert "MAE" in metrics
        assert "RMSE" in metrics
        assert "R2" in metrics

    def test_feature_importance_shape(self, df):
        predictor = SalesPredictor()
        predictor.train(df)
        importance = predictor.feature_importance()
        assert len(importance) == 14  # number of feature columns

    def test_predict_before_train_raises(self, df):
        predictor = SalesPredictor()
        with pytest.raises(RuntimeError):
            predictor.predict(df)

    def test_predict_best_product_returns_results(self, df):
        predictor = SalesPredictor()
        result = predictor.predict_best_product_next_month(df)
        assert "predicted_sales" in result.columns
        assert len(result) > 0

    def test_predictions_are_sorted_descending(self, df):
        predictor = SalesPredictor()
        result = predictor.predict_best_product_next_month(df)
        sales = result["predicted_sales"].tolist()
        assert sales == sorted(sales, reverse=True)