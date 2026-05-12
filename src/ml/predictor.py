import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config.settings import Settings
from src.ml.feature_engineer import FeatureEngineer


class SalesPredictor:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.model = RandomForestRegressor(
            n_estimators=Settings.n_estimators,
            random_state=Settings.random_state,
            n_jobs=-1,
        )
        self.metrics: dict = {}
        self._trained = False

    def train(self, df: pd.DataFrame):
        X_train, X_test, y_train, y_test = self.fe.prepare_train_test(
            df, target=Settings.target_column,
            test_size=Settings.test_size,
            random_state=Settings.random_state,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.metrics = {
            "MAE": round(mean_absolute_error(y_test, y_pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            "R2": round(r2_score(y_test, y_pred), 4),
        }
        self._trained = True
        return self.metrics

    def predict(self, input_df: pd.DataFrame) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model is not trained yet. Call .train() first.")
        featured = self.fe.build_features(input_df)
        feature_cols = self.fe.get_feature_columns()
        X = featured[feature_cols].values
        if self.fe._is_fitted:
            X = self.fe.scaler.transform(X)
        return self.model.predict(X)

    def predict_best_product_next_month(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._trained:
            self.train(df)

        last_date = df["date"].max()
        next_month_date = last_date + pd.Timedelta(days=30)

        combos = df[["region", "product", "channel"]].drop_duplicates().copy()
        combos["date"] = next_month_date
        combos["sales"] = 0
        combos["profit"] = 0
        combos["units"] = 0
        combos["profit_margin"] = 0
        combos["revenue_per_unit"] = 0
        combos["month"] = next_month_date.to_period("M")

        latest = df.sort_values("date").groupby(["region", "product", "channel"]).last().reset_index()
        lag_cols = {"sales": "prev_sales", "profit": "prev_profit", "units": "prev_units"}
        for orig, lag_name in lag_cols.items():
            latest = latest.rename(columns={orig: lag_name})

        combos = combos.merge(
            latest[["region", "product", "channel", "prev_sales", "prev_profit", "prev_units"]],
            on=["region", "product", "channel"],
            how="left",
        )

        featured = self.fe.build_features(combos)

        for orig, lag_name in lag_cols.items():
            if lag_name in combos.columns:
                featured[lag_name] = combos[lag_name].values

        feature_cols = self.fe.get_feature_columns()
        X = featured[feature_cols].values
        if self.fe._is_fitted:
            X = self.fe.scaler.transform(X)

        combos["predicted_sales"] = self.model.predict(X)
        combos = combos.sort_values("predicted_sales", ascending=False).reset_index(drop=True)
        return combos[["region", "product", "channel", "predicted_sales"]]

    def feature_importance(self) -> pd.DataFrame:
        if not self._trained:
            raise RuntimeError("Model is not trained yet.")
        cols = self.fe.get_feature_columns()
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )