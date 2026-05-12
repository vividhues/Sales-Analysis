import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureEngineer:
    def __init__(self):
        self.encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self._is_fitted = False

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df.copy()

        X["year"] = X["date"].dt.year
        X["month_num"] = X["date"].dt.month
        X["day_of_week"] = X["date"].dt.dayofweek
        X["day_of_month"] = X["date"].dt.day
        X["is_weekend"] = (X["day_of_week"] >= 5).astype(int)
        X = X.sort_values(["product", "date"])
        X["prev_sales"] = X.groupby("product")["sales"].shift(1)
        X["prev_profit"] = X.groupby("product")["profit"].shift(1)
        X["prev_units"] = X.groupby("product")["units"].shift(1)
        X["sales_rolling_3"] = (X.groupby("product")["sales"].transform(lambda s: s.rolling(3, min_periods=1).mean()))

        cat_cols = ["region", "product", "channel"]
        for col in cat_cols:
            le = LabelEncoder()
            X[f"{col}_enc"] = le.fit_transform(X[col])
            self.encoders[col] = le

        X["region_x_product"] = X["region_enc"] * X["product_enc"]
        X["channel_x_product"] = X["channel_enc"] * X["product_enc"]

        X.bfill(inplace=True)

        return X

    def get_feature_columns(self) -> list[str]:
        return [
            "region_enc", "product_enc", "channel_enc",
            "year", "month_num", "day_of_week", "day_of_month", "is_weekend",
            "prev_sales", "prev_profit", "prev_units", "sales_rolling_3",
            "region_x_product", "channel_x_product",
        ]

    def prepare_train_test(self, df: pd.DataFrame, target: str = "sales",
                           test_size: float = 0.2, random_state: int = 42):

        from sklearn.model_selection import train_test_split

        featured = self.build_features(df)
        feature_cols = self.get_feature_columns()

        X = featured[feature_cols].values
        y = featured[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        self._is_fitted = True

        return X_train, X_test, y_train, y_test