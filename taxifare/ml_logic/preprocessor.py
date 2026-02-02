import numpy as np
import pandas as pd

from colorama import Fore, Style

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from taxifare.ml_logic.encoders import (
    transform_time_features,
    transform_lonlat_features,
    compute_geohash
)

def create_sklearn_preprocessor() -> ColumnTransformer:

    time_pipe = make_pipeline(
        FunctionTransformer(transform_time_features, validate=False),
        OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse=False
        )
    )


    lonlat_pipe = make_pipeline(
        FunctionTransformer(transform_lonlat_features, validate=False)
    )

    geohash_pipe = make_pipeline(
        FunctionTransformer(compute_geohash, validate=False),
        OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse=False
        )   
    )


    final_preprocessor = ColumnTransformer(
        transformers=[
            ("time", time_pipe, ["pickup_datetime"]),
            ("lonlat", lonlat_pipe, [
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
            ]),
            ("geohash", geohash_pipe, [
                "pickup_longitude",
                "pickup_latitude",
                "dropoff_longitude",
                "dropoff_latitude",
            ]),
        ],
        remainder="drop"
    )


    return final_preprocessor


def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    print(Fore.BLUE + "\nPreprocessing features..." + Style.RESET_ALL)

    preprocessor = create_sklearn_preprocessor()
    X_processed = preprocessor.fit_transform(X)

    print("✅ X_processed, with shape:", X_processed.shape)
    return X_processed

