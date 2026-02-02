import math
import numpy as np
import pandas as pd
import pygeohash as gh

def transform_time_features(X: pd.DataFrame) -> np.ndarray:
    X = X.copy()

    X["hour"] = X["pickup_datetime"].dt.hour
    X["dayofweek"] = X["pickup_datetime"].dt.dayofweek
    X["month"] = X["pickup_datetime"].dt.month

    return X[["hour", "dayofweek", "month"]].values





def transform_lonlat_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    X["abs_lon_diff"] = (X["pickup_longitude"] - X["dropoff_longitude"]).abs()
    X["abs_lat_diff"] = (X["pickup_latitude"] - X["dropoff_latitude"]).abs()

    return X[["abs_lon_diff", "abs_lat_diff"]]


def compute_geohash(X: pd.DataFrame, precision: int = 3) -> pd.DataFrame:
    X = X.copy()

    pickup_geohash = X.apply(
        lambda row: gh.encode(
            row["pickup_latitude"],
            row["pickup_longitude"],
            precision
        ),
        axis=1
    )

    dropoff_geohash = X.apply(
        lambda row: gh.encode(
            row["dropoff_latitude"],
            row["dropoff_longitude"],
            precision
        ),
        axis=1
    )

    return pd.DataFrame({
    "pickup_geohash": pickup_geohash,
    "dropoff_geohash": dropoff_geohash
    })







