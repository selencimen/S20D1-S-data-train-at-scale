import pandas as pd

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from taxifare.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """

    # 1️⃣ Drop NaNs
    df = df.dropna()

    # 2️⃣ Remove buggy transactions
    df = df[df["fare_amount"] > 0]
    df = df[df["passenger_count"] > 0]

    # 3️⃣ Remove geographically irrelevant rows (NYC bounding box)
    df = df[
        (df["pickup_longitude"].between(-74.3, -73.7)) &
        (df["pickup_latitude"].between(40.5, 40.9)) &
        (df["dropoff_longitude"].between(-74.3, -73.7)) &
        (df["dropoff_latitude"].between(40.5, 40.9))
    ]

    print("✅ data cleaned")

    return df
