"""
Utilities for obtaining and loading policy documents.
"""

from pathlib import Path
from typing import Literal

import pandas as pd


def extract_raw_text_from_pdf(url: str) -> str:
    raise NotImplementedError


def load_dataset_from_json(
        filepath: Path,
) -> pd.DataFrame:
    df = pd.read_json(filepath)
    return df


def load_dataset_from_pyspark(
        table_name: str,
) -> pd.DataFrame:
    from pyspark.sql.context import SparkSession
    spark = SparkSession.getActiveSession()
    sql_cmd = spark.sql(f"SELECT * FROM {table_name}")
    df = sql_cmd.toPandas()
    return df
