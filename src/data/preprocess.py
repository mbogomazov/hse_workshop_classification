import pandas as pd
import numpy as np
from src.config import *
import datetime as dt


def drop_unnecesary_id(df: pd.DataFrame) -> pd.DataFrame:
    if 'ID_y' in df.columns:
        df = df.drop('ID_y', axis=1)
    return df


def fill_sex(df: pd.DataFrame) -> pd.DataFrame:
    most_freq = df[SEX_COL].value_counts().index[0]
    df[SEX_COL] = df[SEX_COL].fillna(most_freq)
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df[CAT_COLS] = df[CAT_COLS].astype('category')

    ohe_int_cols = df[OHE_COLS].select_dtypes('number').columns
    df[ohe_int_cols] = df[ohe_int_cols].astype(np.int8)

    df[REAL_COLS] = df[REAL_COLS].astype(np.float32)
    return df


def fill_sleep_time(df: pd.DataFrame) -> pd.DataFrame:
    most_freq = df[WAKE_UP_TIME_COL].value_counts().index[0]
    df[WAKE_UP_TIME_COL] = df[WAKE_UP_TIME_COL].fillna(most_freq)

    most_freq = df[FALL_ASLEEP_COL].value_counts().index[0]
    df[FALL_ASLEEP_COL] = df[FALL_ASLEEP_COL].fillna(most_freq)
    return df


def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    df = df.set_index(idx_col)
    return df


def fill_freq(df: pd.DataFrame) -> pd.DataFrame:
    df[FREQ_COL] = df[FREQ_COL].fillna('0')
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, ID_COL)
    df = drop_unnecesary_id(df)
    df = fill_sex(df)
    df = fill_freq(df)
    df = fill_sleep_time(df)
    df = df.fillna(0)
    df = cast_types(df)

    return df
