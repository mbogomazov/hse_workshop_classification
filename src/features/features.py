import pandas as pd
import datetime as dt
from src.config import *


def add_sleeping_mode_features(df):
    for ind, row in df.iterrows():
        fall_asleep_time = dt.datetime.strptime(
            row[FALL_ASLEEP_COL], '%H:%M:%S')
        wake_up_time = dt.datetime.strptime(
            row[WAKE_UP_TIME_COL], '%H:%M:%S')
        if fall_asleep_time.hour > 12:
            wake_up_time += dt.timedelta(days=1)

        df.at[ind, SLEEP_DURATION_COL] = round(
            (wake_up_time - fall_asleep_time).total_seconds() / 3600, 3)

        if fall_asleep_time.hour > 12 or wake_up_time.hour < 8:
            df.at[ind, SLEEP_BEHAVIORS_COL] = '0'
        elif fall_asleep_time.hour < 12 and wake_up_time.hour > 8:
            df.at[ind, SLEEP_BEHAVIORS_COL] = '1'
        else:
            df.at[ind, SLEEP_BEHAVIORS_COL] = '2'

    return df


def remove_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.drop([WAKE_UP_TIME_COL, FALL_ASLEEP_COL], axis=1, inplace=True)
    return df


def gen_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_sleeping_mode_features(df)
    df = remove_unnecessary_columns(df)

    return df
