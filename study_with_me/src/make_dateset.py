import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy
import sklearn
import jpholiday

"""

"""
def make_price_df(df, area="エリアプライス東京(円/kWh)"):
    df_price = pd.DataFrame({
        "slot": df["時刻コード"],
        "price": df[area],
        "is_holiday": df.index.map(
            lambda x: jpholiday.is_holiday(x) or x.weekday() >= 5
        )
    })

    return df_price
    
"""
休日の情報を減らし、平日の情報に変換する関数
"""
def make_scaled_df(df, price_col="price", holiday_col="is_holiday"):
    df = df.copy()
    df["date"] = df.index.date
    # 
    df["price_weekday"] = np.where(~df[holiday_col], df[price_col], np.nan)
    df["price_holiday"] = np.where(df[holiday_col], df[price_col], np.nan)

    daily = df.groupby("date").agg({
        "price_weekday": "mean",
        "price_holiday": "mean"
    })

    daily["weekday_cummean"] = daily["price_weekday"].expanding().mean()
    daily["holiday_cummean"] = daily["price_holiday"].expanding().mean()

    daily["k"] = daily["weekday_cummean"] / daily["holiday_cummean"]

    df = df.merge(daily["k"], left_on="date", right_index=True, how="left")

    df["price_scaled"] = df[price_col]
    df.loc[df[holiday_col], "price_scaled"] = (
        df.loc[df[holiday_col], price_col] * df.loc[df[holiday_col], "k"]
    )

    # slotを保持したまま返す
    return df[["slot" , price_col, holiday_col, "price_scaled"]]

"""
# スパイク発生表現uと価格差xを追加する
"""
def add_spike_features(df, threshold, price_col="price_scaled", slot_col="slot", holiday_col="is_holiday"):
    df = df.copy()

    df["u"] = (df[price_col] > threshold).astype(int)

    df["x"] = np.where(
        df["u"] == 1,
        df[price_col] - threshold,
        0.0
    )

    return df[[slot_col, holiday_col, "price", "price_scaled", "u", "x"]]


"""
ラッパー関数
"""
def make_dateset(
    df , # もとになるデータセット
    area , # どこのエリアか
    threshold = 25 # スパイク閾値
):
    df_price_area = make_price_df(df,area)
    df_price_area_scaled = make_scaled_df(df_price_area)
    df_price_area_return = add_spike_features(df_price_area_scaled , threshold)

    return df_price_area_return
    