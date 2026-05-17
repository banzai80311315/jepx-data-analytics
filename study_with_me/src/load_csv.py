"""
start_yearからend_yearまでの年度のデータを一括読み込み
indexをtimeindexに変更
"""
import pathlib 
import pandas as pd
def load_jepx_data(start_year, end_year, base_path, encoding="cp932"):
    base_path = pathlib.Path(base_path)
    dfs = []
    # 年度ごとのCSVファイルを読み込み，リストとして保持する
    for year in range(start_year, end_year + 1):
        csv_path = base_path/ f"spot_summary_{year}.csv"
        df_year = pd.read_csv(csv_path, encoding=encoding)
        dfs.append(df_year)
    # 複数年度のデータを結合し，一つのデータフレームを生成する
    df = pd.concat(dfs, ignore_index=True)
    # indexを加工、時系列データを扱うのでdatetimeindexへ
    df["受渡日"] = pd.to_datetime(df["受渡日"])
    df["datetime"] = (
        df["受渡日"]
        + pd.to_timedelta((df["時刻コード"] - 1) * 30, unit="min")
    )
    df = df.set_index("datetime").sort_index()
    return df