from pathlib import Path
import pandas as pd
import numpy as np

def load_csv_with_datetime_index(
    file_path,
    date_col="Year",
    date_format="%Y",
    index_name="Year",
    sort_index=True,
    verbose=True,
):
    """
    CSVを読み込み、指定カラムをDatetimeIndexに変換する

    Parameters
    ----------
    file_path : str or Path
        CSVファイルのパス
    date_col : str
        日付として扱うカラム名（例: "Year"）
    date_format : str
        日付フォーマット（例: "%Y"）
    index_name : str
        index名
    sort_index : bool
        indexを昇順に並べるか
    verbose : bool
        ログ出力

    Returns
    -------
    df : pandas.DataFrame
        DatetimeIndexを持つDataFrame
    """

    # --- Path化 ---
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # --- CSV読み込み ---
    df = pd.read_csv(file_path)

    if verbose:
        print("=== CSV loaded ===")
        print(f"shape: {df.shape}")
        print(f"columns: {list(df.columns)}")

    # --- 日付変換 ---
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found in CSV")

    df[date_col] = pd.to_datetime(df[date_col], format=date_format)

    # --- index設定 ---
    df = df.set_index(date_col)

    if sort_index:
        df = df.sort_index()

    df.index.name = index_name

    # --- validation ---
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index is not DatetimeIndex")

    if verbose:
        print("=== DatetimeIndex set ===")
        print(type(df.index))
        print(df.index[:5])

    return df

def save_init_params(init_params, base_dir="../data/init_para"):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1. スカラー系
    # =========================
    scalar_dict = {
        "lambda0_init": init_params.lambda0_init,
        "alpha_init": init_params.alpha_init,
        "theta_init": init_params.theta_init,
        "gamma_init": init_params.gamma_init,
        "lambda0_se": init_params.lambda0_se,
        "alpha_se": init_params.alpha_se,
        "theta_se": init_params.theta_se,
        "gamma_se": init_params.gamma_se,
        "lambda0_var": init_params.lambda0_var,
        "alpha_var": init_params.alpha_var,
        "theta_var": init_params.theta_var,
        "gamma_var": init_params.gamma_var,
    }

    df_scalar = pd.DataFrame([scalar_dict])
    df_scalar.to_csv(base / "init_scalar.csv", index=False)

    # =========================
    # 2. y_t（最重要）
    # =========================
    df_y = pd.DataFrame({
        "t": np.arange(len(init_params.y_init)),
        "y_init": init_params.y_init
    })
    df_y.to_csv(base / "y_init.csv", index=False)

    # =========================
    # 3. ACF
    # =========================
    df_acf = pd.DataFrame({
        "lag": init_params.acf_lags,
        "acf": init_params.acf_values
    })
    df_acf.to_csv(base / "acf.csv", index=False)

    print(f"[INFO] saved to {base.resolve()}")

def load_init_params(base_dir="../data/init_para"):
    base = Path(base_dir)

    df_scalar = pd.read_csv(base / "init_scalar.csv")
    df_y = pd.read_csv(base / "y_init.csv")
    df_acf = pd.read_csv(base / "acf.csv")

    return {
        "scalar": df_scalar.iloc[0].to_dict(),
        "y_init": df_y["y_init"].values,
        "acf_lags": df_acf["lag"].values,
        "acf_values": df_acf["acf"].values,
    }