from __future__ import annotations

import pandas as pd


def normalize_default_counts(
    df: pd.DataFrame,
    default_cols: dict[str, str] | None = None,
    obligor_cols: dict[str, str] | None = None,
    scale: int = 3000,
    suffix: str = "_norm",
    inplace: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    デフォルト件数を共通母数に正規化する。

    例:
        SG_norm = D_SG / SG * 3000

    Parameters
    ----------
    df : pd.DataFrame
        元データ
    default_cols : dict[str, str], optional
        各グループのデフォルト件数列
        例: {"SG": "D_SG", "IG": "D_IG", "ALL": "D_ALL"}
    obligor_cols : dict[str, str], optional
        各グループの企業数列
        例: {"SG": "SG", "IG": "IG", "ALL": "ALL"}
    scale : int, default 3000
        正規化後の基準企業数
    suffix : str, default "_norm"
        生成する列名の接尾辞
    inplace : bool, default False
        True の場合は元のdfを直接更新する
    verbose : bool, default True
        ログ出力を行うか

    Returns
    -------
    pd.DataFrame
        正規化列を追加したDataFrame
    """

    if default_cols is None:
        default_cols = {
            "SG": "D_SG",
            "IG": "D_IG",
            "ALL": "D_ALL",
        }

    if obligor_cols is None:
        obligor_cols = {
            "SG": "SG",
            "IG": "IG",
            "ALL": "ALL",
        }

    work_df = df if inplace else df.copy()

    for group in default_cols:
        if group not in obligor_cols:
            raise ValueError(f"Group '{group}' is missing in obligor_cols")

        d_col = default_cols[group]
        n_col = obligor_cols[group]
        out_col = f"{group}{suffix}"

        if d_col not in work_df.columns:
            raise ValueError(f"Default count column not found: {d_col}")
        if n_col not in work_df.columns:
            raise ValueError(f"Obligor count column not found: {n_col}")

        if (work_df[n_col] <= 0).any():
            bad_rows = work_df.index[work_df[n_col] <= 0].tolist()
            raise ValueError(
                f"Column '{n_col}' contains non-positive values at rows: {bad_rows}"
            )

        work_df[out_col] = work_df[d_col] / work_df[n_col] * scale

        if verbose:
            print(
                f"[normalize_default_counts] created '{out_col}' "
                f"from '{d_col}' / '{n_col}' * {scale}"
            )

    return work_df