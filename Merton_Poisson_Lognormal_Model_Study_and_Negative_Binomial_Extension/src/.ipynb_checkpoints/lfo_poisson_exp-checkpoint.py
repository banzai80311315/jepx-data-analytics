import numpy as np
import pandas as pd
from scipy.special import gammaln
from .models_poisson_exp import sample_exp_kernel_model

def logmeanexp(a, axis=None):
    """
    数値安定な log(mean(exp(a)))
    """
    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.mean(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def poisson_logpmf(k, mu):
    """
    log Poisson(k | mu)
    """
    k = np.asarray(k)
    mu = np.asarray(mu, dtype=float)
    return k * np.log(mu) - mu - gammaln(k + 1)


def posterior_array(idata, var_name):
    """
    idata.posterior[var_name] を
    (chain, draw, ...) -> (sample, ...)
    に変換
    """
    arr = idata.posterior[var_name].values
    return arr.reshape(-1, *arr.shape[2:])

# 1時点ぶんの LFO 寄与を計算する関数
def log_pred_exp_poisson(
    trace,
    x_next,
    n_y_future=200,
    random_seed=42,
):
    """
    1時点先 x_{t+1} の log p(x_{t+1} | x_{1:t}) を
    posterior samples と y_{t+1} のモンテカルロ平均で近似する

    Parameters
    ----------
    trace : arviz.InferenceData
        sample_exp_kernel_model の戻り値
    x_next : int
        観測された次時点のカウント
    n_y_future : int
        各 posterior sample について y_{t+1} を何回サンプルするか
    random_seed : int

    Returns
    -------
    float
        log predictive density
    """
    rng = np.random.default_rng(random_seed)

    lambda0_s = posterior_array(trace, "lambda0")   # shape: (S,)
    alpha_s   = posterior_array(trace, "alpha")     # shape: (S,)
    theta_s   = posterior_array(trace, "theta")     # shape: (S,)
    y_s       = posterior_array(trace, "y")         # shape: (S, T_train)

    y_last_s = y_s[:, -1]                           # 学習区間の最後の y_t
    S = len(lambda0_s)

    # 条件付き分布: y_{t+1} | y_t, theta ~ N(theta y_t, 1-theta^2)
    mean_future = theta_s * y_last_s
    var_future = np.maximum(1.0 - theta_s**2, 1e-12)
    sd_future = np.sqrt(var_future)

    # shape: (S, R)
    y_future_sr = rng.normal(
        loc=mean_future[:, None],
        scale=sd_future[:, None],
        size=(S, n_y_future),
    )

    # mu_{t+1}^{(s,r)} = lambda0^{(s)} * exp(alpha^{(s)} y_{t+1}^{(s,r)})
    mu_sr = lambda0_s[:, None] * np.exp(alpha_s[:, None] * y_future_sr)

    # log Poisson(x_next | mu_sr)
    logp_sr = poisson_logpmf(x_next, mu_sr)

    # log [ (1/S) sum_s (1/R) sum_r p(...) ]
    return float(logmeanexp(logp_sr.ravel()))

# LFO全体を回す関数
def compute_lfo_exp(
    data,
    lambda0_init,
    alpha_init,
    theta_init,
    lambda0_se,
    alpha_se,
    theta_se,
    y_init,
    prior_scale=1.0,
    t0_grid=None,
    draws=1000,
    tune=1000,
    chains=2,
    target_accept=0.9,
    jitter=1e-6,
    random_seed=42,
    verbose=True,
):
    """
    指数減衰モデルの LFO を計算する

    Parameters
    ----------
    data : array-like
        全時系列データ
    t0_grid : list[int] or None
        学習終端 t のリスト
        各 t に対して x[:t+1] で学習し、x[t+1] を予測する
        None のときは全部回す

    Returns
    -------
    lfo_total : float
    df_lfo : pd.DataFrame
    """
    x = np.asarray(data).astype(int)
    T = len(x)

    if t0_grid is None:
        t0_grid = list(range(10, T - 1))

    rows = []

    for i, t in enumerate(t0_grid):
        x_train = x[:t+1]
        x_next = x[t+1]

        # y_init も学習区間に切る
        y_init_train = np.asarray(y_init)[:t+1]

        trace = sample_exp_kernel_model(
            data=x_train,
            lambda0_init=lambda0_init,
            alpha_init=alpha_init,
            theta_init=theta_init,
            lambda0_se=lambda0_se,
            alpha_se=alpha_se,
            theta_se=theta_se,
            y_init=y_init_train,
            prior_scale=prior_scale,
            jitter=jitter,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed + i,
            return_model=False,
        )

        logp = log_pred_exp_poisson(
            trace=trace,
            x_next=x_next,
            n_y_future=200,
            random_seed=random_seed + 10000 + i,
        )

        rows.append({
            "t_train_end": t,
            "x_next": x_next,
            "log_pred": logp,
        })

        if verbose:
            print(f"[{i+1}/{len(t0_grid)}] t={t}, x_next={x_next}, log_pred={logp:.4f}")

    df_lfo = pd.DataFrame(rows)
    lfo_total = df_lfo["log_pred"].sum()

    return float(lfo_total), df_lfo