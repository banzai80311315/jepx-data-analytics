import numpy as np
import pandas as pd
from scipy.special import gammaln
from .models_poisson_power import sample_power_kernel_model


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
    mu = np.maximum(mu, 1e-300)  # 数値安定
    return k * np.log(mu) - mu - gammaln(k + 1)


def posterior_array(idata, var_name):
    """
    idata.posterior[var_name] を
    (chain, draw, ...) -> (sample, ...)
    に変換
    """
    arr = idata.posterior[var_name].values
    return arr.reshape(-1, *arr.shape[2:])


def _power_cov_from_gamma(T, gamma, jitter=1e-6):
    """
    学習区間 0,...,T-1 に対する power-law kernel 共分散行列
    K_ij = 1 / (1 + |i-j|)^gamma
    """
    idx = np.arange(T)
    dist = np.abs(idx[:, None] - idx[None, :]).astype(float)
    K = (1.0 + dist) ** (-gamma)
    K = K + jitter * np.eye(T)
    return K


def _power_cross_cov_next(T, gamma):
    """
    学習区間 0,...,T-1 と次時点 T の cross covariance
    k_*[i] = Cov(y_i, y_T) = 1 / (1 + |T-i|)^gamma
    """
    idx = np.arange(T)
    dist = np.abs(T - idx).astype(float)
    k_star = (1.0 + dist) ** (-gamma)
    return k_star


def _sample_y_next_power(y_hist, gamma, rng, n_y_future=200, jitter=1e-6):
    """
    power-law kernel GP における
    y_{t+1} | y_{1:t}, gamma
    の条件付き正規分布からサンプルする

    Parameters
    ----------
    y_hist : array-like, shape (T,)
        学習区間の latent y
    gamma : float
        power-law kernel parameter
    rng : np.random.Generator
    n_y_future : int
    jitter : float

    Returns
    -------
    y_future_r : ndarray, shape (n_y_future,)
    """
    y_hist = np.asarray(y_hist, dtype=float)
    T = len(y_hist)

    K = _power_cov_from_gamma(T=T, gamma=gamma, jitter=jitter)
    k_star = _power_cross_cov_next(T=T, gamma=gamma)

    # mean = k_*^T K^{-1} y
    alpha_vec = np.linalg.solve(K, y_hist)
    mean_future = float(k_star @ alpha_vec)

    # var = k_** - k_*^T K^{-1} k_*
    # power kernel の自己共分散は lag=0 なので 1
    v = np.linalg.solve(K, k_star)
    var_future = float(1.0 - k_star @ v)
    var_future = max(var_future, 1e-12)

    return rng.normal(
        loc=mean_future,
        scale=np.sqrt(var_future),
        size=n_y_future,
    )


def log_pred_pow_poisson(
    trace,
    x_next,
    n_y_future=200,
    jitter=1e-6,
    random_seed=42,
):
    """
    1時点先 x_{t+1} の log p(x_{t+1} | x_{1:t}) を
    posterior samples と y_{t+1} のモンテカルロ平均で近似する

    Parameters
    ----------
    trace : arviz.InferenceData
        sample_power_kernel_model の戻り値
    x_next : int
        観測された次時点のカウント
    n_y_future : int
        各 posterior sample について y_{t+1} を何回サンプルするか
    jitter : float
    random_seed : int

    Returns
    -------
    float
        log predictive density
    """
    rng = np.random.default_rng(random_seed)

    lambda0_s = posterior_array(trace, "lambda0")   # shape: (S,)
    alpha_s   = posterior_array(trace, "alpha")     # shape: (S,)
    gamma_s   = posterior_array(trace, "gamma")     # shape: (S,)
    y_s       = posterior_array(trace, "y")         # shape: (S, T_train)

    S = len(lambda0_s)
    logp_s = np.empty(S, dtype=float)

    for s in range(S):
        y_future_r = _sample_y_next_power(
            y_hist=y_s[s],
            gamma=float(gamma_s[s]),
            rng=rng,
            n_y_future=n_y_future,
            jitter=jitter,
        )

        mu_r = float(lambda0_s[s]) * np.exp(float(alpha_s[s]) * y_future_r)
        logp_r = poisson_logpmf(x_next, mu_r)

        # posterior sample s の中で y_future を平均化
        logp_s[s] = logmeanexp(logp_r)

    # 最後に posterior sample 平均
    return float(logmeanexp(logp_s))


def compute_lfo_power(
    data,
    lambda0_init,
    alpha_init,
    gamma_init,
    lambda0_se,
    alpha_se,
    gamma_se,
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
    べき乗カーネルモデルの LFO を計算する

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

        trace = sample_power_kernel_model(
            data=x_train,
            lambda0_init=lambda0_init,
            alpha_init=alpha_init,
            gamma_init=gamma_init,
            lambda0_se=lambda0_se,
            alpha_se=alpha_se,
            gamma_se=gamma_se,
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

        logp = log_pred_pow_poisson(
            trace=trace,
            x_next=x_next,
            n_y_future=200,
            jitter=jitter,
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