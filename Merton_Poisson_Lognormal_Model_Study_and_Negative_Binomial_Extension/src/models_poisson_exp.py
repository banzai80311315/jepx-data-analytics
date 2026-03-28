import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt


def _to_1d_count_array(data):
    """
    入力データを 1 次元の非負整数カウント配列に変換する。
    pandas.Series / list / np.ndarray を想定。
    """
    x = np.asarray(data, dtype=float).reshape(-1)

    if x.ndim != 1:
        raise ValueError("data は 1 次元配列である必要があります。")

    if np.any(~np.isfinite(x)):
        raise ValueError("data に NaN または inf が含まれています。")

    if np.any(x < 0):
        raise ValueError("data には非負の値のみを入れてください。")

    return np.rint(x).astype("int64")


def _make_distance_matrix(T):
    """
    |t_i - t_j| の距離行列を作る。
    """
    t = np.arange(T, dtype=float)
    return np.abs(t[:, None] - t[None, :])


def _theta_to_l(theta):
    """
    theta = exp(-1 / l) から l = -1 / log(theta) を返す。
    """
    theta = float(theta)

    if not (0.0 < theta < 1.0):
        raise ValueError("theta_init は 0 < theta_init < 1 を満たす必要があります。")

    return -1.0 / np.log(theta)


def _delta_method_sigma_l(theta_init, theta_se, eps=1e-12):
    """
    l = -1 / log(theta) に対するデルタ法での標準偏差近似を返す。

    dl/dtheta = 1 / [theta * (log theta)^2]

    よって
    Var(l) ≈ (dl/dtheta)^2 Var(theta)
           = theta_se^2 / [theta^2 (log theta)^4]

    したがって
    sigma_l = sqrt(Var(l))
    """
    theta_init = float(theta_init)
    theta_se = float(theta_se)

    if not (0.0 < theta_init < 1.0):
        raise ValueError("theta_init は 0 < theta_init < 1 を満たす必要があります。")

    if theta_se <= 0:
        raise ValueError("theta_se は正である必要があります。")

    log_theta = np.log(theta_init)

    if abs(log_theta) < eps:
        raise ValueError("theta_init が 1 に近すぎて sigma_l が不安定です。")

    var_l = (theta_se ** 2) / (theta_init ** 2 * (log_theta ** 4))
    sigma_l = np.sqrt(var_l)

    return sigma_l


def build_exp_kernel_model(
    data,
    lambda0_init,
    alpha_init,
    theta_init,
    lambda0_se,
    alpha_se,
    theta_se,
    y_init,
    prior_scale=1.0,
    jitter=1e-6,
):
    """
    Exponential-kernel Poisson latent GP model

    Model
    -----
    X_t | eta0, alpha, y ~ Poisson(exp(eta0 + alpha y_t))
    y ~ N(0, K_exp)
    K_exp[i,j] = exp(-|t_i - t_j| / l)

    Parameter mapping
    -----------------
    lambda0 = exp(eta0)
    theta   = exp(-1 / l)
    l       = -1 / log(theta)

    Priors
    ------
    eta0  ~ Normal(log(lambda0_init), prior_scale * sigma_eta0)
    alpha ~ TruncatedNormal(alpha_init, prior_scale * alpha_se, lower=0)
    l     ~ LogNormal(log(l_init), prior_scale * sigma_log_l)

    Notes
    -----
    - eta0 の事前標準偏差はデルタ法で sigma_eta0 ≈ lambda0_se / lambda0_init
    - l の事前標準偏差は theta からのデルタ法で近似
    """
    x = _to_1d_count_array(data)
    T = len(x)

    y_init = np.asarray(y_init, dtype=float).reshape(-1)
    if len(y_init) != T:
        raise ValueError("y_init の長さが data と一致していません。")

    if lambda0_init <= 0:
        raise ValueError("lambda0_init は正である必要があります。")

    if lambda0_se <= 0:
        raise ValueError("lambda0_se は正である必要があります。")

    if alpha_se <= 0:
        raise ValueError("alpha_se は正である必要があります。")

    if theta_se <= 0:
        raise ValueError("theta_se は正である必要があります。")

    eta0_init = np.log(float(lambda0_init))
    sigma_eta0 = float(lambda0_se) / float(lambda0_init)

    l_init = _theta_to_l(theta_init)
    sigma_l = _delta_method_sigma_l(theta_init, theta_se)

    # LogNormal の underlying normal 用に変換
    # l ~ LogNormal(mu_log_l, sigma_log_l)
    # 近似的に mu_log_l = log(l_init), sigma_log_l ≈ sigma_l / l_init
    mu_log_l = np.log(l_init)
    sigma_log_l = sigma_l / l_init

    dist = _make_distance_matrix(T)
    coords = {"time": np.arange(T)}

    with pm.Model(coords=coords) as model:
        x_obs = pm.Data("x_obs", x, dims="time")

        eta0 = pm.Normal(
            "eta0",
            mu=eta0_init,
            sigma=prior_scale * sigma_eta0,
        )

        alpha = pm.TruncatedNormal(
            "alpha",
            mu=float(alpha_init),
            sigma=prior_scale * float(alpha_se),
            lower=0.0,
        )

        log_l = pm.Normal(
            "log_l",
            mu=mu_log_l,
            sigma=prior_scale * sigma_log_l,
        )

        l = pm.Deterministic("l", pt.exp(log_l))
        theta = pm.Deterministic("theta", pt.exp(-1.0 / l))
        lambda0 = pm.Deterministic("lambda0", pt.exp(eta0))

        K = pt.exp(-dist / l) + jitter * pt.eye(T)

        y = pm.MvNormal(
            "y",
            mu=pt.zeros(T),
            cov=K,
            dims="time",
        )

        mu = pm.Deterministic(
            "mu",
            pt.exp(eta0 + alpha * y),
            dims="time",
        )

        pm.Poisson(
            "x",
            mu=mu,
            observed=x_obs,
            dims="time",
        )

    return model


def sample_exp_kernel_model(
    data,
    lambda0_init,
    alpha_init,
    theta_init,
    lambda0_se,
    alpha_se,
    theta_se,
    y_init,
    prior_scale=1.0,
    jitter=1e-6,
    draws=1000,
    tune=1000,
    chains=2,
    target_accept=0.9,
    random_seed=42,
    return_model=False,
):
    """
    build_exp_kernel_model を作成して MCMC まで回す。
    """
    model = build_exp_kernel_model(
        data=data,
        lambda0_init=lambda0_init,
        alpha_init=alpha_init,
        theta_init=theta_init,
        lambda0_se=lambda0_se,
        alpha_se=alpha_se,
        theta_se=theta_se,
        y_init=y_init,
        prior_scale=prior_scale,
        jitter=jitter,
    )

    l_init = _theta_to_l(theta_init)

    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            initvals={
                "eta0": np.log(float(lambda0_init)),
                "alpha": float(alpha_init),
                "log_l": np.log(l_init),
                "y": np.asarray(y_init, dtype=float).reshape(-1),
            },
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

    if return_model:
        return model, trace
    return trace

