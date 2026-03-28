import numpy as np
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


def build_power_kernel_model(
    data,
    lambda0_init,
    alpha_init,
    gamma_init,
    lambda0_se,
    alpha_se,
    gamma_se,
    y_init,
    prior_scale=1.0,
    jitter=1e-6,
):
    """
    Power-law-kernel Poisson latent GP model

    Model
    -----
    X_t | eta0, alpha, y ~ Poisson(exp(eta0 + alpha y_t))
    y ~ N(0, K_pow)
    K_pow[i,j] = 1 / (1 + |t_i - t_j|)^gamma

    Parameter mapping
    -----------------
    lambda0 = exp(eta0)

    Priors
    ------
    eta0      ~ Normal(log(lambda0_init), prior_scale * sigma_eta0)
    alpha     ~ TruncatedNormal(alpha_init, prior_scale * alpha_se, lower=0)
    log_gamma ~ Normal(log(gamma_init), prior_scale * sigma_log_gamma)

    Notes
    -----
    - eta0 の事前標準偏差はデルタ法で sigma_eta0 ≈ lambda0_se / lambda0_init
    - ご指定どおり、
        Var(log gamma) ≈ gamma_se^2 / gamma_init^2
      を使うので、
        sigma_log_gamma ≈ gamma_se / gamma_init
    """
    x = _to_1d_count_array(data)
    T = len(x)

    y_init = np.asarray(y_init, dtype=float).reshape(-1)
    if len(y_init) != T:
        raise ValueError("y_init の長さが data と一致していません。")

    if lambda0_init <= 0:
        raise ValueError("lambda0_init は正である必要があります。")

    if gamma_init <= 0:
        raise ValueError("gamma_init は正である必要があります。")

    if lambda0_se <= 0:
        raise ValueError("lambda0_se は正である必要があります。")

    if alpha_se <= 0:
        raise ValueError("alpha_se は正である必要があります。")

    if gamma_se <= 0:
        raise ValueError("gamma_se は正である必要があります。")

    eta0_init = np.log(float(lambda0_init))
    sigma_eta0 = float(lambda0_se) / float(lambda0_init)

    mu_log_gamma = np.log(float(gamma_init))
    sigma_log_gamma = float(gamma_se) / float(gamma_init)

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

        log_gamma = pm.Normal(
            "log_gamma",
            mu=mu_log_gamma,
            sigma=prior_scale * sigma_log_gamma,
        )

        gamma = pm.Deterministic("gamma", pt.exp(log_gamma))
        lambda0 = pm.Deterministic("lambda0", pt.exp(eta0))

        K = (1.0 + dist) ** (-gamma) + jitter * pt.eye(T)

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


def sample_power_kernel_model(
    data,
    lambda0_init,
    alpha_init,
    gamma_init,
    lambda0_se,
    alpha_se,
    gamma_se,
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
    build_power_kernel_model を作成して MCMC まで回す。
    """
    model = build_power_kernel_model(
        data=data,
        lambda0_init=lambda0_init,
        alpha_init=alpha_init,
        gamma_init=gamma_init,
        lambda0_se=lambda0_se,
        alpha_se=alpha_se,
        gamma_se=gamma_se,
        y_init=y_init,
        prior_scale=prior_scale,
        jitter=jitter,
    )

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
                "log_gamma": np.log(float(gamma_init)),
                "y": np.asarray(y_init, dtype=float).reshape(-1),
            },
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

    if return_model:
        return model, trace
    return trace