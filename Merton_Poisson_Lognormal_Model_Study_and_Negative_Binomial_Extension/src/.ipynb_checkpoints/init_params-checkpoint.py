from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit


@dataclass
class InitParams:
    lambda0_init: float
    alpha_init: float
    y_init: np.ndarray
    theta_init: float
    gamma_init: float

    lambda0_se: float
    alpha_se: float
    theta_se: float
    gamma_se: float

    lambda0_var: float
    alpha_var: float
    theta_var: float
    gamma_var: float

    acf_lags: np.ndarray
    acf_values: np.ndarray


def _validate_positive_series(x: np.ndarray, name: str = "x") -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    if np.any(~np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values")
    if len(x) < 5:
        raise ValueError(f"{name} must have length >= 5")

    return x


def _poisson_neg_loglik_eta_alpha(params: np.ndarray, x: np.ndarray) -> float:
    eta0, log_alpha = params
    alpha = np.exp(log_alpha)

    z = np.log(x + 0.5)
    z = (z - z.mean()) / (z.std(ddof=0) + 1e-8)

    log_mu = eta0 + alpha * z
    mu = np.exp(log_mu)

    return float(np.sum(mu - x * log_mu))


def _numerical_hessian(func, params: np.ndarray, args=(), eps: float = 1e-5) -> np.ndarray:
    """
    数値ヘッセ行列（中心差分）
    """
    params = np.asarray(params, dtype=float)
    n = len(params)
    hess = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = eps
            ej[j] = eps

            fpp = func(params + ei + ej, *args)
            fpm = func(params + ei - ej, *args)
            fmp = func(params - ei + ej, *args)
            fmm = func(params - ei - ej, *args)

            hess[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)

    return hess


def estimate_poisson_mle(
    x: np.ndarray,
    eta0_guess: Optional[float] = None,
    alpha_guess: float = 1.0,
) -> tuple[float, float, float, float]:
    """
    Returns
    -------
    lambda0_init, alpha_init, lambda0_se, alpha_se
    """
    x = _validate_positive_series(x, name="x")

    if eta0_guess is None:
        eta0_guess = float(np.log(np.mean(x) + 1e-8))

    init = np.array([eta0_guess, np.log(max(alpha_guess, 1e-3))], dtype=float)

    result = minimize(
        _poisson_neg_loglik_eta_alpha,
        x0=init,
        args=(x,),
        method="L-BFGS-B",
    )

    if not result.success:
        eta0_hat = eta0_guess
        log_alpha_hat = np.log(alpha_guess)
    else:
        eta0_hat = float(result.x[0])
        log_alpha_hat = float(result.x[1])

    alpha_hat = float(np.exp(log_alpha_hat))
    lambda0_hat = float(np.exp(eta0_hat))

    # 数値ヘッセ行列 -> 共分散行列
    try:
        hess = _numerical_hessian(
            _poisson_neg_loglik_eta_alpha,
            np.array([eta0_hat, log_alpha_hat]),
            args=(x,),
            eps=1e-5,
        )
        cov = np.linalg.inv(hess)

        se_eta0 = float(np.sqrt(max(cov[0, 0], 0.0)))
        se_log_alpha = float(np.sqrt(max(cov[1, 1], 0.0)))

        # デルタ法
        lambda0_se = lambda0_hat * se_eta0
        alpha_se = alpha_hat * se_log_alpha

    except np.linalg.LinAlgError:
        lambda0_se = np.nan
        alpha_se = np.nan

    return lambda0_hat, alpha_hat, lambda0_se, alpha_se


def restore_latent_y(
    x: np.ndarray,
    lambda0_init: float,
    alpha_init: float,
    eps: float = 0.5,
    standardize: bool = True,
) -> np.ndarray:
    x = _validate_positive_series(x, name="x")

    if lambda0_init <= 0:
        raise ValueError("lambda0_init must be positive")
    if alpha_init <= 0:
        raise ValueError("alpha_init must be positive")

    y = (np.log(x + eps) - np.log(lambda0_init)) / alpha_init

    if standardize:
        y = (y - y.mean()) / (y.std(ddof=0) + 1e-8)

    return y


def sample_acf(y: np.ndarray, max_lag: int = 20) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)

    if y.ndim != 1:
        raise ValueError("y must be 1-dimensional")
    if len(y) <= max_lag:
        raise ValueError("len(y) must be greater than max_lag")

    y_centered = y - y.mean()
    denom = np.dot(y_centered, y_centered)

    lags = np.arange(1, max_lag + 1)
    acf_vals = np.empty(max_lag, dtype=float)

    for i, h in enumerate(lags):
        numer = np.dot(y_centered[:-h], y_centered[h:])
        acf_vals[i] = numer / (denom + 1e-12)

    return lags, acf_vals


def _exp_acf_model(h: np.ndarray, theta: float) -> np.ndarray:
    return np.exp(-h / theta)


def _pow_acf_model(h: np.ndarray, gamma: float) -> np.ndarray:
    return (1.0 + h) ** (-gamma)


def fit_acf_exponential(
    lags: np.ndarray,
    acf_vals: np.ndarray,
    theta_init: float = 5.0,
) -> tuple[float, float]:
    popt, pcov = curve_fit(
        lambda h, ell: np.exp(-h / ell),
        lags,
        acf_vals,
        p0=[theta_init],
        bounds=(1e-6, np.inf),
        maxfev=10000,
    )

    ell = float(popt[0])
    ell_var = float(max(pcov[0, 0], 0.0))

    # 変換
    theta = np.exp(-1.0 / ell)

    # デルタ法
    dtheta_dell = np.exp(-1.0 / ell) * (1.0 / ell**2)
    theta_var = (dtheta_dell**2) * ell_var
    theta_se = np.sqrt(theta_var)

    return float(theta), float(theta_se)


def fit_acf_power_law(
    lags: np.ndarray,
    acf_vals: np.ndarray,
    gamma_init: float = 0.7,
) -> tuple[float, float]:
    popt, pcov = curve_fit(
        _pow_acf_model,
        lags,
        acf_vals,
        p0=[gamma_init],
        bounds=(1e-6, np.inf),
        maxfev=10000,
    )
    gamma_init = float(popt[0])
    gamma_var = float(max(pcov[0, 0], 0.0))
    gamma_se = float(np.sqrt(gamma_var))
    return gamma_init, gamma_se


def estimate_acf_params(
    y: np.ndarray,
    max_lag: int = 20,
    clip_negative: bool = True,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    lags, acf_vals = sample_acf(y, max_lag=max_lag)

    fit_vals = acf_vals.copy()
    if clip_negative:
        fit_vals = np.clip(fit_vals, 1e-6, None)

    theta_init, theta_se = fit_acf_exponential(lags, fit_vals)
    gamma_init, gamma_se = fit_acf_power_law(lags, fit_vals)

    return theta_init, gamma_init, theta_se, gamma_se, lags, acf_vals


def build_init_params(
    x: np.ndarray,
    max_lag: int = 20,
    eps: float = 0.5,
    standardize_y: bool = True,
) -> InitParams:
    x = _validate_positive_series(x, name="x")

    lambda0_init, alpha_init, lambda0_se, alpha_se = estimate_poisson_mle(x)

    y_init = restore_latent_y(
        x,
        lambda0_init=lambda0_init,
        alpha_init=alpha_init,
        eps=eps,
        standardize=standardize_y,
    )

    theta_init, gamma_init, theta_se, gamma_se, lags, acf_vals = estimate_acf_params(
        y_init,
        max_lag=max_lag,
    )

    return InitParams(
        lambda0_init=lambda0_init,
        alpha_init=alpha_init,
        y_init=y_init,
        theta_init=theta_init,
        gamma_init=gamma_init,
        lambda0_se=lambda0_se,
        alpha_se=alpha_se,
        theta_se=theta_se,
        gamma_se=gamma_se,
        lambda0_var=lambda0_se**2 if np.isfinite(lambda0_se) else np.nan,
        alpha_var=alpha_se**2 if np.isfinite(alpha_se) else np.nan,
        theta_var=theta_se**2 if np.isfinite(theta_se) else np.nan,
        gamma_var=gamma_se**2 if np.isfinite(gamma_se) else np.nan,
        acf_lags=lags,
        acf_values=acf_vals,
    )


def init_params_to_series(init_params: InitParams, name: str = "ALL") -> pd.Series:
    return pd.Series(
        {
            "group": name,
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
    )