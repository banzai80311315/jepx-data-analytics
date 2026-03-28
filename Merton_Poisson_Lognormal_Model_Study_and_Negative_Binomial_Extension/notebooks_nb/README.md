# 研究運用ルール

---

## 0. 基本原則

- 型は固定する
- 初期値推定法と事前分布の設定法も固定する
- 変更するのは「潜在過程」または「観測尤度」のみ
- 一度に変更する軸は1つだけ

---

## 1. モデル構造（固定）

### 観測モデル
$$
X_t \sim p(X_t \mid \lambda_t)
$$

### リンク関数
$$
\lambda_t = \exp(\eta_0 + \alpha y_t)
$$

---

## 2. 潜在過程

### (A) 指数減衰
$$
y_t = \theta y_{t-1} + \sqrt{1-\theta^2}\,\epsilon_t,
\qquad
\epsilon_t \sim \mathcal{N}(0,1)
$$

- 短期記憶を表現
- AR(1)で実装
---

### (B) べき減衰

自己相関構造を以下で固定する：

$$
\rho(i) = \frac{1}{(1+i)^\gamma}\ , \ \gamma > 0
$$

- 長期記憶を表現
- Potentialで実装する

---

### (C) Gaussian Process（余裕があれば）
$$
y \sim \mathcal{GP}(0, k_\theta(t,t'))
$$

使用するカーネル候補は以下に固定する：

- Exponential
- Rational Quadratic
- Matérn

---

## 3. 観測尤度（段階的解放）

### Phase 1
Poisson を基準モデルとする：

$$
X_t \sim \mathrm{Poisson}(\lambda_t)
$$

### Phase 2
Negative Binomial ：

$$
X_t \sim \mathrm{NB} \left(\frac{\lambda_t}{\omega},\frac{1}{1 + \omega} \right)
$$

---

## 4. 初期値推定ルール（固定）

### 4.1 $\eta_0, \alpha$ の初期値

潜在変数 $$y_t$$ が独立であると仮定して、観測系列に対して MLE を実施し、

$$\eta_0^{init}\ , \ \alpha^{init}$$

を求める。

このとき、MLE から各推定量の分散も得る。

---

### 4.2 $y_t$ の初期値
$$ 
k_t \simeq \mathbb{E}[X_t|y_t] = \lambda_t = \exp(\eta_0 + \alpha y_t)
$$

から逆算して$\eta_0^{init}, \alpha^{init}$ を用いて

$$
y_t^{init} = \frac{\log{ (k_t + 0.5)}- \eta_0^{init}}{\alpha^{init}}
$$

とする。

---

### 4.3 $\theta, \gamma$ の初期値

初期化した $y_t^{init}$ の標本自己相関を用いて MLE を行う。

#### AR(1) の場合
$$
\theta^{init}
$$
は、$y_t^{init}$ の自己相関構造に対する AR(1) モデルの MLE とする。

#### power-law の場合
$$
\gamma^{init}
$$
は、$y_t^{init}$ の標本自己相関
$$
\hat{\rho}(i)
$$
に対して

$$
\hat{\rho}(i) \approx \frac{1}{(1+i)^\gamma}
$$

を最もよく当てはめる MLE とする。

このとき、$\theta$ および $\gamma$ の推定分散も取得する。

---

## 5. 推定ルール

### 5.1 パラメトリックモデル（AR / power）

1. MAP推定を必須とする
$$
(\hat{\eta}_0,\hat{\alpha},\hat{\psi},\hat{y})
=
\arg\max \log p(\eta_0,\alpha,\psi,y \mid x)
$$

ここで $\psi$ は $\theta$ または $\gamma$ を表す。

2. 必要に応じて MCMC を実施する

---

### 5.2 非パラメトリックモデル（GP）

Gaussian Process 回帰を行う。

- ハイパーパラメータ推定は MLE または MAP
- カーネル比較を実施する

---

## 6. 事前分布ルール（高度化版）

事前分布の平均は初期値推定結果を使い、分散は MLE で得た推定分散を使う。

---

### $\eta_0$ の事前分布
$$
\eta_0 \sim \mathcal{N}
\left(
\eta_0^{init},
\;
\mathrm{Var}(\hat{\eta}_0^{MLE})
\right)
$$

---

### $\alpha$ の事前分布
$$
\alpha \sim \mathcal{N}
\left(
\alpha^{init},
\;
\mathrm{Var}(\hat{\alpha}^{MLE})
\right)
$$

---

### $\theta$の事前分布

AR(1) の場合、$$\theta \in (0,1)$$ を仮定し、Beta 分布を用いる。

$$
\theta \sim \mathrm{Beta}(a_\theta, b_\theta)
$$

ここで

- モードが $\theta^{init}$ になるように設定する

---

### $\gamma$ の事前分布

power-law の場合、$\gamma > 0$ を仮定し、Gamma 分布を用いる。

$$
\gamma \sim \mathrm{Gamma}(a_\gamma, b_\gamma)
$$

ここで

- モードが $\gamma^{init}$ に一致

するように $$a_\gamma, b_\gamma$$ を決める。

---

### GP ハイパーパラメータ

GP の length-scale, variance についても、可能であれば初期推定値とその分散を使って事前分布を置く。

例：

$$
\ell \sim \mathrm{LogNormal}(\mu_\ell, \sigma_\ell^2)
$$

$$
\sigma_f \sim \mathrm{HalfNormal}(\tau_f)
$$

---

## 7. 可視化（必須）

各モデルについて必ず以下を出力する。

1. 観測 vs 推定強度
$$
x_t \quad \text{vs} \quad \hat{\lambda}_t
$$

2. 潜在因子
$$
\hat{y}_t
$$

3. 自己相関
$$
\mathrm{ACF}(\hat{y}_t)
$$

4. フィット確認図

---

## 8. 比較指標

数値指標として以下を用いる：

- WAIC
- LOO（可能なら）
- MAP objective

定性的には以下も評価する：

- フィットの自然さ
- 潜在因子の滑らかさ
- ACF の減衰構造
- 解釈可能性

---

## 9. 分析フロー

### Step 1：Poisson 固定で潜在比較

- Poisson + AR
- Poisson + power
- Poisson + GP

---

### Step 2：判定

#### Case A：AR 優勢
- 短期記憶
- 指数減衰

#### Case B：power 優勢
- 長期記憶
- 臨界現象の可能性

#### Case C：GP 優勢
- parametric モデルでは不足

#### Case D：識別困難
- AR と power の差が不明瞭
- GP でも決め手に欠ける

---

### Step 3：観測尤度の検証

Case C または Case D で、なおフィットが不十分な場合：

- NB + 最良潜在構造

を実行し、Poisson と比較する。

---

## 10. 解釈ルール

### AR 優勢
- 短期記憶
- 指数減衰
- 非臨界的挙動

### power 優勢
- 長期記憶
- べき減衰
- 臨界性の可能性

### GP 優勢
- 柔軟な相関構造が必要
- parametric 仮説では捉えきれない

### NB 改善
- 過分散が主要因
- 潜在構造だけでは説明不十分

---

## 11. 実装構造

```python
def estimate_init_eta0_alpha(data):
    ...

def estimate_init_y(data, eta0_init, alpha_init):
    ...

def estimate_init_theta(y_init):
    ...

def estimate_init_gamma(y_init):
    ...

def build_model_ar(data, init_values, prior_values):
    ...

def build_model_power(data, init_values, prior_values):
    ...

def build_model_gp(data, init_values, prior_values):
    ...

def fit_parametric(model):
    ...

def fit_gp(model):
    ...

def evaluate(result):
    ...

def run_pipeline(data):
    ...