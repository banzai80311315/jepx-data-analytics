# 先行研究の理解と再現

--- 
## 概要
先行研究の論文を精読し、以下を理解する。

- Merton モデルの構造
- 共通因子モデル
- default probability の導出
- logistic 近似
- lognormal intensity の導出
- Poisson 極限
- Impact Analysis
- Super-normal transition
- パラメータ推定法
- Bayesian 推定
- LFO / WAIC / WBIC


特に、このnotebooksでは推定部分では以下の流れで**先行研究のモデルを Python で実装する。**

$$
\text{Poisson MLE}  
\to  
潜在変数 y_t の復元  
\to 
\text{ACF} による相関パラメータ推定  
\to  
\text{Bayesian} 推定  
\to
モデル評価
$$


**実装内容notebooks**

- notebook「01_data_check.ipynb」
    - Moodys 1920から2018年のデータを確認
    - 分析用に加工
- notebook 「study_reproduction1_init.ipynb」
    - Poisson-lognormal モデル
    - MLE 推定（$\alpha , \lambda_0$の初期値）
    - 潜在変数復元（$y_t$の初期値）
    - ACF 推定（$\theta , \gamma$の初期値）
- notebook 「study_reproduction2_bayesian.ipynb」
    - exponential decayでのベイズ推定
    - power-law decayでのベイズ推定
    - モデル比較（WAIC,LFO）

--- 

## モデル
### モデル
潜在マクロ因子の確率モデル$y_t$
$$
y_t \sim \mathcal{N}(0,1)\ ,\ d_i = \mathbb{E}[y_t y_{t+i}]
$$
マートンモデル近似後の条件付きデフォルト数分布
$$
X_t|y_t \sim \text{Poisson}(\lambda_t)\ , \ \lambda_t = \lambda_0 \exp{(\alpha y_t)} \sim \mathcal{LN}(\log \lambda_0 , \alpha^2)
$$
ただし、
$$
\alpha^2 = \frac{\rho_A}{1-\rho_A}\beta^2 \ , \ \beta=1.596
$$
ここで$\rho_A$は、標準化された企業の資産を$U_{it}$($i$は企業、$t$は時間)として
$$
U_{it} = \sqrt{\rho_A} y_t + \sqrt{1-\rho_A}\epsilon_{it}
$$

--- 

### 統計量の理論値
#### 強度$\lambda_t$
$$
\mathbb{E}[\lambda_t] = \exp{(\log{\lambda_0} + \frac{\alpha^2}{2})} = \bar{\lambda} \ , \ Var(\lambda_t) = \bar{\lambda}^2 (\exp{\alpha^2}-1) = \bar{V}
$$
#### 無条件のデフォルト$X_t$
条件付きのデフォルト$X_t|y_t$と期待値の繰り返し公式を利用して
$$
\mathbb{E}[X_t] = \mathbb{E}[\mathbb{E}[X_t|y_t]] = \mathbb{E}[\lambda_t]
$$
分散も同様に
$$
Var(X_t) = \mathbb{E}[Var[X_t|y_t]] + Var[\mathbb{E}[X_t|y_t]] = \mathbb{E}[\lambda_t] + Var[\lambda_t]
$$
分散比の理論値は
$$
\frac{Var(X_t)}{\mathbb{E}[X_t]} = 1 + \frac{\bar{V}}{\bar{\lambda}} = 1 + \bar{\lambda} (\exp{\alpha^2}-1)
$$

- 個別企業がマクロの影響を受けない$\alpha = 0$ならデフォルト数は条件付きではないただのポアソン分布
- $\alpha$が大きいほど、$\lambda_t$の揺れが大きくなり、過分散が強くなる

--- 
## 事前分布の初期値決定法
### $\lambda_0,\alpha$
```python
estimate_poisson_mle
```
- モーメント法により$\lambda_0$を決定、$\alpha$は適当に決定
- 観測データ$k^*$と上記パラメータにより、暫定的に$y_t$を決定
- MLEを用いて初期値$\lambda_0,\alpha$を決定
- 分散は対数パラメータに対する数値ヘッセ行列の逆行列からデルタ法により換算

### $y_t$
```python
restore_latent_y
```
以下で復元
$$
y_t = \frac{\log k_t - \log {\lambda_0}}{\alpha}
$$
### $\theta , \gamma$
```python
fit_acf_exponential
fit_acf_power_law

```
- 復元した$y_t$から自己相関関数を計算
- 理論値とMLE
- 分散は非線形最小二乗推定の共分散行列

MLEは以下のLOSSを最小とするパラメータを選ぶ
$$
\text{LOSS} = \sum_{lag} (\hat{\rho}(lag) - d(lag;\theta \ \text{or}\  \gamma))^2
$$

### 結果
| parameter | estimate (SE) | \[table1\]paper estimate (SE) |
|----------|--------------------|---------------------|
| $\lambda_0$  | 15.4 (0.48) | 18.1 (0.1)        |
| $\alpha$    | 1.60 (0.031)  | 1.4 (2.6)         |
| $\theta$    | 0.871 (0.0085)  | 0.890 (0.004)     |
| $\gamma$    | 0.554 (0.042)  | 0.64 (0.09)       |

--- 

## Exponential-kernel model

$$
X_t \mid \eta_0,\alpha,\mathbf y
\sim
\text{Poisson}\!\left(\exp(\eta_0+\alpha y_t)\right)
$$

$$
\mathbf y \sim \mathcal N(\mathbf 0, K^{(\mathrm{exp})})
$$

$$
K^{(\mathrm{exp})}_{ij}
=
\exp\!\left(-\frac{|t_i-t_j|}{l}\right)
$$

$$
\theta = \exp(-1/l), \qquad
l = -\frac{1}{\log \theta}
$$

$$
\eta_0 \sim \mathcal N(\eta_0^{init}, c\sigma_{\eta_0}^2)
$$

$$
\alpha \sim \mathcal N^+(\alpha^{init}, c\sigma_{\alpha}^2)
$$

$$
\log l
\sim
\mathcal N\!\left(
\log\!\left(-\frac{1}{\log \theta^{init}}\right),
c\sigma_{l}^2
\right)
$$

## Power-law-kernel model

$$
X_t \mid \eta_0,\alpha,\mathbf y
\sim
\text{Poisson}\!\left(\exp(\eta_0+\alpha y_t)\right)
$$

$$
\mathbf y \sim \mathcal N(\mathbf 0, K^{(\mathrm{pow})})
$$

$$
K^{(\mathrm{pow})}_{ij}
=
\frac{1}{(1+|t_i-t_j|)^\gamma}
$$

$$
\eta_0 \sim \mathcal N(\eta_0^{init}, c\sigma_{\eta_0}^2)
$$

$$
\alpha \sim \mathcal N^+(\alpha^{init}, c\sigma_{\alpha}^2)
$$

$$
\log \gamma \sim \mathcal N(\log \gamma^{init}, c\sigma_{\gamma}^2)
$$

### 暫定結果
$sc = 5$
| Dataset | $\lambda_0$ (Exp) | $\alpha$ (Exp) | $\theta$ | $\lambda_0$ (Pow) | $\alpha$ (Pow) | $\gamma$ |
|--------|----------|--------|----|-----------|--------|----|
| Moody’s ALL | 18.1 (0.4) | 1.6 (0.2) | 0.88 (0.01) | 18.1 (0.4) | 1.3 (0.3) | 0.4 (0.2) |
| This work (Exp) | 15.964 (2.363) | 1.561 (0.132) | 0.859 (0.030) | - | - | - |
| This work (Pow) | -| - | - | 15.554 (2.262) | 1.499 (0.133) | 0.349 (0.090) |

$sc = 1$
| Dataset | $\lambda_0$ (Exp) | $\alpha$ (Exp) | $\theta$ | $\lambda_0$ (Pow) | $\alpha$ (Pow) | $\gamma$ |
|--------|------------------|----------------|----------|------------------|----------------|----------|
| Moody’s ALL | 18.1 (0.4) | 1.6 (0.2) | 0.88 (0.01) | 18.1 (0.4) | 1.3 (0.3) | 0.4 (0.2) |
| This work (Exp) | 15.407 (0.475) | 1.607 (0.031) | 0.871 (0.008) | - | - | - |
| This work (Pow) | - | - | - | 15.378 (0.475) | 1.592 (0.031) | 0.523 (0.038) |

$sc$の変更に対して$\lambda_0$のposteriorは大きく変化しており、$\lambda_0$はprior依存性が強い。

一方、指数減衰モデルの$\alpha,\theta$は比較的頑健に見えるが、べき減衰モデルの$\gamma$はpriorの影響を無視できず、相関パラメータ全般が安定であるとは現時点で断定できない

### $\lambda_0$の分散が広くなっていることに対する考察

**識別可能性**
:観測データからパラメータの真の値を一意に特定できるか、あるいは事後分布が特定のパラメータ一点に収束するかどうか

このモデルは識別不可能なモデルで$\lambda_0 , y$が異なっても、$\lambda_t$が同じになり$\text{Poisson}(\lambda_t)$が同じになる。つまり尤度が同じになる。

論文にも記載があるが観測できるのは$\lambda_t$であり$\lambda_0,y$ではないことから、異なるパラメータが同じ尤度になってしまう。

--- 
## モデル評価

$$
\text{LFO}
= \sum_{t=t_0}^{T-1} \log{p(x_{t+1}|\vec{x})}
= \sum_{t=t_0}^{T-1} 
\log\left( \int d\theta \ p(x_{t+1} | \theta , \vec{x}) p(\theta | \vec{x})  \right) 
$$
$$
\simeq 
\sum^{T-1}_{t=t_0} \log{ \left(\frac{1}{S} \sum^S_{s=1} 
\left(\frac{1}{R}\sum^{R}_{r=1}\text{Poisson}(x_{t+1} | \exp{(\eta_0^{(s)}} + \alpha^{(s)} y_{t+1}^{(s,r)}))\right)\right) }
$$
