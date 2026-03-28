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


実装内容

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