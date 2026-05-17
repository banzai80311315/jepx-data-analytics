# 潜在状態付き時系列モデル（Latent State Time Series Model）

## モデル概要

背後にある潜在状態 $y_t$ を導入し、そのダイナミクスを通じて観測データを説明する

---
## ディレクトリ構成
自作アプリに使うライブラリpowerlibのディレクトリ構成は
```
src/
  powerlib/
    __init__.py

    io/ # 外部ファイルを読むだけ
      jepx_csv.py          # JEPX CSV読み込み
      schema.py            # 列名定義・標準カラム名

    preprocessing/ # 生データを分析可能な形に整える
      datetime.py          # 受渡日 + 時刻コード → datetime
      cleaning.py          # 数値変換、欠損処理、型変換
      scaling.py           # 休日・曜日補正

    features/ # 分析・モデルに使う列を作る
      time_features.py     # 時刻、曜日、休日、slot特徴量
      spike_features.py    # spike indicator, excess magnitude

    analysis/ # モデルではない集計分析
      summary.py           # 平均、分位点、基本統計
      distribution.py      # ヒストグラム用集計、分布分析
      time_profile.py      # 30分単位平均、時間帯別分析

    models/ # 予測・推定モデル
      hawkes.py            # Hawkesモデル
      threshold.py         # 閾値ベースモデル
      baseline.py          # persistenceなどベースライン

    metrics/ # 結果評価
      classification.py    # accuracy, precision, recall, F1, MCC
      forecasting.py       # MAE, WACCなど

    visualization/ # 図を作る
      timeseries.py        # plot用関数
      distribution.py      # histogram用関数
      spike.py             # spike可視化

    config.py              # 共通定数
    exceptions.py          # 独自例外
```

対応をするために、このプロジェクトディレクトリの責務も以下を意識する
```
io : 0.data_input
preprocessing : 1.data_processing
features , analysis : 2.feature_extraction
models : 3.model
metrics : 4.evaluation
```
---

## 1. 観測モデル（Observation Model）

$$
X_t \mid y_t \sim p(X_t \mid y_t, \phi)
$$

### ベルヌーイ
$$
X_t \sim \text{Bernoulli}(p_t), \quad
p_t = \frac{1}{1 + e^{-y_t}}
$$

### ポアソン
$$
X_t \sim \text{Poisson}(\lambda_t), \quad
\log \lambda_t = y_t
$$

### 負の二項分布
$$
X_t \sim \text{NB}(\mu_t, r), \quad
\log \mu_t = y_t
$$

---

## 2. 潜在状態モデル（State Model）

$$
y_t = \beta + \theta y_{t-1} + \boldsymbol{\omega}^\top \mathbf{z}_{t-1} + \sigma \varepsilon_t
$$

$$
\varepsilon_t \sim \mathcal{N}(0,1)
$$

---

## 3. 対数事後分布（Log Posterior）

$$
\phi = (\beta, \theta, \boldsymbol{\omega}, \sigma)
$$

$$
p(\phi, y \mid X) \propto p(X \mid y)\, p(y \mid \phi)\, p(\phi)
$$

### 状態尤度
$$
\log p(y \mid \phi)
=
- \frac{1}{2\sigma^2}
\sum_{t=2}^T
(y_t - \beta - \theta y_{t-1} - \boldsymbol{\omega}^\top \mathbf{z}_{t-1})^2
- (T-1)\log \sigma + C
$$

---

## 4. 観測尤度の違い

### Bernoulli
$$
\sum_t \left[X_t y_t - \log(1+e^{y_t})\right]
$$

### Poisson
$$
\sum_t \left[X_t y_t - e^{y_t}\right]
$$

### Negative Binomial
$$
\sum_t \left[X_t y_t - (X_t+r)\log(r+e^{y_t})\right]
$$

---


## 5. 推論手法

### 5.1 MCMC（NUTS）

$$
(\phi^{(s)}, y^{(s)}) \sim p(\phi, y \mid X)
$$

- 高精度（漸近的に真の事後分布に収束）
- 計算コスト大
- フルベイズ推定（不確実性を保持）

#### ハミルトニアンの定義

未知パラメータと潜在変数をまとめて

$$
q = (\phi, y)
$$

とおく。

補助変数として運動量

$$
p \sim \mathcal{N}(0, M)
$$

を導入し、拡張空間 $(q,p)$ を考える。

ハミルトニアンは

$$
H(q,p) = U(q) + K(p)
$$

で定義される。

#### ポテンシャルエネルギー

$$
U(q) = -\log p(q \mid X)
$$

（正規化定数は不要）

#### 運動エネルギー

$$
K(p) = \frac{1}{2} p^\top M^{-1} p
$$

#### 同時分布

$$
p(q,p \mid X) \propto \exp\{-H(q,p)\}
$$

### ハミルトン方程式

$$
\frac{dq}{dt} = M^{-1}p
$$

$$
\frac{dp}{dt} = \nabla \log p(q \mid X)
$$

#### 性質

- エネルギー保存
- 体積保存（Liouvilleの定理）
- 可逆性（reversibility）

### リープフロッグ法（離散化）

ステップサイズ $\epsilon$ を用いて：

$$
p \leftarrow p + \frac{\epsilon}{2} \nabla \log p(q \mid X)
$$

$$
q \leftarrow q + \epsilon M^{-1} p
$$

$$
p \leftarrow p + \frac{\epsilon}{2} \nabla \log p(q \mid X)
$$

これを $L$ 回繰り返す。

### Metropolis補正

$$
\alpha = \min\left(1,\exp(-H(q^*,p^*) + H(q,p))\right)
$$

により採択判定を行う。

### NUTS（No-U-Turn Sampler）

HMCのハイパーパラメータ

- $\epsilon$（ステップサイズ）
- $L$（軌道長）
- $M$（質量行列）

を自動調整する。

#### 各要素の扱い

- $\epsilon$：dual averaging により適応的に更新  
- $M$：サンプル共分散から推定  
- $L$：Uターン条件により動的決定  

#### Uターン条件

$$
(q_t - q_0)^\top p_t < 0
$$

となった時点で軌道の拡張を停止する。

## 本モデルへの適用

本研究のモデル

$$
\lambda_t = \beta \exp(\alpha y_t)
$$

$$
y_t = \theta y_{t-1} + \xi_t
$$

に対して、ポテンシャルエネルギーは

$$
U(q)
=
- \sum_{t=1}^T \left[
x_t(\log \beta + \alpha y_t)
- \beta e^{\alpha y_t}
\right]
+ \frac{1}{2\sigma^2}\sum_{t=2}^T (y_t - \theta y_{t-1})^2
- \log p(\phi)
$$

となる。

### 理論的利点

- 勾配情報を利用した効率的探索
- ランダムウォークの回避
- 高受理率（エネルギー保存）
- 詳細釣り合いに基づく正当性
### 5.2 変分推論（Variational Inference）

$$
q(\phi, y) \approx p(\phi, y \mid X)
$$

$$
\mathrm{KL}(q || p) を最小化
$$

- 高速
- スケーラブル
- 近似誤差あり

---

## 6. モデル評価（Model Evaluation）

### 6.1 LFO（Leave-Future-Out）

逐次予測性能：

$$
\log p(X_{t+1} \mid X_{1:t})
$$

$$
=
\log \mathbb{E}_{p(\phi,y|X_{1:t})}
\left[
p(X_{t+1}|\phi,y)
\right]
$$

全期間で：

$$
\sum_{t} \log p(X_{t+1} \mid X_{1:t})
$$

### 特徴

- 時系列に適した交差検証
- 未来情報リークなし
- モデル比較に最適

---

## 7. モデル解釈

- $y_t$：潜在状態（市場の内部状態）
- $\theta$：持続性
- $\boldsymbol{\omega}$：外生影響

---

## 8. モデリング思想

> 観測は潜在状態の非線形変換

---

## 9. 一般化

線形AR(1)モデル

$$
y_t = \beta + \theta y_{t-1} + \boldsymbol{\omega}^\top z_{t-1} + \sigma \varepsilon_t
$$

は、より一般に

$$
y_t = f(y_{t-1}, z_{t-1}) + \sigma \varepsilon_t
$$

と書ける。

---

## 10. ガウス過程拡張

この一般化に対して、潜在状態そのもの、あるいは状態遷移関数 $f$ を
ガウス過程でモデル化することができる。

### 10.1 潜在状態そのものをガウス過程とみなす方法

離散時点 $t = 1, \dots, T$ における潜在状態ベクトル

$$
\mathbf{y} = (y_1, \dots, y_T)^\top
$$

に対して、

$$
\mathbf{y} \sim \mathcal{N}(\mathbf{m}, K)
$$

と仮定する。

ここで

- $\mathbf{m}$：平均ベクトル
- $K$：共分散行列

である。たとえば平均関数を一定とすれば

$$
m_t = \beta
$$

より

$$
\mathbf{m} = \beta \mathbf{1}
$$

となる。

また、共分散行列はカーネル関数 $k(t,s)$ を用いて

$$
K_{ts} = k(t,s)
$$

と定義する。

したがって、

$$
\mathbf{y} \sim \mathcal{N}(\beta \mathbf{1}, K)
$$

となる。

### 10.2 代表的なカーネル

#### 指数カーネル

$$
k(t,s) = \alpha^2 \exp\left(-\frac{|t-s|}{\ell}\right)
$$

- $\alpha^2$：潜在状態の分散スケール
- $\ell$：相関の減衰速度を決める長さ尺度

このとき

$$
\operatorname{Cov}(y_t, y_s) = \alpha^2 \exp\left(-\frac{|t-s|}{\ell}\right)
$$

であり、時点が近いほど強く相関する。

このカーネルは連続時間OU過程や離散時間AR(1)と近い構造を持つ。

#### 二乗指数カーネル

$$
k(t,s) = \alpha^2 \exp\left(
-\frac{(t-s)^2}{2\ell^2}
\right)
$$

指数カーネルよりも滑らかな潜在軌道を与える。

#### 周期カーネル

電力価格のように日周期・週周期が重要な場合には、

$$
k(t,s)
=
\alpha^2
\exp\left(
-\frac{2\sin^2\left(\pi |t-s| / p\right)}{\ell^2}
\right)
$$

のような周期カーネルも考えられる。

- $p$：周期長
- $\ell$：周期内での滑らかさ

### 10.3 GP潜在状態モデルの全体構造

たとえば Bernoulli 観測なら、

$$
X_t \mid y_t \sim \operatorname{Bernoulli}(p_t)
$$

$$
p_t = \frac{1}{1+e^{-y_t}}
$$

$$
\mathbf{y} \sim \mathcal{N}(\beta \mathbf{1}, K)
$$

となる。

同様に Poisson 観測では

$$
X_t \mid y_t \sim \operatorname{Poisson}(\lambda_t), \quad
\lambda_t = e^{y_t}
$$

とすればよい。

このように、**観測モデルはそのままに、潜在状態の事前分布だけをAR(1)からGPへ置き換える**ことができる。

### 10.4 GP化したときの対数事前分布

潜在状態 $\mathbf{y}$ の事前分布が

$$
\mathbf{y} \sim \mathcal{N}(\beta \mathbf{1}, K)
$$

なら、その対数事前分布は

$$
\log p(\mathbf{y} \mid \beta, K)
=
-\frac{1}{2}
(\mathbf{y} - \beta \mathbf{1})^\top K^{-1} (\mathbf{y} - \beta \mathbf{1})
-\frac{1}{2}\log |K|
+ C
$$

である。

したがって、Bernoulli観測のときの対数事後分布は

$$
\log p(\phi, \mathbf{y} \mid X)
=
\sum_{t=1}^T
\left[
X_t y_t - \log(1+e^{y_t})
\right]
$$

$$
\quad
-\frac{1}{2}
(\mathbf{y} - \beta \mathbf{1})^\top K^{-1} (\mathbf{y} - \beta \mathbf{1})
-\frac{1}{2}\log |K|
+ \log p(\phi)
+ C
$$

となる。

ここで $\phi$ は、たとえば

$$
\phi = (\beta, \alpha, \ell)
$$

のように、平均とカーネルパラメータをまとめたものである。

Poisson観測なら第1項だけが

$$
\sum_{t=1}^T (X_t y_t - e^{y_t})
$$

に置き換わる。

### 10.5 AR(1)モデルとの関係

AR(1)状態モデル

$$
y_t = \beta + \theta y_{t-1} + \sigma \varepsilon_t
$$

は、局所的な一次マルコフ構造を持つ。

一方、GPモデルでは

$$
\mathbf{y} \sim \mathcal{N}(\beta \mathbf{1}, K)
$$

とまとめて表すため、

- より一般の相関構造を導入できる
- 周期性や長期依存を自然に組み込める
- 外生変数なしでも柔軟な時系列依存を表現できる

という利点がある。

ただし、

- $K^{-1}$ や $\log|K|$ の計算が必要
- 計算量が大きい
- 長系列では近似が必要

という欠点もある。

### 10.6 遷移関数そのものをガウス過程化する方法

もう一つの拡張として、状態遷移を

$$
y_t = f(y_{t-1}, \mathbf{z}_{t-1}) + \sigma \varepsilon_t
$$

と書き、この関数 $f$ 自体に対して

$$
f \sim \mathrm{GP}(m, k)
$$

を仮定する方法がある。

この場合、GPの対象は潜在状態列 $\mathbf{y}$ そのものではなく、  
**「1期前の状態と外生変数から次の状態を生成する関数」** である。

この拡張により、

- 非線形な自己回帰
- 外生変数との非線形相互作用
- 線形AR(1)では表せない複雑な遷移

を記述できる。

ただし、この形式はモデルも推論もかなり重くなるため、  
まずは

1. AR(1)状態モデル  
2. 潜在状態GPモデル  

の順で考える方が実装上は自然である。

### 10.7 この研究における位置づけ

重要なのは、潜在状態 $y_t$ を

- AR(1)で置くか
- ガウス過程で置くか

を比較することである。

すなわち、

- AR(1)：簡潔で計算しやすい
- GP：柔軟で多様な依存構造を表現できる

という対比になる。

したがって、本モデルにおけるガウス過程拡張とは、  
基本的には

> **潜在状態の事前分布を AR(1) から GP に置き換えること**

を意味する。

---

## 11. 位置づけ

- 状態空間モデル
- 動的GLM
- 潜在ガウスモデル

---

