# jepx-data-analytics

# Merton–Poisson–Lognormal Model Study and Negative Binomial Extension

本プロジェクトは、先行研究  

[Merton model and Poisson process with Log Normal intensity function](https://arxiv.org/abs/2505.13822)

を理解・再現し、そのモデルを **Negative Binomial 分布へ拡張**し、  
さらに **電力市場データ（JEPX）への応用可能性を検証する研究プロジェクト**である。

---

# 研究背景

信用リスクや極端イベントの発生回数は、しばしば **Poisson 過程**でモデル化される。

先行研究では

- Merton モデル
- 共通因子モデル
- rare-event limit

を用いて

Poisson process with lognormal intensity

というモデルを導出している。

このモデルでは

$$
X_t | Y_t  \sim Poisson(\lambda(Y_t))
$$

$$
\lambda(Y_t) = \lambda_0 \exp(\alpha Y_t)
$$

$$
Y_t : 時系列相関を持つ潜在因子
$$

となり

- exponential decay
- power-law decay

の2種類の相関モデルを比較し  
**LFO・WAIC・WBIC**などの指標で評価している。

---

# 研究目的

本研究の目的は以下の3点である。

1. 先行研究のモデルを数式レベルで理解し再現する  
2. Poisson 近似を **Negative Binomial 近似へ拡張**する  
3. 電力データへ適用可能か検証する  

特に Poisson モデルでは扱えない

- 過分散
- イベント群発

などを説明できるか確認する。

---

# 研究内容

## 1 先行研究の理解

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

特に推定部分では以下の流れを理解する。

Poisson MLE  
↓  
潜在変数 y_t の復元  
↓  
ACF による相関パラメータ推定  
↓  
Bayesian 推定  

---

## 2 モデルの再現

先行研究のモデルを Python で実装する。

実装内容

- Poisson-lognormal モデル
- MLE 推定
- 潜在変数復元
- ACF 推定
- exponential decay
- power-law decay
- モデル比較

---

## 3 Negative Binomial 拡張

先行研究の Poisson 近似を以下のように拡張する。

### 元モデル
$$
X_t | Y_t \sim Poisson(\lambda(Y_t))
$$

### 拡張モデル
$$
X_t | Y_t \sim NegativeBinomial(\mu(Y_t), \omega)
$$

ここで

$$
\mu(Y_t) = \lambda_0 \exp(\alpha Y_t)
$$

$\omega$ = 過分散パラメータである。

このモデルは
$$
Poisson model  
⊂  
Negative Binomial model  
$$
という包含関係を持つ。

---

## 4 モデル比較

Poisson モデルと Negative Binomial モデルを比較する。

比較指標

- Log likelihood
- AIC
- BIC
- WAIC
- WBIC
- LFO
- 平均再現性
- 分散再現性
- tail fit
- generalization performance

---

## 5 電力データへの応用

信用リスクデータの代わりに  
**電力市場データ（JEPX）**を用いて同様の分析を行う。

対象データは

価格スパイク発生回数

などの **イベントカウントデータ**とする。

例

- 日次スパイク回数  
- 週次スパイク回数  
- 閾値超過回数  

その上で

- Poisson model  
- Negative Binomial model  

を適用し

- 過分散
- 群発現象

の説明力を検証する。

---

# 研究スケジュール（1年間）

## Phase 1（1〜3ヶ月）

先行研究理解

- 論文精読
- 数式導出の再現
- 推定アルゴリズム理解

成果物

- 論文読解ノート
- 数式まとめ

---

## Phase 2（4〜6ヶ月）

モデル再現

- Poisson モデル実装
- 推定コード作成
- 論文再現実験

成果物

- Python 実装
- 再現レポート

---

## Phase 3（7〜9ヶ月）

NB 拡張

- Negative Binomial モデル定式化
- 推定アルゴリズム作成
- Poisson モデルとの比較

成果物

- 拡張モデル数式
- 比較実験

---

## Phase 4（10〜12ヶ月）

電力データ応用

- JEPX データ整形
- スパイク検出
- モデル適用
- 結果分析

成果物

- 電力データ分析コード
- 最終レポート

---

# 使用予定データ

- 信用リスクデータ（論文）
- JEPX 電力価格データ

---

# 使用技術

- Python
- NumPy
- pandas
- SciPy
- statsmodels
- PyMC / Stan
- matplotlib

---

# 成果物

最終的に以下を作成する。

- モデル実装コード
- Negative Binomial 拡張モデル
- モデル比較実験
- 電力データ分析
- 研究レポート

---

# 最終目標

以下を達成することを目標とする。

- 先行研究モデルの完全理解
- モデル再現
- NB 拡張モデル構築
- Poisson vs NB の比較
- 電力市場データへの応用

---

# Keywords

Merton model  
Poisson process  
Lognormal intensity  
Negative Binomial  
Rare events  
Credit risk  
Electricity price spike  
Critical phenomena