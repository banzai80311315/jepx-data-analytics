# One-week-ahead electricity price forecasting using weather forecasts , and its application to arbitrage in the forward market: an empirical study of the JEPX

```
Step 1: データ整形
  - JEPXスポット価格を週次平均にする
  - base load: 24時間平均
  - daytime load: 平日 08:00–18:00 平均

Step 2: 最小モデルを作る
  - まず LNG なし
  - 天気なし
  - St ~ St-1 + seasonality
  - log(St) ~ log(St-1) + seasonality

Step 3: 温度を入れる
  - measured temperature residual
  - forecasted temperature residual
  - direct forecast と two-step forecast を分ける

Step 4: 論文モデルへ近づける
  - Fourier series で f(t), g(t), β(t) を近似
  - rolling 3年学習 → 1週先予測

Step 5: 評価
  - MAE
  - log-price bias correction
  - QRによる分位点予測
  - pinball loss

Step 6: 裁定戦略
  - forward価格がある週だけ評価
  - OLS戦略
  - QR戦略
```

## Paper model

$$
\log S_t
=
\alpha \log S_{t-1}
+
\beta(t)\log G_t
+
f(t)
+
g(t)\varepsilon_t
+
\eta_t
$$

- $G(t)$ : LNG価格
- $f(t)$ : seasonal trend
- $g(t)\varepsilon_t$ : 温度影響
- $\varepsilon_t$ : weekly average measured temperature residual (実測気温 - 季節平均)
- $\_t$ : 回帰誤差

## Construction of the L2 model

いきなすべての変数を入れたモデルを作れないので、まずは実測気温と季節性を入れた基本モデルを構築する

- $L$ : logarithmic series
- $2$ : uses the measured temperature

$$
\log{S_t} = \alpha \log{S_{t-1}} + \beta \text{Temp}_t  + \eta_t
$$