# A Hawkes Model Approach to Modeling Price Spikes in the Japanese Electricity Market
[A Hawkes Model Approach to Modeling Price Spikes in the Japanese Electricity Market](https://share.google/UHuffuuGfEOrYFusC)

## モデル
### Hawkes I

$$
\lambda_d = \alpha \lambda_{d-1} + \beta + \gamma u_d
$$

$$
\alpha = \exp{\left(-\frac{1}{\tau}\right)}\ , \ \beta = \mu(1- \alpha)
$$

$$
\gamma : \text{スパイクが来た時のジャンプ量}
$$

### Hawkes II
強いショックは強い連鎖を生む

$$
\lambda_d = \alpha \lambda_{d-1} + \beta + \gamma_d u_d
$$

$$
\gamma_d = \gamma_0 (1 - \exp{\left( -\frac{x_d}{x_0}\right)})
$$

### Hawkes III
強いショックは長く尾を引く

$$
\lambda_d = \alpha_d \lambda_{d-1} + \beta + \gamma u_d
$$

$$
\tau_d = \tau_0 (1 - \exp{\left( -\frac{x_d}{x_0}\right)})
$$

## 論文の結論
Hawkes II が優位：「スパイクの強さが、その後の発生確率に聞いてくる」