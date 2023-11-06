# Control As Inference

POMDP&MPOの記事を読んでいてControl as Inferenceが何か気になったので書きました。

以下では、[この記事](https://dibyaghosh.com/blog/rl/controlasinference.html)を中心にしてControl as Inferenceの記述を行います。

### MDP (マルコフ決定過程)

以下では、ステップ数 $T$ の有限MDPをMDPと呼称します。また、以下の議論は容易にステップ数が無限大の場合に拡張できるようです[1]。加えて、以下ではMDPを $(\mathcal{S,A,T},\rho,R)$ の組とします。ここで、 $\mathcal{S,A}$ はそれぞれ状態空間と行動空間、 $T(\cdot|s,a)$ は遷移関数（kernel）、 $\rho$ は初期状態分布、 $R$ は報酬とします。

## グラフィカルモデル



## 参考文献

[1] [An Introduction to Control as Inference](https://dibyaghosh.com/blog/rl/controlasinference.html)