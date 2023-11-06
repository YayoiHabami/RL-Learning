# ELBO・変分推論

## はじめに

以下では、**ELBO**（*evidence lower bound*）と、その理解の助けになる**変分推論**（*variational inference*; 変分ベイズとも）を説明します。ここで、変分推論は**変分法**（*calculus of variations*）に起源をもつ近似的な推論法ですが、その主目的がELBOの最大化であり、イメージをつかむ手助けになることからこれを付記します。

## 目次

- [ELBO・変分推論](#elbo変分推論)
  - [はじめに](#はじめに)
  - [目次](#目次)
  - [ELBOを簡潔に](#elboを簡潔に)
  - [変分推論](#変分推論)
    - [概要：変分推論](#概要変分推論)
    - [ELBOの導出](#elboの導出)
    - [補足：ELBOの最大化](#補足elboの最大化)
  - [参考文献](#参考文献)


## ELBOを簡潔に

観測変数を $\pmb{X}$ 、潜在変数を $\pmb{Z}$ とします。対数周辺尤度 $\log p(\pmb{X})$ について、 $\pmb{Z}$ に従う任意の分布 $q(\pmb{Z})$ を用いて次のように書けます。

$$\begin{aligned}\log p(\pmb{X})=&\log\left(\int_q p(\pmb{X,Z})d\pmb{Z}\right)\\=&\log\left(\int_q q(\pmb{Z})\frac{p(\pmb{X,Z})}{q(\pmb{Z})}d\pmb{Z}\right)\\\geq&\int_q{q(\pmb Z)\log\frac{p(\pmb{Z,X})}{p(\pmb{Z})}d\pmb{Z}}\space\space(\because Jensen's inequality)\\=&\mathrm{ELBO}(q)\end{aligned}$$

このとき対数周辺尤度 $\log p(\pmb{X})$ の下限が $\mathrm{ELBO}(q)$ に定まり、これを**ELBO**（*evidence lower bound*）と呼びます。

## 変分推論

### 概要：変分推論

すべてのパラメタが事前分布を与えられた、完全にベイズ的なモデルを仮定します。モデルの（パラメタではない）潜在変数を $\pmb{Z}$ と書き、観測変数を $\pmb{X}$ とします。例えば、同時独立分布から発生した $N$ 個のデータがある場合、

$$\pmb{X}=\{\pmb{x}_1,...,\pmb{x}_N\}, \pmb{Z}=\{\pmb{z}_1,...,\pmb{z}_N\}$$

と書けます。ここで、EMアルゴリズムの議論と異なりパラメタベクトル $\theta$ を示さず、確率変数として $\pmb Z$ の中に含まれることとします[1]。ここから事後分布 $p(\pmb{Z}|\pmb{X})$ を求めるのがベイズ推定などでの目的ですが、ベイズの定理により出てくる周辺尤度 $\log p(\pmb X)$ は計算が大変であり、この事後分布は扱いにくいです。

> 計算が大変：解析式がない・計算が難解など

**変分推論**（*variational inference*）では、最適化を利用することで、事後分布 $p(\pmb{Z|X})$ に近い $q(\pmb Z)$ を見つけることを目的とします。ここで確率分布 $q(\pmb Z)$ は分布の集合 $\mathcal{Q}\left(\ni q(\pmb Z)\right)$ に属する、事後分布 $p(\pmb{Z|X})$ に比して単純で扱いやすい分布です[4]。この推定においては、KL情報量を利用して次のように最適化を行います[4]：

$$q^*(\pmb{Z})=\argmin_{q(\pmb{Z})\in\mathcal{Q}}\mathrm{KL}[q(\pmb{Z})\|p(\pmb{Z|X})]\tag{1}$$

> KL情報量の最小化は、２つの分布のエントロピーの最小化とも解釈することができます。
>
> 分布の集合 $\mathcal{Q}$ : 同じ確率密度関数 $q$ を持つが、そのパラメタ $\pmb Z$ が異なる分布（の集合）

<img src="imgs/Variational_inference_optimization.png" width=400>

図1. 変分推論における最適化のイメージ[4]

### ELBOの導出

ところで式 $(1)$ のKL情報量は周辺尤度 $p(\pmb{X})$ に依存しており、依然として計算の難解さは残っています。KL情報量を展開すると

$$\begin{aligned}
\mathrm{KL}[q(\pmb{Z})\|p(\pmb{Z|X})]=&\int_q{q(\pmb{Z})\log\frac{q(\pmb{Z})}{p(\pmb{Z|X})}d\pmb{Z}}\\
=&\int_q{q(\pmb{Z})\log\frac{q(\pmb{Z})p(\pmb{X})}{p(\pmb{Z|X})p(\pmb{X})}d\pmb{Z}}\\
=&\underbrace{-\int_q{q(\pmb{Z})\log\frac{p(\pmb{Z,X})}{q(\pmb{Z})}d\pmb{Z}}}_{-\mathrm{ELBO}(q)} + \log p(\pmb{X})
\end{aligned}\tag{2}$$

ここで、右辺の第一項と第二項の和から、**ELBO**（*Evidence lower bound*）を次のように定義します。

$$\mathrm{ELBO}(q)=\int_q{q(\pmb Z)\log\frac{p(\pmb{Z,X})}{p(\pmb{Z})}d\pmb{Z}}\tag{3}$$

ここで、なぜ式 $(3)$ がELBOと呼ばれるかですが、KL情報量の非負性を用いて式 $(2)$ を書き直すと、

$$\begin{aligned}\log p(\pmb{X})=&\mathrm{ELBO}(q)+\mathrm{KL}[q(\pmb{Z})\|p(\pmb{Z|X})]\\
\geq&\mathrm{ELBO}(q)\end{aligned}$$

が導けます。対数周辺尤度 $\log p(\pmb{X})$ は *evidence* とも呼ばれることがあり[3]、上式からELBOが evidence の下限であることがわかりました。

また、対数周辺尤度 $\log p(\pmb{X})$ はデータ $\pmb{X}$ が定まれば定数となります。したがって、式 $(1)$ の最小化問題は**ELBOの最大化問題**と捉えなおすことができます。

### 補足：ELBOの最大化

ELBOそれ自体の説明には関係はありませんが、変分推論における最適化の大まかな進め方を以下に示します。

以下では、モデル上、真の事後分布は求めることが不可能であると仮定します。すなわち、分布 $q(\pmb{Z})$ を事後分布 $p(\pmb{Z|X})$ と等しくする（KL情報量を $0$ とする）ことが不可能であるとします。このとき、代わりにある制限したクラスの $q(\pmb Z)$ を考え、この中でKL情報量を最小化するものを探すこととします。

ここで、 $\pmb Z$ の要素をいくつかの背反なグループに分割し、 $\pmb{Z}_i (i=1,...,M)$ と書くこととします。すべての $\pmb{Z}_1,...,\pmb{Z}_M$ が互いに独立であり、分布 $q$ がこれらグループに関して分解されると仮定すると、

$$q(\pmb Z)=\prod_{i=1}^M{q_i(\pmb{Z}_i)}\tag{5}$$

となります。ここでは分布について、これ以上の仮定はしておらず、特に各因子 $q_i(\pmb{Z}_i)$ の関数形については何の制限もしていません[1]。このような仮定は、物理学における平均場近似（*mean field approximation*）に対応します。

この $q(\pmb{Z})$ を式 $(3)$ に代入し、各因子についてELBOの最適化を行うのが以降の流れとなります。詳細は省きますが、任意の因子 $q_j(\pmb{Z}_j)\eqqcolon q_j$ に対するELBOの依存項を抜き出すと、 $q_j$ の最適解 $q^*_j$ は

$$\log \tilde{p}(\pmb{X},\pmb{Z}_j)=\mathbb{E}_{i\neq j}[\log p(\pmb{X,Z})]+const.$$

で定義される $\tilde{q}(\pmb{X},\pmb{Z}_j)$ に等しい、すなわち

$$\begin{aligned}\log q_j^*(\pmb{Z}_j)=&\mathbb{E}_{i\neq j}[\log p(\pmb{X,Z})]+const.\\
\Big(=&\int \log p(\pmb{X,Z})\prod_{i\neq j} q_i \mathrm{d} \pmb{Z_i}+const.\Big)
\end{aligned}$$

を満たすことがわかります[1]。この式から、因子 $q_j$ の最適解の対数は、観測データと隠れ変数の同時分布の対数を考え、 $i\neq j$ である他の因子 $\{q_i\}$ すべてについての期待値を取ったものに等しいことがわかります。

## 参考文献

[1] パターン認識と機械学習 -ベイズ推論による統計的予測- 下, C.M.ビショップ, Springer Japan, 第2刷（Chapter 10）

[2] [【変分ベイズ】変分推論やELBOを理解する](https://disassemble-channel.com/gmm-variational-inference/)

[3] [変分法をごまかさずに変分ベイズの説明をする](https://statmodeling.hatenablog.com/entry/variational-bayesian-inference-1)

[4] [The ELBO in Variationall Inference](https://gregorygundersen.com/blog/2021/04/16/variational-inference/)