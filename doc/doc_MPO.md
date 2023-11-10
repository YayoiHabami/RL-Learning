# Maximum a Posteriori Policy Optimization (MPO)

## 目次

- [目次](#目次)
- [概要](#概要)
- [はじめに](#はじめに)
- [理論](#理論)
  - [環境設定](#環境設定)
  - [Control as inferenceとは（概要）](#control-as-inferenceとは概要)
    - [最適性確率変数](#最適性確率変数)
  - [最適制御確率の下界の導出](#最適制御確率の下界の導出)
  - [最適制御確率の下界の最大化](#最適制御確率の下界の最大化)
    - [E-step](#e-step)
- [参考文献](#参考文献)
- [Appendix](#appendix)
  - [用語の解説](#用語の解説)
    - [補足：maximum a posteriori（MAP; 最大事後確率）](#補足maximum-a-posteriorimap-最大事後確率)
    - [補足：objective (目的関数)](#補足objective-目的関数)
    - [補足：KLダイバージェンス](#補足klダイバージェンス)
    - [補足：期待値](#補足期待値)
    - [補足：イエンゼンの不等式](#補足イエンゼンの不等式)
    - [補足：ラグランジュ未定乗数法](#補足ラグランジュ未定乗数法)
  - [数式の説明](#数式の説明)
    - [補足：なぜlog？](#補足なぜlog)
    - [補足：argmaxの意味](#補足argmaxの意味)
    - [補足：イェンセンの不等式を利用した式変形](#補足イェンセンの不等式を利用した式変形)
    - [補足：なぜ報酬和と指数比例？](#補足なぜ報酬和と指数比例)
  - [参考文献（Appendix）](#参考文献appendix)


## 概要

**Maximum a Posteriori Policy Optimization**(MPO, [1])とは、**ICLR2018**で発表された、方策が最適である確率の下界を最大化する強化学習手法です。

## はじめに

強化学習で連続値行動環境をコントロールする方策（Policy）を訓練するためには**方策勾配法**が広く使われています。一方で方策勾配法は基本的にオンポリシーであるためにサンプルを使い捨てること（**劣悪なサンプル効率**）、また勾配の分散が大きいこと（**不安定なネットワーク更新**）が問題となります。この改善策として**Trust Region Policy Optimization (2015)**（TRPO, [3]）があり、信頼領域法の導入により安定性は大きく向上しました。ただし依然としてオンポリシーゆえのサンプル効率の悪さは課題として残っています。

MPOは方策勾配法ではなく**Control as Inference**、すなわち確率推論のフレーうワークで制御ポリシーを訓練する手法です。MPOはオフポリシーであるため**サンプル効率が良好**であり、また信頼領域法を用いるため**ロバストな更新**を実現しています。

## 理論

### 環境設定

割引強化報酬（RL）問題の最適な方策（ポリシー） $\pi$ を見つける問題を、**マルコフ決定過程**（MDP）によってモデル化します。MDPは、（連続）状態 $s$ 、アクション $a$ 、遷移確率 $p(s_{t+1}|s_t,a_t)$ 、報酬関数 $r(s,a)\in\mathbb{R}$ 、および割引率 $\gamma\in[0,1)$ から構成されます。また、ポリシー $\pi(a|s,\pmb\theta)$ （パラメタは $\pmb\theta$ ）は、任意の状態に対してアクション選択のための確率分布を指定すると仮定し、遷移確率とともに定常分布 $\mu_\pi(s)$ を作ります。

> 遷移確率 $p(s_{t+1}|s_t,a_t)$ は、状態 $s_t$ から $s_{t+1}$ への行動 $a_t$ による遷移確率を指定します。

> 定常分布とは、任意の状態 $s$ から開始した場合の、無限時間経過後の状態分布のこと[5]

これらのパラメタを用いて、ポリシー $\pi$ に従った際の軌跡（トラジェクトリ） $\tau_\pi=\{(s_0,a_0)...(s_T,a_T)\}$ が定義されます。すなわち、トラジェクトリは $p_\pi(\tau)=p(s_0)\prod_{t>0}p(s_{t+1}|s_t,a_t)\pi(a_t|s_t)$ を用いて、 $\tau_\pi\sim p_\pi(\tau)$ と定義されます。期待収益は $\mathbb E_{\tau_\pi}\left[\sum_{t=0}^\infty\gamma^tr(s_t,a_t)\right]$ となります。ここで、以下では $r_t=r(s_t,a_t)$ として扱います。

### Control as inferenceとは（概要）

Control as inferenceとは、制御問題を確率的な推論(*inference*)問題として捉えるフレームワークです[4]。このフレームワークは強化学習や最適制御の一般化とみることができ、エージェントは観察と報酬から最適な行動を推論しようとします。エージェントは期待報酬に加えて行動のエントロピーを最大化するため、報酬と不確実性の両方がある行動を好み、探索とロバスト性を確保できます。

はじめにエージェントの行動を表す潜在変数を導入し、状態、行動、報酬に関する同時分布を定義します。次にエージェントは状態と報酬が与えられたときの行動に関する事後分布を求めます。結果として得られる目的は標準的な強化学習の目的と似ていますが、探索を促すエントロピーの項があるという点において異なります。

Control as inferenceは部分観測性などを含めて問題を拡張することを可能にしたり、ナチュラルな探索を可能にする（Maximum Entropy Model）などの利点を持ちます[4]。

#### 最適性確率変数

このフレームワークにいて最も重要なコンセプトは**最適性確率変数** $\mathcal O$ です。直感的には $\mathcal{O}$ は、（行動を選択することにより）最大の報酬を獲得できるイベントとして解釈可能です。もしくは、強化学習タスクを成功させるイベントとしても捉えられるかもしれません[1]。

この手法では行動の最適性を確率分布で表現します。例えばあるトラジェクトリ $\tau$ が与えられたとき、それが最適トラジェクトリである確率は $p(\mathcal O=1|\tau)$、そうでない確率は $p(\mathcal O=0|\tau)$ と表現されます。同様に、状態 $s_t$においてアクション $a_t$ が最適行動である確率は $p(\mathcal O_t=1|s_t, a_t)$ となります。

![](imgs/最適制御確率変数_グラフィカルモデル.png)

最適性確率変数導入の最大のメリットは、MDPにおける「最適な制御」をグラフィカルに表現できるようになることです。これにより行動の最適性を明示的に確率分布で表現でき、環境の不確実性を自然に扱えるようになります。また、確率推論のさまざまなツールを利用可能になるのも大きなメリットであり、実際にMPOではEMアルゴリズムやELBOなどを活用しています[2]。

### 最適制御確率の下界の導出

最適性確率変数を $\mathcal O$ とします。MPOの目的は方策 $\pi$で行動決定を実行した際に、それが最適制御である確率 $p_\pi(\mathcal{O}=1)$ を最大化することです。最適制御である確率とは、具体的には<sup>[補足1](#補足なぜlog)</sup>

$$\log{p_\pi(\mathcal O=1)}=\log{\int {p_\pi(\tau)p(\mathcal O=1|\tau)d\tau}}$$

すなわちトラジェクトリ $\tau$ が方策 $\pi$ に従って生成されるときに $\tau$ が最適である確率の[**期待値**](#補足期待値)となります。

次にイエンゼンの不等式<sup>[補足](#補足イエンゼンの不等式)</sup>から、任意の確率分布 $q(\tau)$ について、

$$\begin{align}
&\log{\int {p_\pi(\tau)p(\mathcal O=1|\tau)d\tau}}\\
&\geq \int{q(\tau)\left(\log{p(\mathcal O=1|\tau)+\log \frac{p_\pi(\tau)}{q(\tau)}}\right)d\tau}\\
&=\int{q(\tau)\log{p(\mathcal O=1|\tau)}d\tau}+\int{q(\tau)\log \frac{p_\pi(\tau)}{q(\tau)}d\tau}
\end{align}$$

が成立します<sup>[式変形の補足](#補足イェンセンの不等式を利用した式変形)</sup>[1, Equ.1]。このうち、第一項はトラジェクトリ $\tau$ が $q$ に従って生成されるときに、 $\tau$ が最適である確率の期待値を意味します。

ここで、尤度関数 $p(\mathcal{O}=1|\tau) \propto \exp{\sum_t \frac{r_t}{\alpha}}$ を、すなわちトラジェクトリが最適である確率 $p$ が $\tau$ の報酬和と指数比例することを想定する<sup>[補足](#補足なぜ報酬和と指数比例)</sup>と、

$$\begin{align}
&\int{q(\tau)\log{p(\mathcal O=1|\tau)}d\tau}+\int{q(\tau)\log \frac{p_\pi(\tau)}{q(\tau)}d\tau} \tag{4}\\
&=\mathbb E_{q} \left[\log{\exp {\sum_t \frac{r_t}{\alpha}}} \right] + \int {q(\tau) \log {\frac{p_\pi(\tau)}{q(\tau)}}}d\tau\\
&=\mathbb E_{q} \left[\sum_t \frac{r_t}{\alpha} \right] + \int {q(\tau) \log {\frac{p_\pi(\tau)}{q(\tau)}}}d\tau
\end{align}$$

と変形できます。ここで $\alpha$ は温度パラメータであり、報酬 $r_t$ をスケーリングする役割を持ちます。このとき $p, q$ はいずれも方策分布を表します。上式から、報酬和が高いほど最適トラジェクトリである確率が指数的に高まることがわかります。上式の第二項はKLダイバージェンス<sup>[補足](#補足klダイバージェンス)</sup> $KL(q(\tau)||p_\pi(\tau|\pmb\theta))$ と同様の形式( $-KL(\cdot)$ )ですから、

$$\begin{align}
\log{p_\pi(\mathcal O=1)}&=\mathbb E_{q}\left[\sum_t \frac{r_t}{\alpha}\right]-\mathrm{KL}(q(\tau)||p_\pi(\tau|\pmb\theta))\\
&=\mathcal J(q,\pi)\end{align}$$

とできます（ $\mathcal J$ は目的関数）[1, Equ.2]。尤度関数を介して確率モデルを構築することにより、推論問題として扱うことができるようになりました。ここで、直感的には、 $\mathcal O$ **は、アクションを選択することにより最大の報酬を獲得するイベント**として解釈できます。

次に、 $p_\pi$ と[同じ方法で因数分解](#環境設定)する変分分布 $q(\tau)=p(s_0)\prod_{t>0}p(s_{t+1}|s_t,a_t)q(a_t|s_t)$ を仮定し変形すると[1, Equ.3]、

$$\mathcal J(q,\pi)=\mathbb E_q\left[\sum_{t=0}^\infty\gamma^t\big[r_t-\alpha\mathrm{KL}\left(q(a|s_t)||\pi(a|s_t,\pmb\theta)\right)\big]\right]+\log p(\pmb\theta)$$

ここで、第一項は、割引報酬の期待合計を最大化しながら、KLダイバージェンスに関して $q(a|s_t)$ と $\pi(a|s_t,\pmb\theta)$ が近づくように操作することを示します（KLダイバージェンスは2分布が近しいほど0に近づくため）。 $\log p(\pmb\theta)$ 項はポリシーパラメタに対する事前変数であり、最大事後推定問題によって動機付けられます。

>$$\mathcal J(q,\pi)=\mathbb E_{q}\left[\sum_t \frac{r_t}{\alpha}\right]-\mathrm{KL}(q(\tau)||p_\pi(\tau|\pmb\theta))$$
>
> 温度パラメタ $\alpha>0$ を掛けても $\arg\max_q \mathcal J$ の結果は変わらないため、これを掛けた関数を新たな $\mathcal J(q,\pi)$ とします。
> 
> 期待報酬は $\mathbb E_{q}\left[\sum_{t=0}^\infty\gamma^tr_t\right]$ であるので、
>
> $$\begin{align}\mathcal J(q,\pi)=\frac{1}{\alpha}\mathbb E_{q}\left[\sum_{t=0}^\infty\gamma^tr_t\right]-\mathbb E_q\Big[\log q(\tau)-\log p_\pi(\tau|\pmb\theta)\Big]\end{align}$$
>
> と変形できます。また、 $p(\tau), q(\tau|\pmb\theta)$ の定義から、 $\log p_\pi(\tau|\pmb\theta)=\log p(s_0)+\sum_{t=0}^\infty \big(\log p(s_{t+1}|s_t)+\log \pi(a_t|s_t,\pmb\theta)\big)$ および $\log q(\tau)=\log p(s_0)+\sum_{t=0}^\infty \big(\log p(s_{t+1}|s_t)+\log q(a_t|s_t)\big)$ がわかります。これを上式に代入すれば
>
> $$\mathcal J(q,\pi)=\frac{1}{\alpha}\mathbb E_{q}\left[\sum_{t=0}^\infty\gamma^tr_t\right]-\mathbb E_q\left[\sum_{t=0}^\infty \big(\log q(a_t|s_t)-\log \pi(a_t|s_t,\pmb\theta)\big)\right]$$
>
> トラジェクトリ $\tau$ は扱いが困難です。ここでトラジェクトリのKLダイバージェンスは各行動ステップのダイバージェンスに分解できると**想定すれば**（若干ここに無理があるようですが）[2]、上式は
> 
> $$\mathcal J(q,\pi)=\frac{1}{\alpha}\mathbb E_q\left[\sum_{t=0}^\infty\gamma^tr_t\right]-\mathbb E_q\left[\sum_{t=0}^\infty\gamma^t \mathrm{KL}\left(q(a|s_t)||\pi(a|s_t,\pmb\theta)\right)\right]$$
> $$\mathcal J(q,\pi)=\mathbb E_q\left[\sum_{t=0}^\infty\gamma^t\big[r_t-\alpha\mathrm{KL}\left(q(a|s_t)||\pi(a|s_t,\pmb\theta)\right)\big]\right]+\log p(\pmb\theta)$$


$$\int{q(\tau)\log \frac{p_\pi(\tau)}{q(\tau)}d\tau}=\mathbb E_{\tau\sim q}\left[-KL(q(\cdot|s_t)||\pi(\cdot|s_t, \pmb\theta))\right] + \log p(\pmb\theta)$$

となります。上式および式 $(4)$ から、方策 $\pi$ で行動決定を実行したときにそれが**最適制御である確率** $p_\pi(\mathcal O=1)$ **の下界** $\mathcal J$ を

$$\mathcal J(q,\pmb\theta)=\mathbb E_{\tau\sim q}\left[\sum_t \frac{r_t}{\alpha}-KL(q(\cdot|\pmb s_t)||\pi(\cdot|\pmb s_t, \pmb\theta))\right] + \log p(\pmb\theta) \tag{10}$$

として規定でき、目的関数<sup>[補足](#補足objective-目的関数)</sup> $\mathcal J$ を任意の確率分布（今回は方策分布） $q$ と方策パラメータ $\pmb\theta$ で表現することができました。

> ここまでで、**方策** $\pi$ **で行動決定を実行した際に、それが最適制御である確率** $p_\pi(\mathcal{O}=1)$ **は** $\mathcal J$ **以上である**こと、すなわち
>
> $$\begin{align}p_\pi(\mathcal{O}=1)&\geq\mathcal J(q,\pmb\theta)\\&=\mathbb E_{\tau\sim q}\left[\sum_t \frac{r_t}{\alpha}-KL(q(\cdot|\pmb s_t)||\pi(\cdot|\pmb s_t, \pmb\theta))\right] + \log p(\pmb\theta)\end{align}$$
>
> となることを導出しました。

### 最適制御確率の下界の最大化

この節では**最適制御である確率** $p_\pi(\mathcal O=1)$ **の下界** $\mathcal J$ を最大化するための手法である、**E-step**および**M-step**を説明します。

|ステップ|概要|
|:-:|-|
|E-step| $\pmb\theta$ を定数とみなして $q$ について $\mathcal{J}$ を最大化する|
|M-step| $q$ を定数とみなして $\pmb\theta$ について $\mathcal{J}$ を最大化する|

大まかに各ステップは上記の動作を行い、このステップを繰り返すことで $\mathcal{J}$ を最大化します。
#### E-step

Eステップでは方策パラメータ $\pmb\theta$ を定数とみなし、下界 $\mathcal{J}$ を最大化するような方策分布 $q$ を算出します。前節で導出した式 $(10)$ を用いれば<sup>[argmaxの補足](#補足argmaxの意味)</sup>

$$\begin{align}
&\arg\max_q \mathcal{J}(q,\pmb\theta)\\
=&\arg\max_q {\mathbb E_{\tau\sim q}\left[\sum_t \frac{r_t}{\alpha}-KL(q(\cdot|\pmb s_t)||\pi(\cdot|\pmb s_t, \pmb\theta))\right] + \log p(\pmb\theta)}
\end{align}$$

となります。ここで、このステップでは $\log p(\pmb\theta)$ は定数であり無視できます。また、温度パラメタ $\alpha$ は非負定数なので、

$$=\arg\max_q {\mathbb E_{\tau\sim q}\left[\sum_t r_t-\alpha KL(q(\cdot|\pmb s_t)||\pi(\cdot|\pmb s_t, \pmb\theta))\right]}$$

さらに、トラジェクトリ期待値 $\mathbb E_{\tau\sim q}$ を状態行動ステップ期待値 $\mathbb E_{\mu(s)}$ に書き直すと

$$\arg\max_q \mathbb E_{\mu(\pmb s)}\left[\mathbb E_{\pmb a\sim q}\right]$$

## 参考文献

[1] [Maximum a Posteriori Policy Optimization](https://openreview.net/forum?id=S1ANxQW0b)

[2] [強化学習 as Inference： Maximum a Posteriori Policy Optimizationの実装](https://horomary.hatenablog.com/entry/2022/07/21/192741)

[3] [Trust Region Policy Optimization (2015)](https://arxiv.org/abs/1502.05477)

[4] [Control as Inference, mendy, Speaker Deck](https://speakerdeck.com/shunichi09/sergey-levine-lecture-remake-14th-control-as-inference?slide=5)

[5] [（9） マルコフ過程（状態遷移行列・極限分布）](http://sysplan.nams.kyushu-u.ac.jp/gen/edu/SystemsDesignEngineering/2019/09.pdf)

## Appendix

### 用語の解説

#### 補足：maximum a posteriori（MAP; 最大事後確率）

**maximum a posteriori**（MAP; 最大事後確率）という語は、一般に最大事後確率推定（MAP推定）というベイズ統計学の手法名において出てくる語です。

> 最大事後確率推定[A16]
>
> 実測データに基づいて未知の量の点推定を行う手法である。ロナルド・フィッシャーの最尤推定 (MLE) に密接に関連するが、推定したい量の事前分布を利用して最適化問題を解き確率が最大の結果を得る。したがってMAP推定は、最尤推定に正則化をつけた物と見ることもできる。

#### 補足：objective (目的関数)

*objective*とは強化学習の**目的関数**のことであり、エージェントが最大化しようとする長期的な報酬の期待値です。一般に、以下のような式で表されます。
 
$$J(\pi) = \mathbb E_\pi [\sum_{t=0}^{\infty} \gamma^t r(s_t,a_t)]$$

ここで、 $\pi$ はエージェントのポリシー、 $\gamma$ は割引率、 $r(s_t,a_t)$ は状態 $s_t$ で 行動 $a_t$ を取った時にエージェントが得られる報酬です。

すなわち、目的関数 $J(\pi)$ は、エージェントがポリシー $\pi$ に従って行動したときに、**未来に得られる報酬の割引和の期待値**を表しています。

#### 補足：KLダイバージェンス

*Kullback-Leivler divergence*（KLダイバージェンス）とは、2つの確率分布がどの程度似ているかを表す尺度です。連続確率変数に対する定義は下記の通りです[A2]。

$$\begin{align}\mathrm{KL}(p||q)&=\int_{-\infty}^\infty {p(x)\log{\dfrac{p(x)}{q(x)}}}dx\\ \Bigg( &=\mathbb E_q\left[\log\frac{p(x)}{q(x)}\right]\Bigg)\end{align}$$
 
重要な特性として、**同じ確率分布では0になる**点（ $KL(p||p)=0$ ）、**似ていないほど大きな値を取る**点が挙げられます。ほかの基本的性質を以下に示します[A19]。

|||
|-|-|
|非負性| $0\leq \mathrm{KL}(p\|\|q)\leq \infty$ |
|非退化性| $\mathrm{KL}(p\|\|q)=0 \iff p=q$ |
|非対称性| $\mathrm{KL}(p\|\|q)\neq\mathrm{KL}(q\|\|p)$$ |

別名として**KL情報量**やKL距離などを持ちます。KL「距離」とこそ言われますが、距離の公理を満たさないため厳密には距離ではありません。

> 距離の公理： $X$ 上の距離（関数） $d$ は $d\colon X\times X\to\mathbb{R}$ で定義される関数であり、非退化性、対称性、三角不等式、非負性の４条件を満たします[A18]。また、非負性は非負性を除いた３条件から導かれるので（ $d(x,y)+d(y,x)\geq d(x,x)=0$ ）、弱めて３条件を公理とすることもあります[A17]。
> |||
> |-|-|
> |非退化性| $\forall x,y\in X\colon d(x,y)=0\iff x=y$ |
> |対称性| $\forall x,y\in X\colon d(x,y)=d(y,x)$ |
> |三角不等式| $\forall x,y,z\in X\colon d(x,y)+d(y,z)\geq d(x,z)$ |
> |非負性| $\forall x,y \in X\colon d(x,y)\geq 0$ |
> 
> KLダイバージェンスはこのうち非退化性（および非負性）しかを満たさないため距離として定義されません。このように非退化性と非負性のみを満たすような量を *divergence* と呼ぶようです[A19]。

#### 補足：期待値

**期待値**は確率変数を含む関数の実現値に確率の重みをつけた加重平均です。確率変数 $X\sim P_X$ を引数にとる関数 $g(X)$ の $X$ に対する期待値 $\mathbb E_{P_X}\left[g(X)\right]$ は、例えば次のように定義されます[A3]：

$$\mathbb E_{P_X}\left[g(X)\right]=\sum_xP_X(x)g(x)=\int P_X(x)g(x)\mathrm dx$$

前者は可能な結果の可算集合を持つ確率変数の場合の、後者は密度を持つ確率変数の場合の定義式です。$X$ は**確率変数**であり、結果ごとに値が決定されます（集合ではない）。 $P_X(x)$ は**確率質量変数**といい、 $X$ が特定の値 $x$ を取った場合にその値をとる確率を与えます。 $g(X)$ は確率変数 $x$ の関数です。

具体例を考えます。2枚のコイン投げで表が出た枚数を数える場合、取りうる値の集合は $\{0,1,2\}$ になります。 $X$ の値はこの集合に含まれますが、 $X$ 自体はこの集合ではありません。 次に、コインを投げたときに合計値が $0,1,2$ になる確率はそれぞれ $0.25,0.5,0.25$ ですが、これはそれぞれ  $P_X(0) = 0.25$, $P_X(1) = 0.5$, $P_X(2) = 0.25$ と表されます。最後に、このコイン投げでプレイヤーはスコアを得るとしましょう。仮に枚数の２倍をスコアとするならば $g(X)=2X$ ですし、２枚以外の場合は０点とするならば $g(X) = \begin{cases} 1 & (X=2) \\ 0 & (X \neq 0) \end{cases}$ などのように定義することができます。後者を採用した場合、スコアの期待値は 

$$\mathbb E_{P_X}\left[g(X)\right]=\sum_{x=0,1,2}P_X(x)g(x)=0.25\space[点]$$

と計算できます。

期待値 $\mathbb E$ は総和や（ルベーグ）積分により定義されるため、総和や積分の持つ性質をすべて持っています。

> **ルベーグ積分**は、高校までで学習した求積区分法の延長線上の積分（リーマン積分）とは異なり、非連続値であっても計算可能な積分です。
> 
> 一般にルベーグ積分はリーマン積分の持つ性質をすべて持つわけではありませんが、有界閉区間上の有界な関数に対しては、リーマン積分とルベーグ積分は等しいようです（広義リーマン積分可能であってもルベーグ積分可能でない関数が存在する）[A6]。

期待値 "E" の表記方法には $\mathrm E, E, \mathbb E$ などさまざまな表記があり、括弧の表記にも同様に種類があるようですが[A3]、ここでは $\mathbb E[g(X)]$ を使用します。

#### 補足：イエンゼンの不等式

$f(x)$ が凸関数のとき、
 
- 任意の $x_1,...,x_n \in \mathbb R$ と
- $\lambda_i\ge0,\sum_{i=1}^n\lambda_i=1$ を満たす任意の $\lambda_1, ..., \lambda_n$

に対して次が成立し、これを**イエンゼンの不等式**（凸不等式）と呼びます。

$$\displaystyle\sum_{i=1}^n\lambda_if(x_i)\ge f\left(\sum_{i=1}^n\lambda_ix_i\right)$$

この不等式は、凸関数上<sup>?</sup>の任意の $n$ 点が作る凸包上の点（左辺）が、曲線上の点（右辺）よりも上に存在することを示しています。

<img src="imgs/イエンゼンの不等式模式図.png" width=300>

上図は凸包をの網掛け領域で表し、左辺 $\sum_{i=1}^3\lambda_if(x_i)$ に基づく赤い点が、右辺 $f(\sum_{i=1}^3\lambda_ix_i)$ に基づく青い点よりも上に存在していることを示した図です。

> **凸関数**とは、 $f(x)=x^2$ のような、曲面上の任意の2点を結ぶ線分が関数の上に存在するような関数です。

連続値の場合にも同様の議論が可能であり、

- 実数 $x$ と
- 実数上の可積分関数 $p(x)(>0)\space s.t.\space\int p(x)\mathrm dx=1$

に対して次が成立します<sup>本当に?</sup>（ $y(x)$ は実数上の可積分関数）。

$$\displaystyle\int f(y(x))p(x)dx\ge f\left(\int y(x)p(x)dx\right)$$

#### 補足：ラグランジュ未定乗数法

**ラグランジュ未定乗数法**とは、例えばなんらかの制約 $g(x,y)=0$ のもとで $f(x,y)$ を最大化するというような、**制約付きの最大化/最小化問題を解く手法**です[A7]。

制約がない場合に二変数関数 $f(x,y)$ の最大化は、 $\frac{\partial f}{\partial x}=\frac{\partial f}{\partial y}=0$ を満たす点、または区間の端の点のうち最大のものを探すことで可能です。これに対して制約 $g(x,y)=0$ を与えた場合、ラグランジュ乗数 $\lambda$ を用いて、ラグランジュ関数

$$L(x,y,\lambda)=f(x,y)-\lambda g(x,y)$$

を定義でき、 $(x,y)=(\alpha,\beta)$ が極値を与えるならば、 $(\alpha,\beta)$ は

$$\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}=\frac{\partial L}{\partial\lambda}=0$$

または $\frac{\partial g}{\partial x}=\frac{\partial g}{\partial y}=0$ の解となります。

また、不等式制約を与える場合であっても、**Karush-Kuhn-Tucker条件**（KKT条件）を用いることで適用が可能です[A8]。不等式制約 $g_i(x,y)\leq0\space(i=1,...,m)$ および等式制約 $h_j(x)=0\space(j=1,...,p)$ を持つ関数 $f(x)$ を考えます。このとき、ラグランジュ関数はKKT乗数 $\pmb\lambda, \pmb\mu$ を用いて以下の式で表されます[A11]。

$$L(x,\pmb\lambda,\pmb\mu)=f(x)+\sum_{i=1}^m\lambda_ig_i(x)+\sum_{j=1}^p\mu_jh_j(x)$$

ここで、 $\tilde x$ が制約付き最適化問題の局所最適解であるためには、次の４条件が必要です[A9][A12]。

|条件|英名|数式|
|:-:|:-:|-|
|停留性|*stationarity*|$\plusmn\partial f(x) + \sum_{i=1}^m \lambda_i \partial g_i(x) + \sum_{j=1}^p \mu_j \partial h_j(x) \ni \pmb0$ <br>（ $\plusmn$ : $f(x)$ の最大化問題では $-$ , 最小化問題では $+$ ）|
|同上||関数 $f(x)$ が微分可能な凸関数の場合は次も同等<br>　　 $\plusmn\nabla f(x) + \sum_{i=1}^m \lambda_i \nabla g_i(x) + \sum_{j=1}^p \mu_j \nabla h_j(x) = 0$ <br>　　（ $\plusmn$ : $f(x)$ の最大化問題では $-$ , 最小化問題では $+$ ）||
|実行可能性|*primal feasibility*| $g_i(x) \leq 0$（すべての$i = 1, \dots, m$に対して）<br> $h_j(x) = 0$（すべての$j = 1, \dots, p$に対して）|
|双対実行可能性|*dual feasibility*| $\lambda_i \geq 0$（すべての$i = 1, \dots, m$に対して）|
|補完スラック|*complementary slackness*|$\lambda_i g_i(x) = 0$（すべての$i = 1, \dots, m$に対して）|

### 数式の説明

調べていてわからなかった数式の表現を以下に記載します。

#### 補足：なぜlog？

- $\log{p_\pi(\mathcal{O}=1)}$

方策 $\pi$ の下で最適なアクションを撮る確率は $p_\pi(\mathcal{O}=1)$で表されます。これに対して $\log$ を使用することで、勾配の計算を簡素化し、数値的な問題を回避させています[A1]。より具体的には、勾配と除算を用いた（数学的に同一な）式に比べて、数値的に安定する傾向があるようです。

#### 補足：argmaxの意味

- $\arg\max_q \mathcal J(q,\pmb\theta)$

関数 $\mathcal J(q,\pmb\theta)$ を最大化する $q$ の値を意味します。言い換えれば、これは $\mathcal J$ が目的関数、 $q$ がモデルパラメータである最適化問題の最適解です。

$\argmax$ 自体は特定の関数や手法を示すものではなく、実際に $\mathcal J(q,\pmb\theta)$ の最適化（最大化）問題を解く際に具体的な手法を指定します（今回はKKT条件有のラグランジュの未定乗数法）。

#### 補足：イェンセンの不等式を利用した式変形

- $\log{\int {p_\pi(\tau)p(\mathcal O=1|\tau)d\tau}}\geq \int{q(\tau)\left(\log{p(\mathcal O=1|\tau)+\log \frac{p_\pi(\tau)}{q(\tau)}}\right)d\tau}$

上記の式変形をもう少し詳しく展開します。はじめに、軌跡 $\tau$ に対して任意の確率分布 $q(\tau)$ を導入します。このとき、 $p_\pi(\tau)$ と $p(\mathcal O=1|\tau)$ が $0$ でない $\tau$ に対して、確率分布は $q(\tau)\neq0$ を満たす必要があります。 $q(\tau)$ を用いて、

$$\log{\int {p_\pi(\tau)p(\mathcal O=1|\tau)d\tau}}=\log{\int {\frac{p_\pi(\tau)p(\mathcal O=1|\tau)}{q(\tau)}q(\tau)d\tau}}$$

がわかります。確率分布の性質から $\int q(\tau)d\tau=1$ が恒等的に成立し、イェンセンの不等式を用いることで

$$\log{\int {\frac{p_\pi(\tau)p(\mathcal O=1|\tau)}{q(\tau)}q(\tau)d\tau}}\geq \int{q(\tau)\log{\frac{p_\pi(\tau)p(\mathcal O=1|\tau)}{q(\tau)}}d\tau}$$

を導くことができました。最後に $\log$ の性質を用いることで、冒頭の式を導出することができます。

#### 補足：なぜ報酬和と指数比例？

- $p(\mathcal{O}=1|\tau) \propto \exp{\sum_t \frac{r_t}{\alpha}}$

上記の式は、軌跡 $\tau$ が最適である確率 $p(\mathcal O=1|\tau)$ が 報酬和 $\sum_t r_t$ に指数比例するということを意味しています。このように定義したのは、**報酬の和が大きいほど、その軌道が最適である可能性が高くなる**ことを数式で表す上で、おそらく指数関数を利用するのが都合がよかったのだろうと考えられます。（指数関数を用いることで、報酬和の大きい領域での差に敏感になるように思われる。）

### 参考文献（Appendix）

[A1] [What is log probability in policy gradient (reinforcement learning)?](https://www.quora.com/What-is-log-probability-in-policy-gradient-reinforcement-learning)

[A2] [正規分布間のKLダイバージェンス](https://qiita.com/ceptree/items/9a473b5163d5655420e8)

[A3] [Expected value, Wikipedia](https://en.wikipedia.org/wiki/Expected_value)

[A6] [ルベーグ積分はリーマン積分の拡張｜証明と計算の例題を解説]（https://math-note.xyz/analysis/measure-theory/basics-of-lebesgue-integral/lebesgue-integral-and-riemann-integral/）

[A7] [ラグランジュの未定乗数法と例題](https://manabitimes.jp/math/879)

[A8] [カルーシュ・クーン・タッカー条件](https://ja.wikipedia.org/wiki/%E3%82%AB%E3%83%AB%E3%83%BC%E3%82%B7%E3%83%A5%E3%83%BB%E3%82%AF%E3%83%BC%E3%83%B3%E3%83%BB%E3%82%BF%E3%83%83%E3%82%AB%E3%83%BC%E6%9D%A1%E4%BB%B6#:~:text=%E3%82%AB%E3%83%AB%E3%83%BC%E3%82%B7%E3%83%A5%E3%83%BB%E3%82%AF%E3%83%BC%E3%83%B3%E3%83%BB%E3%82%BF%E3%83%83%E3%82%AB%E3%83%BC%E6%9D%A1%E4%BB%B6%EF%BC%88,%E3%82%82%E6%89%B1%E3%81%86%E3%81%93%E3%81%A8%E3%81%8C%E3%81%A7%E3%81%8D%E3%82%8B%E3%80%82)

[A9] [ML day 8: ラグランジュの未定乗数法と制約付き最適化問題](https://blog.takanabe.tokyo/2023/03/df12434a-322b-4fc6-9e6c-2f3703581b76/)

[A11] [Karush–Kuhn–Tucker conditions](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions)

[A12] [Lecture 12: KKT Conditions](https://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/12-kkt-scribed.pdf)

[A16] [最大事後確率](https://ja.wikipedia.org/wiki/%E6%9C%80%E5%A4%A7%E4%BA%8B%E5%BE%8C%E7%A2%BA%E7%8E%87#:~:text=%E6%9C%80%E5%A4%A7%E4%BA%8B%E5%BE%8C%E7%A2%BA%E7%8E%87%EF%BC%88%E3%81%95%E3%81%84%E3%81%A0,%E6%9C%80%E5%A4%A7%E3%81%AE%E7%B5%90%E6%9E%9C%E3%82%92%E5%BE%97%E3%82%8B%E3%80%82)

[A17] [距離空間](https://ja.wikipedia.org/wiki/%E8%B7%9D%E9%9B%A2%E7%A9%BA%E9%96%93)

[A18] [Metric space](https://en.wikipedia.org/wiki/Metric_space)

[A19] [Kullback-Leibler Divergenceについてまとめる](https://yul.hatenablog.com/entry/2019/01/07/152738)