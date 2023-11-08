# 最適性の理論

強化学習の論文を読んでいると、式変形において**イエンゼンの不等式**（*Jensen's inequality*）がよく出てきます。この式自体は平易ですが、イメージも合わせて理解するとより概観が掴みやすくなります。はじめに非線形計画法の種々の理論の基礎となる凸集合と凸関数について学習したのち、この不等式の説明を行います。

また、強化学習においては（ハード）制約をつけたうえで最適化問題を解くことが往々にしてあります。このとき、これを解析的に解くための１手法が**ラグランジュ未定乗数法**（*Lagrange's method of indeterminate coefficients*）です。以下では、与えられた制約が等式制約である場合に加えて、**不等式制約**が与えられた場合に**KKT条件**（*Karush-Kuhn-Tucker conditions*）を用いてどのように解くのかを説明します。

## イエンゼンの不等式

### 凸集合と凸関数

**凸集合**（*convex set*）は、集合に属する任意の２点を結ぶ線分上のすべての点が自身に属するような集合です。すなわち、凸集合 $X$ は

$$\theta \pmb{x}_1+(1-\theta)\pmb{x}_2\in X,\space\forall \pmb{x}_1,\pmb{x}_2\in X, \space\forall \theta\in [0,1]$$

を満たします[1]。

<img src="imgs/凸集合と非凸集合.png" width=600>

図１．凸集合と非凸集合

**凸関数**（*convex function*）とは、 $f(x)=x^2$ のような、曲面上の任意の2点を結ぶ線分が関数の上に存在するような関数です。いわゆる「下に凸な関数」と言えると思います。より正確には、 凸集合 $X\subset \R^n$ 上で定義される関数 $f\colon X\to \R^1$ に対して、 

$$\begin{aligned}&\theta f(\pmb{x}_1)+(1-\theta)f(\pmb{x}_2)\geq f\big(\theta \pmb{x}_1+(1-\theta)\pmb{x}_2\big)\\
&s.t.\space\forall \pmb{x}_1,\pmb{x}_2\in X\space\forall\theta\in [0,1]
\end{aligned}$$

が成り立つような関数です（ $n=2$ のときのイエンゼンの不等式）[2]。また、$-f$ が凸関数である場合にこれを**凹関数**（*concave function*）といいます。

<img src="imgs/凸関数と凹関数.png" width=600>

図２．凸関数と凹関数

### イエンゼンの不等式

$f(x)$ が凸関数のとき、
 
- 任意の $x_1,...,x_n \in \mathbb R$ と
- $\lambda_i\ge0,\sum_{i=1}^n\lambda_i=1$ を満たす任意の $\lambda_1, ..., \lambda_n$

に対して次が成立し、これを**イエンゼンの不等式**（凸不等式; *Jensen's inequality*）と呼びます[3][4]。

$$\displaystyle\sum_{i=1}^n\lambda_if(x_i)\ge f\left(\sum_{i=1}^n\lambda_ix_i\right)$$

この不等式は、凸関数上<sup>?</sup>の任意の $n$ 点が作る凸包上の点（左辺）が、曲線上の点（右辺）よりも上に存在することを示しています。

<img src="imgs/イエンゼンの不等式模式図.png" width=300>

図３．イエンゼンの不等式（ $n=3$ ）

上図は凸包をの網掛け領域で表し、左辺 $\sum_{i=1}^3\lambda_if(x_i)$ に基づく赤い点が、右辺 $f(\sum_{i=1}^3\lambda_ix_i)$ に基づく青い点よりも上に存在していることを示した図です。

連続値の場合にも同様の議論が可能であり、

- 実数 $x$ と
- 実数上の可積分関数 $p(x)(>0)\space s.t.\space\int p(x)\mathrm dx=1$

に対して次が成立します<sup>本当に?</sup>（ $y(x)$ は実数上の可積分関数）。

$$\displaystyle\int f(y(x))p(x)dx\ge f\left(\int y(x)p(x)dx\right)$$

## ラグランジュ未定乗数法

### 等式制約のラグランジュ未定乗数法

#### なんとなくラグランジュ未定乗数法

**ラグランジュ未定乗数法**（*Lagrange's method of indeterminate coefficients*）とは、例えばなんらかの制約 $g(x,y)=0$ のもとで $f(x,y)$ を最大化するというような、**制約付きの最大化/最小化問題を解く手法**です[5]。

制約がない場合に二変数関数 $f(x,y)$ の最大化は、 $\frac{\partial f}{\partial x}=\frac{\partial f}{\partial y}=0$ を満たす点、または区間の端の点のうち最大のものを探すことで可能です。これに対して制約 $g(x,y)=0$ を与えた場合、ラグランジュ乗数 $\lambda$ を用いて、ラグランジュ関数

$$L(x,y,\lambda)=f(x,y)-\lambda g(x,y)$$

を定義でき、 $(x,y)=(\alpha,\beta)$ が極値を与えるならば、 $(\alpha,\beta)$ は

$$\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}=\frac{\partial L}{\partial\lambda}=0$$

または $\frac{\partial g}{\partial x}=\frac{\partial g}{\partial y}=0$ の解となります。

#### 詳しくラグランジュ未定乗数法

関数 $f\colon\R^n\to\R^1$ が（探索範囲で）微分可能であるとします。 $h_j\colon\R^n\to\R^1,j=1,...,p$ を用いて $p$ 個の不等式制約 $\pmb{h}(\pmb{x})=\pmb{0}$ を与えます。すなわち、本最小化問題は次のように書くことができます。

$$\begin{aligned}
    &\min_{\pmb{x}} &f(\pmb{x})&\\
    &\mathrm{sub.\space to} &h_j(\pmb{x})=0,&\space j=1,...,p
\end{aligned}\tag{3}$$

次に $\pmb{\mu}=[\mu_1,...,\mu_p]^T\in\R^p$ を用いて**ラグランジュ関数**（*Lagrangean function*）を次のように定義します。

$$L(\pmb{x,\mu})=f(\pmb{x})+\pmb{\mu}^T\pmb{h}(\pmb{x})$$

ある $\pmb{x}^*\in\R^n$ が式 $(3)$ の局所最適解であれば、ある $\pmb{\mu}^*\in\R^p$ が存在し、次の式を満たします[1]。

$$\begin{aligned}\nabla_x L(\pmb{x}^*,\pmb{\mu}^*)=&&\pmb{0}\\ \nabla_\mu L(\pmb{x}^*,\pmb{\mu}^*)=&&\pmb{h}(\pmb{x}^*)=\pmb{0}\end{aligned}$$

これをより平易に書けば

$$\begin{aligned}
\nabla f(\pmb{x}^*)+{\pmb{\mu}^*}^T\nabla\pmb{h}(\pmb{x}^*)=&\pmb{0}\\\pmb{h}(\pmb{x}^*)=&\pmb{0}
\end{aligned}$$

または

$$\begin{aligned}
\nabla f(\pmb{x}^*)+\sum_{j=1}^p{\mu_j^*\nabla h_j(\pmb{x}^*)}=&\pmb{0}\\h_j(\pmb{x}^*)=&0,\space&j=1,...,p
\end{aligned}$$

とできます。ここで、 $\nabla \pmb{h}(\pmb{x})$ は $\pmb{x}$ に関する制約関数の勾配ベクトルを意味します。また、 $\pmb{\mu}^*$ は**ラグランジュ乗数**（*Lagranean multiplier*）とよばれます。

### 等式・不等式制約のラグランジュ未定乗数法

シンプルなラグランジュ未定乗数法では等式制約しか与えることはできませんでした。これに対して不等式制約を適用できるようにするための条件設定を**Karush-Kuhn-Tucker条件**（KKT条件）といいます[11]。

以下では、等式制約と不等式制約を与える場合のラグランジュ未定乗数法を考えます。はじめに、最適化問題は

$$\begin{aligned}&\min_x&f(\pmb{x})&&\\&\mathrm{sub.\space to}&g_i(\pmb{x})\leq 0,&\space i=1,...,m\\&&h_j(\pmb{x})=0,&\space j=1,...,p\end{aligned}$$

とします。前節と同様に $\pmb{f,g}$ は微分可能とします。また、 $\pmb{h}$ は連続微分可能（かつ Mangasarian-Fromovitz の制約想定を満足する）とします。ある $\pmb{x}^*\in\R^p$ が式 $(3)$ の局所最適解であれば、ある $\pmb{\lambda}^*\in\R^m,\space \pmb{\mu}^*\in\R^n$ が存在し、次の式を満たします[1]

$$\begin{aligned}
    \nabla_x L(\pmb{x}^*,\pmb{\lambda}^*,\pmb{\mu}^*)=&\pmb{0}\\
    \pmb{g}(\pmb{x}^*)\leq\pmb{0},\space{\pmb{\lambda}^*}^T\pmb{g}(\pmb{x}^*)=0,\space\pmb{\lambda}^*\geq&\pmb{0}\\
    \pmb{h}(\pmb{x}^*)=&\pmb{0}
\end{aligned}$$

> Mangasarian-Fromovitz の制約想定を満足する：上の条件設定において、次式を満たす $\pmb{s}\in\R^m$ が存在し、 $\nabla h_j(\pmb{x}^*),\space j=1,...,p$ が１次独立であること[1]
>
> $$\begin{aligned}\nabla g_i(\pmb{x}^*)\pmb{s}<0,i\in I(\pmb{x}^*),\nabla\pmb{h}(\pmb{x}^*)\pmb{s}=\pmb{0}\\\Big(I(\pmb{x}^*)=\{i|g_i(\pmb{x}^*\}=0\Big)\end{aligned}$$

### KKT条件の意味

KKT条件を付けた場合のラグランジュ未定乗数法について概説を述べました。次に、平易なラグランジュ未定乗数法の表現を用いて、KKT条件がそもそもどのような意味を持つものであるかを説明します。

不等式制約 $g_i(x)\leq0\space(i=1,...,m)$ および等式制約 $h_j(x)=0\space(j=1,...,p)$ を持つ関数 $f(x)$ を考えます。このとき、ラグランジュ関数はKKT乗数 $\pmb\lambda, \pmb\mu$ を用いて以下の式で表されます[6]。

$$L(x,\pmb\lambda,\pmb\mu)=f(x)+\sum_{i=1}^m\lambda_ig_i(x)+\sum_{j=1}^p\mu_jh_j(x)$$

ここで、 $\tilde x$ が制約付き最適化問題の局所最適解であるためには、次の４条件が必要です[7][8]。

|条件|英名|数式|
|:-:|:-:|-|
|停留性|*stationarity*|$\plusmn\partial f(x) + \sum_{i=1}^m \lambda_i \partial g_i(x) + \sum_{j=1}^p \mu_j \partial h_j(x) \ni \pmb0$ <br>（ $\plusmn$ : $f(x)$ の最大化問題では $-$ , 最小化問題では $+$ ）|
|同上||関数 $f(x)$ が微分可能な凸関数の場合は次も同等<br>　　 $\plusmn\nabla f(x) + \sum_{i=1}^m \lambda_i \nabla g_i(x) + \sum_{j=1}^p \mu_j \nabla h_j(x) = 0$ <br>　　（ $\plusmn$ : $f(x)$ の最大化問題では $-$ , 最小化問題では $+$ ）||
|実行可能性|*primal feasibility*| $g_i(x) \leq 0$（すべての$i = 1, \dots, m$に対して）<br> $h_j(x) = 0$（すべての$j = 1, \dots, p$に対して）|
|双対実行可能性|*dual feasibility*| $\lambda_i \geq 0$（すべての$i = 1, \dots, m$に対して）|
|補完スラック|*complementary slackness*|$\lambda_i g_i(x) = 0$（すべての$i = 1, \dots, m$に対して）|

各条件の説明は下記の通りです。

- 停留性：この条件は、ラグランジュ関数 $L(x,\pmb\lambda,\pmb\mu)$ の勾配が最適点でゼロであることを意味します。最適点がラグランジュ関数の停留点であり、制約を満たしながら移動することが不可能であることを保証しています[9, §4.6]。
- 実行可能性：この条件は、最適点が元の問題のすべての制約条件を満たすことを意味します。言い換えれば、最適点は元の問題の実行可能点です。これにより、最適な主変数 $x^*$ がどの制約にも違反しない、元の問題に対する有効な解決策であることが保証されます[9, §4.2]。
- 双対実行可能性：この条件は、不等式制約に対するラグランジュ乗数が非負であることを示しています。
- 補完スラック：この条件は、 $\lambda_i$ または $g_i(x)$ のどちらかが必ずゼロとなる必要があることを示しています。すなわち、最適な $x^*, \lambda_i^*$ に対して $\lambda_i^*>0$ ならば $g_i(x^*)=0$ が必要となり、 $g_i(x^*)<0$ ならば $\lambda_i^*=0$ が必要となります（ $g_i(x)\leq0, \lambda_i\geq0$ のため）[9, §4.5]。これは、制約条件がアクティブ（等式で満たされる）であるか、またはラグランジュ乗数がゼロ（目的関数に影響しない）であるかのどちらかであることを示しています。

また、これら４条件に加えて、目的関数 $f(x)$ およびすべての制約関数 $g_i(x,y), h_j(x,y)$ が凸関数であれば、それら５条件と $\tilde x$ が制約付き最適化問題の局所最適解であることは必要十分となります[12]。

- 備考
  - ラグランジュ乗数とは、**目的関数が制約条件に対してどれだけ敏感であるか**を表す変数です。
  - 不等号が $g(x)\geq0$ である不等式制約は、 $-1$ を乗じた関数 $-g(x)$ **を制約関数として扱い**ます。
  - 制約がアクティブであるとは、その制約関数に対するラグランジュ乗数が非ゼロであることを意味します。
  - **スラック変数**を導入し、不等式制約を等式制約として扱う手法もあります[13]。例えば、制約条件 $g(x)\geq0$ がある場合、スラック変数 $s^2$ を導入し $h(x)-s^2=0$ とします。ここで、スラック変数は、**正**値であることを明示するために2乗されています。この手法を用いる場合、双対実行可能性条件などを考える必要がなくなる一方で、変数が一つ増える（ $L(x,\lambda)$ が $L(x,\lambda, s)$ になるような）ため、どちらをとるかは個人の好みによります。
  - $i$ 番目の不等式制約から等号を除く場合（ $g_i(x)<0$ ）は、その補完スラック条件 $\lambda_i g_i(x) = 0$ は、$\lambda_i=0$ のみを満たすことになります（その $x$ で $g_i(x)$ がアクティブな場合）。ただし、そもそも"厳密な不等式" $g(x)<0$ をKKT条件に適用するという状況自体が間違いをはらんでいる可能性が高いかもしれません[10]。

### ラグランジュ未定乗数法のイメージ

最後に、KKT条件を付した場合のラグランジュ未定乗数法のイメージを以下に示します。

<img src="imgs/KKT条件付ラグランジュ未定乗数法のイメージ.png" width=600>

図４．ラグランジュ未定乗数法のイメージ[1]（左）等式制約１つのみ（右）不等式制約１つのみ

- （左）： $\pmb{x}=[x_1,x_2]^T$ であり、課せられた制約が等式制約 $h(\pmb{x})=0$ **１つのみ**である場合の、KKT条件の幾何学的意味を示しています。
- （右）： $\pmb{x}=[x_1,x_2]^T$ であり、課せられた制約が不等式制約 $g(\pmb{x})\leq0$ **１つのみ**である場合の、KKT条件の幾何学的意味を示しています。

どちらの場合においても最適解 $\pmb{x}^*$ においては、目的関数の勾配ベクトル $\nabla f(\pmb{x})$ と制約関数の勾配ベクトル $\nabla h(\pmb{x})$ または $\nabla g(\pmb{x})$ が互いに線形従属になっていることがわかります。

また、可能領域に $f(\pmb{x})$ の最適解が含まれる場合は、当然ですがそれ自体が上の最適化問題の解となります。

<img src="imgs/KKT条件付ラグランジュ未定乗数法のイメージ２.png" width=300>

図５．ラグランジュ未定乗数法のイメージ．可能領域に最適解が含まれる場合

## 参考文献

[1] EE Text システム最適化, 玉置久, オーム社, 第７刷

[2] 凸解析と最適化理論, 田中謙輔, オーム社, 第１刷

[3] [イェンゼンの不等式の３通りの証明](https://manabitimes.jp/math/600)

[4] [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality)

[5] [ラグランジュの未定乗数法と例題](https://manabitimes.jp/math/879)

[6] [Karush–Kuhn–Tucker conditions](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions)

[7] [ML day 8: ラグランジュの未定乗数法と制約付き最適化問題](https://blog.takanabe.tokyo/2023/03/df12434a-322b-4fc6-9e6c-2f3703581b76/)

[8] [Lecture 12: KKT Conditions](https://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/12-kkt-scribed.pdf)

[9] [KKT Conditions, First-Order and Second-Order Optimization, and Distributed Optimization: Tutorial and Survey](https://arxiv.org/abs/2110.01858)

[10] [Questions about constraints and KKT conditions](https://math.stackexchange.com/questions/84252/questions-about-constraints-and-kkt-conditions)

[11] [カルーシュ・クーン・タッカー条件](https://ja.wikipedia.org/wiki/%E3%82%AB%E3%83%AB%E3%83%BC%E3%82%B7%E3%83%A5%E3%83%BB%E3%82%AF%E3%83%BC%E3%83%B3%E3%83%BB%E3%82%BF%E3%83%83%E3%82%AB%E3%83%BC%E6%9D%A1%E4%BB%B6#:~:text=%E3%82%AB%E3%83%AB%E3%83%BC%E3%82%B7%E3%83%A5%E3%83%BB%E3%82%AF%E3%83%BC%E3%83%B3%E3%83%BB%E3%82%BF%E3%83%83%E3%82%AB%E3%83%BC%E6%9D%A1%E4%BB%B6%EF%BC%88,%E3%82%82%E6%89%B1%E3%81%86%E3%81%93%E3%81%A8%E3%81%8C%E3%81%A7%E3%81%8D%E3%82%8B%E3%80%82)

[12] [数理計画法 第11回](http://www.dais.is.tohoku.ac.jp/~shioura/teaching/mp06/mp06-11.pdf)

[13] [Lagrange Multiplier Approach with Inequality Constraints](https://machinelearningmastery.com/lagrange-multiplier-approach-with-inequality-constraints/)