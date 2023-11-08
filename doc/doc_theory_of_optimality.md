# 最適性の理論

## イエンゼンの不等式

$f(x)$ が凸関数のとき、
 
- 任意の $x_1,...,x_n \in \mathbb R$ と
- $\lambda_i\ge0,\sum_{i=1}^n\lambda_i=1$ を満たす任意の $\lambda_1, ..., \lambda_n$

に対して次が成立し、これを**イエンゼンの不等式**（凸不等式）と呼びます[A4][A5]。

$$\displaystyle\sum_{i=1}^n\lambda_if(x_i)\ge f\left(\sum_{i=1}^n\lambda_ix_i\right)$$

この不等式は、凸関数上<sup>?</sup>の任意の $n$ 点が作る凸包上の点（左辺）が、曲線上の点（右辺）よりも上に存在することを示しています。

<img src="imgs/イエンゼンの不等式模式図.png" width=300>

上図は凸包をの網掛け領域で表し、左辺 $\sum_{i=1}^3\lambda_if(x_i)$ に基づく赤い点が、右辺 $f(\sum_{i=1}^3\lambda_ix_i)$ に基づく青い点よりも上に存在していることを示した図です。

> **凸関数**とは、 $f(x)=x^2$ のような、曲面上の任意の2点を結ぶ線分が関数の上に存在するような関数です。いわゆる「下に凸な関数」と言えると思います。より正確には、 $E^n$ において凸集合 $S\subset E^n$ 上で定義される関数 $f\colon S\to E^1$ に対して、 $n=2$ のときのイエンゼンの不等式（ $\lambda_1 f(\pmb{x}_1)+(1-\lambda_1)f(\pmb{x}_2)\geq f\big(\lambda_1 \pmb{x}_1+(1-\lambda_1)\pmb{x}_2\big)$ ）が成り立つような関数です（ $E$ は $\R$ など何らかの集合 ）[A20]。

連続値の場合にも同様の議論が可能であり、

- 実数 $x$ と
- 実数上の可積分関数 $p(x)(>0)\space s.t.\space\int p(x)\mathrm dx=1$

に対して次が成立します<sup>本当に?</sup>（ $y(x)$ は実数上の可積分関数）。

$$\displaystyle\int f(y(x))p(x)dx\ge f\left(\int y(x)p(x)dx\right)$$

## ラグランジュ未定乗数法

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

各条件の説明は下記の通りです。

- 停留性：この条件は、ラグランジュ関数 $L(x,\pmb\lambda,\pmb\mu)$ の勾配が最適点でゼロであることを意味します。最適点がラグランジュ関数の停留点であり、制約を満たしながら移動することが不可能であることを保証しています[A13, §4.6]。
- 実行可能性：この条件は、最適点が元の問題のすべての制約条件を満たすことを意味します。言い換えれば、最適点は元の問題の実行可能点です。これにより、最適な主変数 $x^*$ がどの制約にも違反しない、元の問題に対する有効な解決策であることが保証されます[A13, §4.2]。
- 双対実行可能性：この条件は、不等式制約に対するラグランジュ乗数が非負であることを示しています。
- 補完スラック：この条件は、 $\lambda_i$ または $g_i(x)$ のどちらかが必ずゼロとなる必要があることを示しています。すなわち、最適な $x^*, \lambda_i^*$ に対して $\lambda_i^*>0$ ならば $g_i(x^*)=0$ が必要となり、 $g_i(x^*)<0$ ならば $\lambda_i^*=0$ が必要となります（ $g_i(x)\leq0, \lambda_i\geq0$ のため）[A13, §4.5]。これは、制約条件がアクティブ（等式で満たされる）であるか、またはラグランジュ乗数がゼロ（目的関数に影響しない）であるかのどちらかであることを示しています。

また、これら４条件に加えて、目的関数 $f(x)$ およびすべての制約関数 $g_i(x,y), h_j(x,y)$ が凸関数であれば、それら５条件と $\tilde x$ が制約付き最適化問題の局所最適解であることは必要十分となります[A10]。

- 備考
  - ラグランジュ乗数とは、**目的関数が制約条件に対してどれだけ敏感であるか**を表す変数です。
  - 不等号が $g(x)\geq0$ である不等式制約は、 $-1$ を乗じた関数 $-g(x)$ **を制約関数として扱い**ます。
  - 制約がアクティブであるとは、その制約関数に対するラグランジュ乗数が非ゼロであることを意味します。
  - **スラック変数**を導入し、不等式制約を等式制約として扱う手法もあります[A14]。例えば、制約条件 $g(x)\geq0$ がある場合、スラック変数 $s^2$ を導入し $h(x)-s^2=0$ とします。ここで、スラック変数は、**正**値であることを明示するために2乗されています。この手法を用いる場合、双対実行可能性条件などを考える必要がなくなる一方で、変数が一つ増える（ $L(x,\lambda)$ が $L(x,\lambda, s)$ になるような）ため、どちらをとるかは個人の好みによります。
  - $i$ 番目の不等式制約から等号を除く場合（ $g_i(x)<0$ ）は、その補完スラック条件 $\lambda_i g_i(x) = 0$ は、$\lambda_i=0$ のみを満たすことになります（その $x$ で $g_i(x)$ がアクティブな場合）。ただし、そもそも"厳密な不等式" $g(x)<0$ をKKT条件に適用するという状況自体が間違いをはらんでいる可能性が高いかもしれません[A15]。

## 参考文献

[A20] 凸解析と最適化理論, 田中謙輔, オーム社, 第１刷

[A4] [イェンゼンの不等式の３通りの証明](https://manabitimes.jp/math/600)

[A5] [Jensen's inequality](https://en.wikipedia.org/wiki/Jensen%27s_inequality)

[A7] [ラグランジュの未定乗数法と例題](https://manabitimes.jp/math/879)

[A8] [カルーシュ・クーン・タッカー条件](https://ja.wikipedia.org/wiki/%E3%82%AB%E3%83%AB%E3%83%BC%E3%82%B7%E3%83%A5%E3%83%BB%E3%82%AF%E3%83%BC%E3%83%B3%E3%83%BB%E3%82%BF%E3%83%83%E3%82%AB%E3%83%BC%E6%9D%A1%E4%BB%B6#:~:text=%E3%82%AB%E3%83%AB%E3%83%BC%E3%82%B7%E3%83%A5%E3%83%BB%E3%82%AF%E3%83%BC%E3%83%B3%E3%83%BB%E3%82%BF%E3%83%83%E3%82%AB%E3%83%BC%E6%9D%A1%E4%BB%B6%EF%BC%88,%E3%82%82%E6%89%B1%E3%81%86%E3%81%93%E3%81%A8%E3%81%8C%E3%81%A7%E3%81%8D%E3%82%8B%E3%80%82)

[A9] [ML day 8: ラグランジュの未定乗数法と制約付き最適化問題](https://blog.takanabe.tokyo/2023/03/df12434a-322b-4fc6-9e6c-2f3703581b76/)

[A10] [数理計画法 第11回](http://www.dais.is.tohoku.ac.jp/~shioura/teaching/mp06/mp06-11.pdf)

[A11] [Karush–Kuhn–Tucker conditions](https://en.wikipedia.org/wiki/Karush%E2%80%93Kuhn%E2%80%93Tucker_conditions)

[A12] [Lecture 12: KKT Conditions](https://www.stat.cmu.edu/~ryantibs/convexopt-S15/scribes/12-kkt-scribed.pdf)

[A13] [KKT Conditions, First-Order and Second-Order Optimization, and Distributed Optimization: Tutorial and Survey](https://arxiv.org/abs/2110.01858)

[A14] [Lagrange Multiplier Approach with Inequality Constraints](https://machinelearningmastery.com/lagrange-multiplier-approach-with-inequality-constraints/)

[A15] [Questions about constraints and KKT conditions](https://math.stackexchange.com/questions/84252/questions-about-constraints-and-kkt-conditions)