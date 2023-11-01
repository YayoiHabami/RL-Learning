# Q関数とは

環境が未知である（**モデルフリー**）場合、行動に対する報酬や次の状態をデータとして実環境やシミュレータから観測しデータから方策を学習することになります。多くの機械学習ではオンライン学習を行いますから、データ収集時（学習途中）の方策も必要となります。このとき、データの**探索と活用のトレードオフ**（*exploration-exploitation trade-off*）の考慮が必要となります。

## 探索と活用とは？

探索と活用はそれぞれ行動選択において何を目的とするかを意味し、次のように定義されます[0]。

- **探索**（*exploration*） ：方策の精度を高めるためのデータ収集
- **活用**（*exploitation*）：リターン（期待割引報酬和）の最大化

探索では必ずしも最良の行動を選択するとは限らず、様々なデータを集めることを目的とします（局所最適解に陥ることを防ぐ）。一方活用はリターンの最大化のみが目的であるため、探索と活用は相反する目的となります。

## 効用関数（Q関数）

数理モデルを利用し直接方策モデルを規定するアプローチ（**方策ベース**）も存在しますが、強化学習においては**効用関数**（*utility function*）を用いて間接的に方策を規定するアプローチ（**価値ベース**）も一般的です。効用関数は状態 $s$ で行動 $a$ を選択することの効用（推定価値）を出力する関数であり、 $q\colon\mathbb{R}^\mathcal{S\times A}\to\mathbb{R}$ と規定されます。これを用いれば、例として $q(s,a)<q(s,a')$ である場合に、状態 $s$ では行動 $a'$ のほうがよい行動であるといった判断ができるようになります。

> 数理モデル：一般化線形モデルやニューラルネットワークなど

効用関数 $q$ に何を用いるかは問題設定に依存しますが、通常はリターン $C_0=\sum_{t=0}^\infty{\gamma^t R_t}$ の、状態行動（と方策）の条件付き期待値

$$Q^\pi(s,a)\triangleq\mathbb{E}^\pi[C_0|S_0=s,A_0=a]\tag1$$

の推定値 $q(s,a)$ として用います。関数 $Q^\pi\colon\mathcal{S\times A}\to\mathbb{R}$ は**行動価値関数**（*action value function*）もしくは**Q関数**とよばれ、状態空間の価値関数 $V^\pi$ を**状態行動空間に拡張した価値関数**といえます。Q関数は期待リターンの不確実性を考慮しないため、これを考慮するためには方策にランダム性を取り入れることになります（ $\varepsilon$ 貪欲方策やソフトマックス方策など）。一方で、不確実度 $u\colon\mathcal{S\times A}\to\mathbb{R}$ により補正した効用関数を用いる方法も存在します（UCB1法など）。

> 価値関数 $V^\pi\colon\mathcal{S}\to\mathbb{R}$
>
> $$V^\pi(s)\triangleq\mathbb{E}^\pi[C_0|S_0=s]$$

効用関数を用いて定義される方策として、貪欲方策、 $\varepsilon$ 貪欲方策、ソフトマックス方策の３方策を以下に簡単に示します[0]。

### 貪欲方策

**貪欲方策**（*greedy policy*） $\pi\colon\mathcal{S}\times\mathbb{R}^\mathcal{S\times A}\colon\to\mathcal{A}$ はもっとも単純な方策モデルです。つねに効用関数が最大となる行動を選択する**決定的方策**であり、

$$\pi(s;q)\triangleq\argmax_{a\in\mathcal{A}}{q(s,a)}$$

で定義されます。常に効用が最大となる行動が選択されるため、探索の面について探索と活用のトレードオフが十分でない方策モデルといえます。

### $\varepsilon$ 貪欲方策

$\boldsymbol\varepsilon$ **貪欲方策**（ε-greedy policy） $\pi\colon\mathcal{A\times S}\times\mathbb{R}^\mathcal{S\times A}\times [0,1]\colon\to[0,1]$ は確率 $\varepsion\in[0,1]$ でランダムに行動を選択する、貪欲方策の**確率的方策**版です。

$$\pi(a|s;q,\varepsilon)=\left\{\begin{aligned}&1-\varepsilon+\dfrac{\varepsilon}{|\mathcal{A}|}\space&\big(a=\argmax_b{a(s,b)}\big)\\&\dfrac{\varepsilon}{|\mathcal{A}|}\space&\big(それ以外\big)\end{aligned}\right.$$

ここで $\varepsilon$ はハイパーパラメタであり、1に近いほど常にランダムな行動を選択するようになります。必ずしも $\varepsilon$ は定数ではなく、徐々にこれを小さくするような手法も存在します（GILE方策など）。

### ソフトマックス方策

**ソフトマックス方策**（*softmax policy*） $\pi\colon\mathcal{A\times S}\times\mathbb{R}^\mathcal{S\times A}\times\mathbb{R}_{\geq0}\colon\to[0,1]$ は、ボルツマン分布に従い行動を選択する、貪欲方策の**確率的方策**版です。

$$\pi(a|s;q,\beta)=\frac{\exp(\beta q(s,a))}{\displaystyle\sum_{b\in\mathcal{A}}\exp(\beta q(s,b))}$$

$\beta\in\mathbb{R}_{\geq0}$ はハイパーパラメタであり、小さいほどランダムな行動を選択するようになります。

特徴として、偏微分 $\frac{\partial\pi}{\partial q(s,b)}$ および $\frac{\partial\pi}{\partial \beta}$ が計算可能な点があります。前者 $\frac{\partial\pi}{\partial q(s,b)}$ は効用関数を微小変化させた際の方策の変化を示すため、効用関数の逐次的更新に役立ちます。また後者 $\frac{\partial\pi}{\partial \beta}$ は $\beta$ を通常のパラメタとして扱うことをも可能とします（方策勾配法など）。

> $\mathbb{R}_{\geq0}$ ： $0$ 以上の実数の集合

## 補足：価値関数

**価値関数**は強化学習において最適な行動則（方策）を比較的容易に計算するために導入される関数です。

> 比較的容易に：すべての初期状態に対するすべての方策を列挙し、その中から最適な方策を選択するためには、現実的にほぼ不可能な量の計算を行うことになります。それと比較すれば、価値関数を導入した手法は容易な計算量だといえます。

状態 $x\in\mathcal{X}$ の最適価値 $V^*(x)$ は、過程が $x$ から始まったときに達成可能な最も高い期待リターンを意味します。したがって、最適な方策では任意の状態 $x$ から始めたときに最適価値 $V^*(x)$ を達成できることになります。強化学習（のうち**価値ベース**手法）では、価値関数 $V(x)$ を**最適価値関数** $V^*(x)$ に近づけるような行動則を探索することで目的を達成することになります。

方策 $\pi$ のもとでの**価値関数**（*value function*） $V^\pi\colon\mathcal{S}\to\mathbb{R}$ は

$$V^\pi(s)=\mathbb{E}\left[\sum_{t=0}^\infty{\gamma^tR_t}\Big|S_0=s\right], \space s\in\mathcal{S}$$

で定義されます。ここで、 $R_t;t\geq0$ は方策 $\pi$ に従った過程のステップ $t$ で得られる報酬です。このとき、初期値 $S_0$ が任意の $s$ に対して $\mathrm{P}(S_0=s)>0$ を満たすような分布から無作為に生成されるように定義すれば、すべての状態 $s$ に対して上式は矛盾なく定義できます[1]。

次に、方策 $\pi$ のもとでの**行動価値関数** $Q^\pi\colon\mathcal{S\times A}\to\mathbb{R}$ は、上に述べたように

$$Q^\pi(s,a)=\mathbb{E}\left[\sum_{t=0}^\infty{\gamma^tR_t}\Big|S_0=s,A_0=a\right], \space s\in\mathcal{S}, a\in\mathcal{A}$$

で定義されます。

文献や記事によりこれら関数の名称にはばらつきがあります。以下に見たことのある表記ゆれをいくつか示します。

| $V^\pi$ の名称 | $Q^\pi$ の名称 |説明|
|-|-|-|
|価値関数|-|最も基本的は価値関数であることから？|
|価値関数|行動価値関数| $Q^\pi$ は $V^\pi$ の対象に行動空間を加えたものであることから？|
|(V関数, V値)|Q関数, Q値|関数の記号から。Q値は正確にはQ関数の出力値<br>V関数&V値についてはほとんど見たことがない|
|状態価値関数|状態行動価値関数| $V^\pi$ が状態空間の、 $Q^\pi$ が状態行動空間の関数であることから|

## 補足：Q関数の更新式

当初Q関数について調べていた際、Q関数の式として次のような式が記述されている記事があり、式 $(1)$ の定義との違いに悩まされたことがありました。

$$\begin{aligned}&Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha_t\delta_t\\

&s.t.\space\delta_t\coloneqq \left\{\begin{aligned}
r_t+\gamma\max_{a'\in\mathcal{A}}{Q(s_{t+1},a')}-Q(s_t,a_t)\space&\big(\textit{Q-learning}\big)\\
r_t+\gamma\max_{a'\in\mathcal{A}}{Q(s_{t+1},a')}-Q(s_t,a_t)\space&\big(SARSA\big)
\end{aligned}\right.\end{aligned}\tag{S.1}$$

ここで、上式の $\delta$ はTD誤差、 $\alpha_t$ は学習率です[0]。節タイトルからわかるように、実際には上式はQ関数の更新式であり、オンライン学習の一種である**Q-learning**法や**SARSA**法で使用されます。Q-learning法はオフポリシー型の手法であり、SARSA法はオンポリシー型の手法です。

以上から、Q関数の定義は式 $(1)$ であり、式 $(S.1)$ はQ関数の更新式であるという違いを理解することができました。

## 参考文献

[0] 強化学習, 森村哲郎, MLP 機械学習プロフェッショナルシリーズ, 講談社, 第一刷
（Chapter 3）

[1] 速習 強化学習 -基礎理論とアルゴリズム-, Csaba Szepesvári, 小山田創哲, 共立出版, 第2刷（1.3章）