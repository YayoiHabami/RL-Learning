# POMDP

## 目次

- [POMDP](#pomdp)
  - [目次](#目次)
  - [概要](#概要)
  - [理論](#理論)
    - [定義](#定義)
    - [信念状態（belief state）](#信念状態belief-state)
  - [参考文献](#参考文献)
    - [定義と基本的な性質](#定義と基本的な性質)
    - [信念状態](#信念状態)
    - [方策](#方策)
      - [信念状態のマルコフ方策の妥当性](#信念状態のマルコフ方策の妥当性)
      - [続き](#続き)
    - [信念MDP](#信念mdp)
      - [まとめ](#まとめ)
  - [POMDPのプランニング](#pomdpのプランニング)
    - [参考文献](#参考文献-1)


## 概要

**マルコフ決定過程**（Markov Decision Process; MDP）は現在の行動と状態を**必ず**知ることができる状態遷移が確率的に起こる動的なモデルです。MDPに対して**観測**の要素を取り入れたものが**部分観測マルコフ決定仮定**（Partially Observable Markov Decision Process; POMDP）です。**一部しか観測できない状況**を前提としたモデルのため、**状態を直接取得することはできません**。

他のグラフィカルモデルとの位置づけを比較すると次のようになります[1]。

||決定論|確率論|
|-|-|-|
|状態観測が完全|有限オートマトン|マルコフ連鎖|

||行動なし|行動あり|
|-|-|-|
|状態観測が完全|マルコフ連鎖|MDP|
|状態観測が一部|隠れマルコフモデル|POMDP|

## 理論

### 定義

離散時間の確率過程である（有限）MDPは4つの要素の組 $(\mathcal{S, A}, p_T, r)$ により規定されます。ここで、その要素を

$$\begin{align}
\mathcal{S}=\{s_1,s_2,...,s_N\}&:状態の有限集合\\
\mathcal{A}=\{a_1,a_2,...,a_M\}&:行動の有限集合\\
p_T&:状態遷移関数\\
r&:報酬関数（即時報酬）\\
\end{align}$$

とします。POMDPを離散時間の確率過程 $P$ であると仮定すると、この $P$ は次の通りに定義されます。

$$P\coloneqq \{\mathcal{S, A}, s_{p_0}, p_T, r, \mathcal{O}, p_o\}$$

ここで、 POMDPで追加された要素は以下の通りです。

$$\begin{align}
s_{p_0} &:初期状態の確率\\
\mathcal{O}=\{o_1,o_2,...,o_L\} &:観測の有限集合\\
p_o &:観測遷移確率
\end{align}$$

### 信念状態（belief state）

次に、新しく信念状態（*belief state*）という状態を考えます。環境の状態に関する情報を得るため、エージェントは行動 $a$ と観測 $o$ に基づき信念 $b$ を更新する必要があります。ここで、信念状態 $b(s)\in[0,1]$ は、環境が状態 $s\in\mathcal S$ にいる確率を表します[3]。信念状態は

$$b(s)=P(s|h)$$

で定義され、過去の履歴 $h$ から一意に定まります。また、確率ですので合計すると $\sum_{s\in\mathcal{S}} {b(s)}=1$ となります。加えて、信念状態はマルコフ性を持っており、直前の信念 $b$ 、行動 $a$ 、観測された状態 $o$ から次の信念状態は次のように予測されます。




## 参考文献

[1] [POMDP下での強化学習の基礎と応用](https://www.slideshare.net/yasunoriozaki12/pomdp)

[2] [部分観測マルコフ決定過程と強化学習](https://qiita.com/pocokhc/items/6bf5a9519440fe5bd0d5)

[3] [部分観測マルコフ決定過程, Wikipedia](https://ja.wikipedia.org/wiki/%E9%83%A8%E5%88%86%E8%A6%B3%E6%B8%AC%E3%83%9E%E3%83%AB%E3%82%B3%E3%83%95%E6%B1%BA%E5%AE%9A%E9%81%8E%E7%A8%8B)


> ※注意※
>
> 後で $Pr($ をすべて　$\mathrm{Pr}($ に変換する
>
> ※注意※

### 定義と基本的な性質

POMDPとして $P\triangleq \{\mathcal{S,A},p_{s_0},p_T,g,\mathcal{O},p_o\}$ の7つ組で定義される離散時間の確率過程を考えます。ここで $\{\mathcal{S,A},p_{s_0},p_T,g\}$ はMDPで用いられるものと同じであり、ほかは次の通りです。

- 有限観測集合 $\mathcal O\triangleq\{o^1,...,o^{|\mathcal O|}\}\ni o$
- 観測確率関数 $p_o$ : $\mathcal{O\times A\times S}\to [0,1]$ : $p_o(o|\grave{a},s)\triangleq \mathrm{Pr}(O_t = o|A_{t-1}=\grave{a},S_t=s), \forall t\in \mathbb{N}$

もっとも重要な点として、エージェントは観測（または観測信号）とよばれる値 $o\in \mathcal O$ を環境から観測します。

> POMDPにおける状態 $s$ は観測できない。このような変数を **潜在変数** または **隠れ状態** と呼ぶ。

POMDPでの時間ステップ $t$ までの履歴 $\breve{h_t}\in\breve{\mathcal{H}_t}$ を次のように定義します。

$$\breve{h_t}\triangleq\{a_0,r_0,o_1,...,a_{t-1},r_{t-1},o_t\} \tag1$$

ここで、履歴 $\breve{h}$ にチェックがついているのは、POMDPの履歴をMDPの履歴と区別するためである。

![img](参考文献1のグラフィカルモデルを参照)

POMDPのグラフィカルモデルから、 $S_t$ が与えられると $\breve{H_t}$ と $\breve{H}_{t+1}$ \ $\breve{H}_t\space (\triangleq\{A_t,R_t,O_{t+1}\})$ が独立であることがわかります。つまり、

$$\mathrm{Pr}(\breve{H}_t,A_t,R_t,O_{t+1}|S_t,P)=\mathrm{Pr}(\breve{H}_t|s_t,P)\mathrm{Pr}(A_t,R_t,O_{t+1}|S_t,P)\tag2$$

が成り立ちます。式 $(2)$ のような条件付独立を、Dawid(1979)[41] の記法を用いて

$$\breve{H}_t \mathop{\perp\!\!\!\perp} A_t,R_t,O_{t+1}|S_t$$

と表記します。同様に、グラフィカルモデルから次のように様々な条件付独立性を示すことができます。

$$\breve{H}_t \mathop{\perp\!\!\!\perp} A_t,R_t,S_{t+1},O_{t+1}|S_t \tag3$$
$$R_t \mathop{\perp\!\!\!\perp} S_{t+1},O_{t+1}|S_t,A_t \tag4$$

POMDPの解法では、このような確率変数の独立性を利用します。

### 信念状態

POMDPを考えるために**信念状態**を導入し、信念状態に基づく方策の性質を説明します。信念状態とはこれまでの履歴から今現在いずれの隠れ状態 $s\in\mathcal S$ にあるかを表す変数で、環境からの観測があるたびに更新されます。具体的には、ある時間ステップ $t\in\mathbb{N}_0$ の信念状態 $b_t$ は履歴 $\breve{h}_t$ が与えられたときの状態 $S_t$ の条件付確率関数 $b_t\colon \mathcal{S\times \breve{H}_t}\to [0,1]$ として定義されます。

$$b_t(s;\breve{h}_t)\triangleq \mathrm{Pr}(S_t=s|\breve{H}_t=\breve{h}_t,P),\space \forall s\in \mathcal S\tag5$$

以降、簡便化のため、履歴 $\breve{h}_t$ の区別が必要でない限り、 $b_t(s;\breve{h}_t)$ の $\breve{h}_t$ を省略して $b_t(s)$ と表記することにします。信念状態の集合を

$$\mathcal{B}\triangleq\left\{b\colon\mathcal{S}\to[0,1]\colon\sum_{s\in\mathcal{S}}{b(s)}=1\right\} \tag6$$

と表記し、**信念空間**と呼びます。なお、信念状態は確率変数である履歴 $\breve{H}$ に依存するので確率変数であり、これまで通り確率変数として扱う場合は $B$ 、実現値の場合は $b$ と表記します。

信念状態の計算方法ですが、その定義から、 $b_{t+1}$ を単純に計算しようとすると、履歴全体 $\breve{h}_{t+1}=\{a_0,r_0,o_1,...,a_{t-1},r_{t-1},o_t\}$ から求める必要があり、大変です。しかし、信念状態の重要な特徴として、次式のように $\{b_t,a_t,r_t,o_{t+1}\}$ が $b_{t+1}$ に対する十分統計量となり、 $\breve{h}_t$ の代わりに $b_t$ を用いても $b_{t+1}$ が求められること、つまり系列 $B_0,B_1,...$ をマルコフ過程とみなすことができます。任意の $s^\prime\in\mathcal S$ について、

$$\begin{align}
b_{t+1}(s^\prime)&=\mathrm{Pr}(S_{t+1}=s^\prime|\breve{h}_{t+1})\\
&=\mathrm{Pr}(S_{t+1}=s^\prime|\breve{h}_t,a_t,r_t,o_{t+1})\\
&=\frac{\mathrm{Pr}(r_t,S_{t+1}=s^\prime ,o_{t+1}|\breve{h}_t,a_t)}{\sum_{s^\prime\in\mathcal{S}}{\mathrm{Pr}(r_t,S_{t+1}=s^\prime ,o_{t+1}|\breve{h}_t,a_t)}} \space (\because Bayes' theorem)\\
&=\frac{\sum_{s\in\mathcal{S}}{\mathrm{Pr}(r_t,S_{t+1}=s^\prime ,o_{t+1}|S_t=s,a_t)\mathrm{Pr}(S_t=s|\breve{h}_t)}}{\sum_{s^\prime\in\mathcal{S}}{\sum_{s\in\mathcal{S}}{\mathrm{Pr}(r_t,S_{t+1}=s^\prime ,o_{t+1}|S_t=s,a_t)\mathrm{Pr}(S_t=s|\breve{h}_t)}}} \space(\because Equ.(3))\\
&=\frac{p_o(o_{t+1}|a_t,s^\prime)\sum_{s\in\mathcal{S}}{p_T(s^\prime|s,a_t)\mathbb{I}_{\{g(s,a_t)=r_t\}}b_t(s)}}{\sum_{s^\prime\in\mathcal{S}}{p_o(o_{t+1}|a_t,s^\prime)\sum_{s\in\mathcal{S}}{p_T(s^\prime|s,a_t)\mathbb{I}_{\{g(s,a_t)=r_t\}}b_t(s)}}} \space(\because Equ.(4)) \tag7
\end{align}$$

> 指示関数 $\mathbb{I}_B$ : 事象 $B$ が真なら $1$ 、そうでないなら $0$ を出力する。本来は白抜きの $1$ などで記述するが、当環境では再現に難があったため $\mathbb{I}$ で代用している。

簡便化のため、**信念状態作用素** $\Psi\colon\mathcal{B\times A\times R\times O\to B}$ を任意の $s^\prime \in \mathcal S$ に対して

$$(\Psi(b,a,r,o^\prime))(s^\prime)\triangleq\frac{p_o(o_{t+1}|a_t,s^\prime)\sum_{s\in\mathcal{S}}{p_T(s^\prime|s,a_t)\mathbb{I}_{\{g(s,a_t)=r_t\}}b_t(s)}}{\sum_{s^\prime\in\mathcal{S}}{p_o(o_{t+1}|a_t,s^\prime)\sum_{s\in\mathcal{S}}{p_T(s^\prime|s,a_t)\mathbb{I}_{\{g(s,a_t)=r_t\}}b_t(s)}}}\tag8$$

と定義します。

> 右辺の分母が0の場合、 $\Psi$ は定義されません。

このとき、信念状態の再帰式 $(7)$ を

$$b_{t+1}=\Psi(b_t,a_t,r_t,o_{t+1})\tag9$$

と書き直せます。作用素 $\Psi$ は、ベルマン作用素 $\mathrm B$ などと同様に、環境モデル $p_{s_0}, p_T,p_o$ が既知であれば計算可能ですから、信念状態 $b$ を逐次的に更新 $b_{t+1}\coloneqq\Psi(b_t,a_t,r_t,o_{t+1})$ することは、（隠れ状態数 $|\mathcal S|$ が膨大でなければ）簡単です。

なお、ここでは行動 $a$ と観測 $o$ だけでなく報酬 $r$ も考慮して式 $(5)$ のように信念状態を定義していますが、POMDPのベンチマーク課題の多くは状態 $S_t$ と報酬系列 $\{R_0,...,R_t\}$ とが条件つき独立になるような特別な構造をもつため、報酬を考慮しないで信念状態を定義することも多いです。

> 任意の $(\grave{a},s)\in\mathcal{A\times S}$ の状態集合 $\mathcal{S}_{\grave{a},o}\triangleq\{s\in\mathcal{S}\colon p_o(o|\grave{a},s)>0\}$ に対して、報酬関数 $g$ を関数 $\bar{g}\colon\mathcal{A\times O\times A\to R}$ を用いて
>
> $$g(s,a)=\bar{g}(\grave{a},o,a),\space\forall s\in\mathcal{S}_{\grave{a},o},\forall a\in\mathcal{A}$$
>
> と書くことのできるPOMDPであれば、状態と報酬は条件付独立となり、信念状態の計算で報酬を考慮する必要がなくなります。ここで、 $\grave{a}$ は $1$ ステップ前の行動を表します。なぜなら、信念状態 $b$ の更新式 $(7)$ より $\{s\in \mathcal {S}\colon b_t(s)>0\}\subseteq\mathcal{S}_{a_{t-1},o_t}$ なので、仮定より
>
> $$g(s,a_t)=\bar{g}(a_{t-1},o_t,a_t)=r_t,\space\forall s\in\{s\in \mathcal {S}\colon b_t(s)>0\}$$
>
> となり、
>
> $$\mathbb{I}_{\{g(s,a_t)=r_t\}}=1,\space\forall s\in\{s\in \mathcal {S}\colon b_t(s)>0\}$$
>
> となりますので、 $b$ の更新式 $(7)$ を $\bar{b}$ の更新式 $(9)$ に書き換えることができるからです。

つまり、式(1) の履歴 $\breve{h}_t$ から報酬を省略した履歴 $\breve{h_t}\triangleq\{a_0,r_0,...,a_{t-1},r_{t-1},o_t\}$ を用いて信念状態を

$$\bar{b}_t(s)\triangleq \mathrm{Pr}(S_t=s|\bar{H}_t=\bar{h}_t)$$

と定義します。このときの信念状態 $\bar{b}$ の更新則は

$$\bar{b}_{t+1}(s^\prime)\propto p_o(o_{t+1}|a_t,s^\prime)\sum_{s\in\mathcal{S}}{p_T(s^\prime|s,a_t)\bar{b}_t(s)}\tag9$$

となり、$b$ の更新式 $(7)$ と比べて簡単です。しかし、状態と報酬が条件付独立とは限らない一般のPOMDPの場合、信念状態の更新に報酬を用いないと、報酬を用いた場合にくらべ信念状態の不確実性が大きくなり、信念状態にもとづく方策の性能が著しく悪くなる場合があることが実験的にも示されています[83]。

### 方策

POMDPにおける方策を定義します。POMDPはMDPとは異なり $s$ を観測できないため、MDPの $\pi^d\in\Pi^d$ や $\pi^h\in\Pi^h$ などの状態 $s$ や履歴 $h$ に関する方策を、次のように信念状態 $b$ やPOMDPでの履歴 $\breve h$ に関する方策として再定義します。

$$\left .\begin{array}{r}
\breve\Pi\triangleq&\left\{\breve{\pi}\colon\mathcal{A\times B}\to[0,1]\colon\displaystyle\sum_{a\in\mathcal{A}}{\breve{\pi}(a|b)}=1,\space\forall s\in\mathcal S\right\}\\
\breve\Pi^d\triangleq&\left\{\breve{\pi}^d\colon\mathcal{B\to A}\right\}\\
\breve\Pi^h_t\triangleq&\left\{\breve{\pi}^h_t\colon\mathcal{A\times \breve{H}_t}\to[0,1]\colon\displaystyle\sum_{a\in\mathcal{A}}{\breve{\pi}^h_t(a|\breve{h}_t)}=1,\space\forall\breve{h}_t\in\breve{H}_t\right\}
\end{array}\right.\tag{10}$$

ここで、 $\breve{\pi}\in\breve\Pi$ と $\breve\pi^d\in\breve{\Pi}^d$ はそれぞれ信念状態 $b$ の確率的方策と決定的方策であり、それらを単にマルコフ方策と呼ぶこともあります。 $\breve{\pi}^h_t\in\breve{\Pi}^h_t$ は履歴 $\breve{h}_t$ の確率的方策です。明示的には示しませんが、方策系列についても同様に再定義でき、以降チェック $\breve$ を付けてマルコフ決定過程のものと区別することにします。

以下、マルコフ方策系列 $\pmb{\breve{\pi}}^m\triangleq\{\breve\pi_0\in\breve\Pi, \breve\pi_1\in\breve\Pi,...\}$ の特徴を確認します。

#### 信念状態のマルコフ方策の妥当性

任意のPOMDP $\mathrm P$ と、履歴依存の方策系列 $\pmb{\breve\pi}^h=\{\breve\pi_0^h\in\breve\Pi_0^h, \breve\pi_1^h\in\breve\Pi_1^h,...\}$ に対して、次を満たすような信念状態の確率的方策の系列 $\pmb{\breve{\pi}}^m\triangleq\{\breve\pi_0\in\breve\Pi, \breve\pi_1\in\breve\Pi,...\}\in\pmb{\breve\Pi}^M$ が存在する。

$$\mathrm{Pr}(S_t=s,A_t=a|P(\pmb{\breve\pi}^h))=\mathrm{Pr}(S_t=s,A_t=a|P(\pmb{\breve\pi}^m)),\space \forall (t,s,a)\in\mathbb{N}_0\times\mathcal{S\times A} \tag{11}$$

#### 続き

以降、方策の集合をひとまとめにして $\pmb{\breve\Pi}\triangleq(\pmb{\breve\Pi}^H\cup\pmb{\breve\Pi}^M)$ と定義します。なお、遷移可能な信念状態はつねに対応する履歴を１つ以上もつので、実質 $\pmb{\breve\Pi}^H\supseteq\pmb{\breve\Pi}^M$ です。よって $(11)$ から、MDPの場合と同様に目的関数 $f\colon\pmb{\breve\Pi}\to\mathbb{R}$ を

$$f(\pmb{\breve\pi})=f^\prime(\mathrm{Pr}(S_0,A_0|P(\pmb{\breve\pi})),\mathrm{Pr}(S_1,A_1|P(\pmb{\breve\pi})),...),\space\forall \pmb{\breve\pi}\in\pmb{\breve\Pi} \tag{12}$$

のように表現できるのならば、

$$\max_{\pmb{\breve\pi}\in\pmb{\breve\Pi}^M}{f(\pmb{\breve\pi})}=\max_{\pmb{\breve\pi}\in\pmb{\breve\Pi}^H}{f(\pmb{\breve\pi})}$$

が成立することになります。そのため、MDPの場合と同様に、期待リターン

$$f_p(\pmb{\breve\pi})\triangleq\mathbb{E}\left[\sum_{t=0}^\infty{\gamma^tR_t|P(\pmb{\breve\pi})}\right]$$

を目的関数とすれば、

> $f_p$ は、特定の状態に対する期待リターンではなく、 $s_0\sim p_{s_0}$ の場合の期待リターンを考えていることになります。

$$\left.\begin{array}{r}
f_p(\pmb{\breve\pi})=&\displaystyle\sum_{t=0}^\infty\gamma^t\mathbb{E}[g(S_t,A_t)|P(\pmb{\breve\pi})]\\
=&\displaystyle\sum_{t=0}^\infty\sum_{s_t\in\mathcal{S}}\sum_{a_t\in\mathcal{A}}{\mathrm{Pr}(S_t,A_t|P(\pmb{\breve\pi}))g(S_t,A_t)}
\end{array}\right.$$

のように書き直せ、式 $(12)$ のような構造をもつので、

$$\max_{\pmb{\breve\pi}\in\pmb{\breve\Pi}^M}f_p(\pmb{\breve\pi})=\max_{\pmb{\breve\pi}\in\pmb{\breve\Pi}^H}f_p(\pmb{\breve\pi})\tag{14}$$

が成立します。したがって、目的関数が式 $(13)$ の場合、履歴依存の方策系列の集合 $\pmb{\breve\Pi}^H$ まで考える必要はなく、信念状態に基づくマルコフ方策系列の集合 $\pmb{\breve\Pi}^M$ を考えれば十分です。つまり、POMDPにおける期待リターンの最大化問題は次の最適方策 $\pmb{\breve\pi}^*$ の探索問題に帰着できます。

$$\pmb{\breve\pi}^*=\argmax_{\pmb{\breve\pi}\in\pmb{\breve\Pi}^M}f_p(\pmb{\breve\pi})\tag{15}$$

以降は、POMDP問題として上式の最適化問題を考えます。

### 信念MDP

式 $(7)$ で示したように信念状態の系列 $B_0,B_1,...$ はマルコフ性を持つ確率過程であり、行動選択に依存するので、信念状態を状態とするMDPを考えることができます。これは**信念マルコフ決定過程**もしくは**信念MDP**とよばれ、基本的にはベルマン方程式の解の一意性などの性質を持ちます。ただし、これまで扱ってきたMDPでは状態数 $|\mathcal S|$ が有限だったのに対して、信念状態の定義域である信念空間 $\mathcal B$ （式 $(6)$ ）は実数空間であり、状態数は発散しうるため、信念MDPは従来のMDPとすこし異なります。とくに、価値関数の扱いが難しく、後述の近似的アプローチが必要です。また、本来であれば信念MDPを厳格に扱うため、可算無限大の状態を持つMDPの数理の確認が必要ですが、考え方は有限状態のMDPとほとんど変わりませんので省略します。以下、信念MDPを定義し、価値関数 $V_b$ を導入し、 $V_b$ とPOMDPにおける目的関数 $f_p$ との関係性を明らかにして、式 $(15)$ のPOMDP問題を解くためい信念MDPを扱うことが妥当であることを示します。

信念MDPの状態である信念状態の遷移確率ですが、状態が連続空間にあるので、確率分布関数ではなく確率密度関数 $p_b\colon\mathcal{B\times B\times A}\to\bar{\mathcal{R}}_{\geq 0},\space \int_{b^\prime\in\mathcal{B}}{p_b(b^\prime|b,a)db^\prime}=1$ として、式 $(8)$ の信念状態作用素 $\Psi$ を用いて、次のように定義できます。

$$p_b(b^\prime|b,a)\triangleq\lim_{\varepsilon\to\infty}{\frac{}{\varepsilon}}$$

(時間がなくて写せていない)

信念MDPにおける報酬関数 $g_b\colon\mathcal{B\times A}\to\mathbb R$ としては、もとの報酬関数 $g$ の信念状態 $b$ による線形和

$$g_b(b,a)\triangleq\sum_{s\in\mathcal{S}}{b(s)g(s,a)}\tag{17}$$

を用いることにします。また、初期状態（時間ステップ $t=0$ の信念状態）は $b_0$ とします。以上より、信念MDPを $M_b\triangleq\{\mathcal{B,A,}b_0,p_b,g_b\}$ の５つ組として定義できます。また、 $M_b$ において方策 $\breve\pi$ に従い行動選択する確率過程を $M_b(\breve\pi)$ として表記することにします。

(時間がなくて写せていない)

よって、任意の $\pmb{\breve\pi}\in\pmb{\breve\Pi}$ について、POMDPの目的関数（式 $(13)$ ）を信念状態の価値関数 $V_b^{\breve\pi}$ を用いて

$$f_p(\pmb{\breve\pi})=V_b^{\breve\pi}(p_{s_0})\tag{24}$$

とかくことができ、POMDP問題（式 $(15)$ ）を信念MDPにおける最適方策

$$\pmb{\breve\pi}^*=\argmax_{\pmb{\breve\pi}\in\pmb{\breve\Pi}^M} {V_b^{\breve\pi}(p_{s_0})}$$

の探索問題に帰着できます。

上記は非常に有益な結果で、隠れ状態のある一見取り扱いが難しそうなPOMDPを直接扱う必要がなくなります。また、アルゴリズムの設計に有用なMDPの理論解析結果を利用でき、特に、命題2.7から、非定常な方策（方策系列） $\breve{\pmb\pi}\in\breve{\pmb\Pi}^M$ でなくても定常な決定方策 $\breve\pi^d\in\breve\Pi^d$ （式 $(10)$ ）でも最適方策を達成できます。

(時間がなくて写せていない)

また、式(2.29)から、次のように最適方策 $\breve\pi^{d*}\colon\mathcal{B\to A}$ を求めることができます。

$$\breve\pi^{d*}\coloneqq\argmax_{a\in\mathcal{A}}\left\{g_b(b,a)+\gamma\int_{b^\prime}{p_b(b^\prime|b,a)V_b^*(b^\prime)db^\prime}\right\},\space b\in\mathcal{B}\tag{27}$$

また、命題2.4から、 $V_b^*$ は次の信念MDPにおけるベルマン最適方程式の唯一の解となります。

$$V_b^*(b)=\max_{a\in\mathcal{A}}\left\{g_b(b,a)+\gamma\int_{b^\prime}{p_b(b^\prime|b,a)V_b^*(b^\prime)db^\prime}\right\},\space\forall b\in\mathcal{B}$$

#### まとめ

以上より、POMDPにおける最適方策の探索問題は、基本的には、MDPにおける価値反復法のように、式 $(28)$ のベルマン最適方程式を解くことで $V_b^*$ を求めて、式 $(27)$ にもとづいて $V_b^*$ から最適方策 $\breve\pi^{d*}$ を求めればよいことがわかります。

ただし注意すべきことは、通常の有限状態数のMDPとことなり、信念MDPでは状態数が有限でないため、テーブル形式の関数で価値関数を正確に表現することはできず、また式 $(27),(28)$ は $b$ に関する積分を含むため、従来の価値反復法などをそのまま適用することはできません。そのため、次節で紹介するような信念MDPの特徴を利用して、プランニング方法を設計する必要があります。

## POMDPのプランニング

POMDPの環境モデルPが基地であるとして、プランニング（最適方策を求める）方法を説明します。前節でPOMDPを信念MDPの問題に帰着できることを確認しましたが、信念状態が連続空間にあるため、離散状態のMDPのプランニング方法をそのまま利用することはできません。そこで、




> Pが未知の場合はPOMCPとか？(7.3節)

> > 式2と式3違くない？

### 参考文献

[メイン] 強化学習, 森村哲郎, MLP 機械学習プロフェッショナルシリーズ, 講談社, 第一刷
おもにChapter 7.1

[44] D.P. de Farias and B. Van Roy. On constraint sampling in the linear programming approach to approximate dynamic proramming. *Mathematics of Operations Research,* 29(3):462-478, 2004

[83] M.T. Izadi and D. Precup. Using rewards for belief state updates in partially observable Markov decision proess. In *European Conference on Machine Learning*, pages 593-600, 2005.