# POMDP

## 目次

- [POMDP](#pomdp)
  - [目次](#目次)
  - [概要](#概要)
  - [理論](#理論)
  - [参考文献](#参考文献)


## 概要

**マルコフ決定過程**（Markov Decision Process; MDP）は現在の行動と状態を**必ず**知ることができる状態遷移が確率的に起こる動的なモデルです。MDPに対して**観測**の要素を取り入れたものが**部分観測マルコフ決定仮定**（Partially Observable Markov Decision Process; POMDP）です。**一部しか観測できない状況**を前提としたモデルのため、**状態を直接取得することはできません**。

## 理論

離散時間の確率過程である（有限）MDPは4つの要素の組 $(\mathcal{S, A}, p_T, r)$ により規定されます。ここで、その要素を

$$\begin{align}
\mathcal{S}=\{s_1,s_2,...,s_N\}&:状態の有限集合\\
\mathcal{A}=\{a_1,a_2,...,a_M\}&:行動の有限集合\\
p_T&:状態遷移関数\\
r&:報酬関数（即時報酬）\\
\end{align}$$

とします。POMDPを離散時間の確率過程 $P$ であると仮定すると、この $P$ は次のとおりに定義されます。

$$P\coloneqq \{\mathcal{S, A}, s_{p_0}, p_T, r, \mathcal{O}, p_o\}$$

ここで、 POMDPで追加された要素は以下の通りです。

$$\begin{align}
s_{p_0} &:初期状態の確率\\
\mathcal{O}=\{o_1,o_2,...,o_L\} &:観測の有限集合\\
p_o &:観測遷移確率
\end{align}$$

## 参考文献

[1] [POMDP下での強化学習の基礎と応用](https://www.slideshare.net/yasunoriozaki12/pomdp)
