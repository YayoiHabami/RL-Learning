# EMアルゴリズム

## 点推定

母集団のパラメータの値を、観測された標本に基づき推測することを**点推定**（*point estimation*）といいます。 $\theta$ をパラメタに持つ確率分布 $P_\theta$ から、ランダムな標本 $X_1,...,x_n,\space iid\sim P_\theta$ を得たとします。 $\theta$ を標本 $(X_1,...,X_n)$ の関数で推定することになるので、 $\theta$ の**推定量**（*estimator*）として関数 $\hat\theta(X_1,...,X_n)$ または $\hat\theta$ を考えます。また、実現値を代入した $\hat\theta(x_1,...,x_n)$ は**推定値**と呼ばれます。

> $iid$（*ndependently and identically distributed*） : 確率変数 $X_1,..,X_n$ が互いに独立

> 推定量：日本語の字面からは関数であるということが伝わりにくいですが、英名estimatorには確かに関数らしさがあるように感じます。

一般に、推定量がパラメタをより正確に推定できる関数であるために、推定量には不偏性と一致性という性質が備わります[1]。不偏性を持つ推定量は**不偏推定量**（*unbiased estimator*）とよばれ、

$$\mathbb{E}\left[\hat\theta(X_1,...,X_n)\right]=\theta$$

を満たします。この式から、不偏的であるとき $\hat\theta$ は平均的に $\theta$ の周りに分布していることがわかります。次に一致性をもつ推定量は**一致推定量**（*consistent estimator*）とよばれ、

$$\lim_{n\to\infty}{P_\theta(|\hat\theta-\theta|\leq c)}=1, \space\forall c>0, \forall \theta$$

を満たします。上式から、一致性を持つとき推定量 $\hat\theta$ は、 $n$ を大きくするだけ真の値 $\theta$ に近づくことがわかります。

## 最尤法

母集団の確率関数（or 確率密度関数）が $f(x;\theta)$ で与えられているとき、ランダム標本を

$$\pmb{X}=X_1,...,X_n,\space iid\sim f(s;\theta)$$

とします。また、実現値を $\pmb{x}=(x_1,...,x_n)$ とします。

> $f(x;\theta)$ : $\theta$ をパラメタとする $x$ の関数 $f$

観測値として $\pmb{x}$ が与えられたとき、（ $\pmb{x}$ を固定して） $\theta$ の関数として $\mathcal{L}(\theta;\pmb{x})$ 

$$\mathcal{L}(\theta;\pmb{x})=\prod_{\pmb{x}}{f(x;\theta)}$$

を考えます。この関数 $\mathcal{L}(\theta;\pmb{x})$ は**尤度関数**（*likelihood function*）とよばれ、 $\pmb{X}$ の $\pmb{x}$ における同時確率関数（or 同時確率密度関数）です。**最尤法**（*maximum likelihood method*）では、尤度関数 $\mathcal{L}(\theta;\pmb{x})$ を最大化して未知パラメタ $\theta$ を推定します。

<img src="imgs/最尤推定のイメージ.png" width=400>

この尤度関数を最大化する $\theta$ は $\pmb{x}$ の関数として与えられるので、これを ${\hat\theta}^*(\pmb{x})$ と書くとき、 ${\hat\theta}^*={\hat\theta}^*(\pmb{X})$ を**最尤推定量**（*maximum likelihood estimator*; MLE）と言います。以下では、最尤推定量を指すことが明らかである場合には ${\hat\theta}^*={\hat\theta}^*(\pmb{X})$ の代わりに $\theta^*=\theta^*(\pmb{X})$ を用いることにします。

定義から、最尤推定量は 

$$\mathcal{L}(\theta^*; \pmb{X})=\max_\theta{\mathcal{L}(\theta; \pmb{X})}$$

を満たします。ここで、尤度関数は確率関数の積の形をしているので、尤度関数の微分を取った

$$\ell(\theta;\pmb{x})=\log\mathcal{L}(\theta;\pmb{x})=\sum_{\pmb{x}}\log{f(x;\theta)}$$

を**対数尤度関数**（*log-likelihood function'）として、これを最大化する $\theta$ を求めることが多いです。

$\mathcal{L}(\theta; \pmb{X})$ が $\theta$ について微分可能であれば、対数尤度関数の偏微分を $D\ell(\theta; \pmb{X})$ とおくと、最適点では傾きが0となるので（上図参照）

$$D\ell(\theta^*; \pmb{X})=\frac{\partial}{\partial \theta}\ell(\theta^*; \pmb{X})=0$$

がわかります。これを**尤度方程式**といい、最尤推定量は通常この方程式の解を求めることで得られます。


## 参考文献

[1] "統計学 Statistics", 久保川達也, 国友直人, 東京大学出版会, 第２刷

[2] "統計学 One Point EMアルゴリズム", 黒田正博, 共立出版, 第１刷