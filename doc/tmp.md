雑多なことを記述する。

## 目次

- [目次](#目次)
- [面白そうな文献](#面白そうな文献)
- [草稿](#草稿)
  - [POMDP\&MPOによる環境での、エージェントのフロー](#pomdpmpoによる環境でのエージェントのフロー)
  - [補足：ノンパラメトリック](#補足ノンパラメトリック)
  - [ボルツマン分布](#ボルツマン分布)
    - [参考文献](#参考文献)
  - [同時確率・周辺確率](#同時確率周辺確率)
    - [離散分布の場合](#離散分布の場合)
    - [連続分布の場合](#連続分布の場合)
    - [参考文献](#参考文献-1)
  - [価値関数・行動関数の記法](#価値関数行動関数の記法)
    - [参考文献](#参考文献-2)
  - [リプレイバッファ（TF-Agents）](#リプレイバッファtf-agents)
    - [TF-Agentsライブラリ](#tf-agentsライブラリ)
    - [リプレイバッファ](#リプレイバッファ)
    - [参考文献](#参考文献-3)


## 面白そうな文献

- 現場で使える！ Python深層強化学習入門, 伊藤多一ほか, 翔泳社, 第一刷
  - 548.13 G75
  - 実装という面に重きが置かれているため、その前提となるシステムの説明もわかりやすいものが多い。実装対象はDQNとかActor-Criticとかではあるが、基礎知識の学習などに便利そうだと感じた
- Pythonで学ぶ強化学習 第二版, 久保宏隆, 
  - 548.13 Ku11
  - 結構いろいろなアルゴリズムの実装がのっている感じがある。Q学習とかはもちろんA2Cとかも。
- パターン認識と機械学習 -ベイズ推論による統計的予測-, C.M.ビショップ, Springer Japan, 
  - 548.13, B47
  - 黄色い分厚い本。とにかくいろんなことが書いてあるから利用するのに便利そうだと感じた。特に理論面において。ｓ

## 草稿

### POMDP&MPOによる環境での、エージェントのフロー

1) 上記の設定でエージェントの行動が最適化される一般的な流れは以下の通りです。

- エージェントは、観測値から行動へのマッピングを行う初期方策$\pi_0$から始めます。
- エージェントは現在の方策に従って環境と相互作用し、観測値、行動、報酬、次の観測値の軌跡を収集します。
- エージェントは収集した軌跡を用いて、現在の方策の下での状態-行動価値関数$Q(s, a)$と状態分布$\xi(s)$を推定します。
- エージェントはMPOの最適化問題を解きます。これは、現在の方策$\pi_t(a|s)$に対するKLダイバージェンス制約の下で、$Q(s, a)$の期待値を最大化するような新しい行動分布$q(a|s)$を見つけるという問題です。
- エージェントは方策$\pi_{t+1}$を更新します。これは、最適な行動分布$q(a|s)$に等しく設定します。
- エージェントは収束するか、終了条件が満たされるまで、2-5のステップを繰り返します。

2) 物理エンジンによる計算は二つの段階で行われます。

- エージェントが環境と相互作用するとき、物理エンジンは環境とエージェントの行動の動力学をシミュレートし、エージェントに観測値と報酬を生成します。
- エージェントがMPOの最適化問題を解くとき、物理エンジンは目的関数と制約の勾配を行動分布$q(a|s)$に関して計算します。これは物理エンジンが微分可能であり、シミュレーションを通して勾配を伝播させることができるからです。

Bingから

### 補足：ノンパラメトリック

**ノンパラメトリック**は分布が固定されたり既知の形やパラメタを持たないことを意味します。分布は事前に定義されたモデルでは**なく**、データに基づいています。逆に**パラメトリック**はパラメタに基づくデータであり、**特定の分布を背後に想定している**ことを意味します。

### ボルツマン分布

**ボルツマン分布**（*Boltzmann distribution*）とは、システムが特定の状態になる確率を、その状態のエネルギーとシステムの温度の関数として与える確率分布です[A1]。システムが状態 $i$ にある確率 $p_i$ （すなわちボルツマン分布）は

$$p_i\propto \exp \left(-\frac{\varepsilon_i}{kT}\right)$$

のような比例関係を持ちます。ここで、 $\varepsilon_i$ は状態 $i$ におけるエネルギー、 $k,T$ はそれぞれボルツマン定数と温度です。

情報処理の文脈では、コスト関数をエネルギー $\varepsilon$ とみなして、ボルツマン分布を

$$\mathrm{P}(\varepsilon)=\frac{1}{Z}\exp\left(-\frac{\varepsilon}{T}\right)$$

として定義することが多いです[A2]。分母 $Z$ は正規化定数であり、全状態について $P(\varepsilon)$ を積分することで計算されます。また、温度 $T$ はパラメタとして与えることができます。

<img src="imgs/ボルツマン分布（情報処理）.png" width=400>

この関係により、**コストと確率を結び付ける**ことが可能になります。ただし $Z$ は定数であるため、最大化/最小化の問題においてはその値を求めることなく、単純に左辺が指数関数に比例することのみを用いることが可能です。

#### 参考文献
[A1] [Boltzmann distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution)

[A2] [Boltzmann分布](https://ibisforest.org/index.php?Boltzmann%E5%88%86%E5%B8%83)

### 同時確率・周辺確率

#### 離散分布の場合

2つの確率変数 $X,Y$ の組 $(X,Y)$ を考えると、その確率分布は２次元平面 $\R^2$ 上に分布することになります。はじめに、 $X,Y$ が離散型確率変数である場合、すなわち

$$\mathcal{X}=\{x_0,x_1,...\}, \mathcal{Y}=\{y_0,y_1,...\}$$

上で値を取る場合を考えます。このとき、 $(X,Y)$ の組の実現値が $(x_i,x_j)$ となる確率の関数、**同時確率**（*joint probability*）関数を

$$P(X=x_i,Y=y_j)=p_{X,Y}(x_i,y_i)$$

として定義します。このとき、同時確率 $p_{X,Y}(x_i,y_i)$ は

- $p_{X,Y}(x_i,y_i)\geq 0$
- $\sum_{(x_i,y_i)\in\mathcal{X\times Y}}p_{X,Y}(x_i,y_i)=1$

を満たします。

次に、すべての $Y$ の値について総和をとり、 $X$ のみについての確率を考えると、

$$p_X(x_i)=\sum_{y_j\in\mathcal{Y}}p_{X,Y}(x_i,y_i)$$

が定義されます。このように事象（変数）の確率の和（積分）をとることを**周辺化**ともいい、これを $X$ の**周辺確率**（*marginal probability*）関数といい、 $Y$ についても同様に定義されます。

下図におみくじを例にした離散確率における同時確率と周辺確率のイメージを示します。

<img src="imgs/周辺確率・同時確率.png" width=400>

#### 連続分布の場合

次に $X,Y$ が $\R$ 上の連続な確率変数である場合を考えます。離散分布の場合と同様に、**同時確率密度**（*joint probability density*）関数 $f_{X,Y}(x,y)$ は、

- $f_{X,Y}(x,y)\geq 0$
- $\int_{-\infty}^\infty\int_{-\infty}^\infty f_{X,Y}(x,y)dxdy=1$

を満たします。また、同様に $X$ の**周辺確率密度**（*marginal probability density*）関数は

$$f_X(x)=\int_{-\infty}^\infty f_{X,Y}(x,y)dy$$

で定義されます。

#### 参考文献

[A1] 統計学-Statistics, 久保川達也, 国友直人, 東京大学出版会, 第2刷（第８章）

### 価値関数・行動関数の記法

Sutton-Bartoの記法では、

- ベルマン方程式の厳密解として定義される価値関数、行動価値関数は英子文字で表記される。（ $v_\pi(s),v^*(s),q_\pi(s,a),q^*(s,a)$ ）
- ベルマン方程式の近似解あるいは推定値として定義されるものは英大文字で表記される。（ $V_t(s), Q_t(s,a)$ ）

#### 参考文献

[A0] R.S. Sutton and A.G. Barto, "Reinforcement Learning: An Introduction" Second Edition, MIT Press, Cambridge, MA, 2018
[A1] 現場で使える！ Python深層学習入門, 伊藤多一ほか, 翔泳社, 第1刷（Chapter 5.5）

### リプレイバッファ（TF-Agents）

#### TF-Agentsライブラリ

TF-Agents ライブラリは、Tensorflow をベースとする強化学習ライブラリであり、OpenAI Gym や PyBullet ライブラリ、DeepMind の DM Control（ MuJoCo ベース）などをサポートします。基本的な強化学習アルゴリズム、効率のよいリプレイバッファや指標など強化学習に活躍する様々な機能を搭載しています。

TF-Agents 環境は次のようにロードされます（いくつかの依存関係のインストールは必要ですが）。

```python
from tf_agents.environments import suite_gym
env = suite_gym.load("Breakout-v4")
```

`env.reset()` や `env.step(1)` により学習を進めます。また、 `env.observation_spec()`, `env.action_spec()`, `env_time_step_spec()` により環境、行動、時間ステップの仕様を取得することが可能です。

一般的にはTF-Agentsの訓練アーキテクチャは下記のように２つに分かれた形状を取ります[A1]。

<img src="imgs/TF-Agentsの典型的な訓練アーキテクチャ.png" width=600>

図１．TF-Agentsの典型的な訓練アーキテクチャ[A1]

収集側では**ドライバ**が中心となり行動を選択しながら環境と相互作用し、軌跡を収集します。一方右側では**エージェント**が軌跡をもとに訓練を行い、方策の更新を行います。

ここで、一般に環境は複数となりますが、これはCPU/GPUのパワーを活かしつつ相関の低い軌跡を収集するためです。また、ドライバが様々な収集作業の仲介を行うことで、全体としての柔軟性（拡張性）を向上させています。

> 課題にあった環境を作成したい場合は、 `tf_agents.environments.py_environment` パッケージの `PyEnvironment` クラスを継承するカスタムクラスを作成し、 `action_spec()`, `observation_spec()`, `_reset()`, `_step()` といったメソッドをオーバーライドすることで実現できます。

#### リプレイバッファ

TF-Agents ライブラリは、`tf_agents.replay_buffers` パッケージで種々のリプレイバッファを実装しています。モジュール名の先頭が `py_` になっているものは純粋に Python で書かれたものであり、一方 `tf_` は TensorFlow ベースのものです。

たとえば、一様サンプリングのリプレイバッファ（ `TFUniformReplayBuffer` クラス）は次のように使用することができます。

```python
from tf_agents.environments.tf_py_environments import TFPyEnvironment
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.replay_buffers import tf_uniform_replay_buffer

# 環境の設定をラップ
tf_env = TFPyEnvironment(env)

# エージェントの作成
agent = DqnAgent(...)
agent.initialize()

# リプレイバッファの作成
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
  data_spec=agent.collect_data_spec,
  batch_size=tf_env.batch_size,
  max_length=1000000)

# オブザーバの作成
replay_buffer_observer = replay_buffer.add_batch
```

ここで、`data_spec` はリプレイバッファに保存されるデータの使用、`batch_size` は各ステップで渡される軌跡の値（＝ドライバが一回の行動で収集する軌跡の数）です。また、リプレイバッファの上限 `max_length` を100万としていますが、これは2015年のDQN論文に従った値です。ただし、非常に大量のRAMを必要とします。

最後の行ではリプレイバッファに軌跡を書き込むオブザーバを作成していますが、独自実装も可能です。

> 余談：例えば `FrameStack4` ラッパーを使用する場合、本来必要な量の４倍のRAMを使用してしまいます（２つの連続した軌跡を保存したときに、２つめの軌跡の 3/4 は一つ前の軌跡と重複するため）。したがって、`tf_agents.replay_buffers.py_hashed_replay_buffer` パッケージの `PyHashedReplayBuffer` の使用も検討してよいかもしれません。

#### 参考文献

[A1] scikit-learn、Keras、TensorFlow による実線機械学習 第２版, Aurélien Géron, オライリージャパン, 第１刷（Chapter 18.12）