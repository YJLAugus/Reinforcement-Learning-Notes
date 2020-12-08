# 强化学习入门基础-马尔达夫决策过程（MDP）

> 作者：YJLAugus  博客： https://www.cnblogs.com/yjlaugus

## MDP背景介绍

### Random Variable

**随机变量（Random Variable）**，通常用大写字母来表示一个随机事件。比如看下面的例子：

$X$: 河水是咸的

$Y$: 井水是甜的

很显然，$X$, $Y$两个随机事件是没有关系的。也就是说$X$和$Y$之间**是相互独立**的。记作：
$$
\large 
X \bot Y
$$

### Stochastic Process

对于一类随机变量来说，它们之间存在着某种关系。比如：

$S_{t}$：表示在 $t$ 时刻某支股票的价格，那么 $S_{t+1}$ 和 $S_t$ 之间一定是有关系的，至于具体什么样的关系，这里原先不做深究，但有一点可以确定，两者之间一定存在的一种关系。随着时间 $t$ 的变化，可以写出下面的形式：
$$
\large
...S_t, S_{t+1},S_{t+2}...
$$
这样就生成了一组随机变量，它们之间存在着一种相当复杂的关系，也就是说，各个随机变量之间存在着关系，即不相互独立。由此，我们会把按照某个时间或者次序上的一组不相互独立的随机变量的这样一个整体作为研究对象。这样的话，也就引出了另外的一个概念：**随机过程（Stochastic Process）**。也就是说随机过程的研究对象不在是单个的随机变量，而是一组随机变量，并且这一组随机变量之间存在着一种非常紧密的关系（不相互独立）。记作：
$$
\large
\lbrace S_t \rbrace ^\infty_{t=1}
$$


### Markov Chain/Process

**马尔科夫链（Markov Chain）**即马尔可夫过程，是一种特殊的随机过程——具备马尔可夫性的随机过程。

* 马尔可夫性：（Markov Property）: 还是上面股票的例子，如果满足 $P(S_{t+1} \mid S_t,S_{t-1}...S_1) = P(S_{t+1}\mid S_t)$，即具备了马尔可夫性。简单来说，$S_{t+1}$ 和$S_t$之间存在关系，和以前的时刻的没有关系，即只和“最近的状态” 有关系。
* 现实例子：下一个时刻仅依赖于当前时刻，跟过去无关。比如：一个老师讲课，明天的讲课状态一定和今天的状态最有关系，和过去十年的状态基本就没关系了。
* 最主要考量：为了简化计算。$P(S_{t+1} \mid S_t,S_{t-1}...S_1) = P(S_{t+1}\mid S_t)$ 如果 $S_{t+1}$ 和 $S_t,S_{t-1}...S_1$ 都有关系的话，计算的话就会爆炸了。

马尔可夫链/过程 即满足马尔可夫性质的随机过程，记作：
$$
\large P(S_{t+1}) \mid S_t,S_{t-1}...S_1) = P(S_{t+1}\mid S_t)
$$

### State Space Model

**状态空间模型（State Space Model）**，常应用于 HMM,Kalman Filterm Particle Filter，关于这几种这里不做讨论。在这里就是指马尔可夫链 + 观测变量，即`Markov Chain + Obervation`

![spm](https://gitee.com/YJLAugus/pic-go/raw/master/img/spm.svg)

如上图所示，s1-s2-s3为马尔可夫链，a1, a2, a3为观测变量，以a2为例，a2只和s2有关和s1, s3无关。状态空间模型可以说是由马尔可夫链演化而来的模型。记作：

<center>Markov Chain + Obervation</center>

### Markov Reward Process

**马尔可夫奖励过程（Markov Reward Process）**，即马尔可夫链+奖励，即：`Markov Chain + Reward`。如下图：

![图片描述](https://gitee.com/YJLAugus/pic-go/raw/master/img/reward.svg)



举个例子，比如说你买了一支股票，然后你每天就会有“收益”，当然了这里的收益是泛化的概念，收益有可能是正的，也有可能是负的，有可能多，有可能少，总之从今天的状态$S_t$ 到明天的状态 $S_{s+1}$  ，会有一个`reward`。记作：

<center>Markov Chain + Reward</center>

### Markov Decision Process

**马尔可夫决策过程（Markov Decision Process）**，即马尔可夫奖励过程的基础上加上`action`，即：`Markov Chain + Reward + action`。如果还用刚才的股票为例子的话，我们只能每天看到股票价格的上涨或者下降，然后看到自己的收益，但是无法操作股票的价格的，只有看到份，只是一个“小散户”。这里的马尔可夫决策过程相当于政策的制定者，相当于一个操盘手，可以根据不同的状态而指定一些政策，也就相当于 action。

在马尔可夫决策过程中，所有的**状态**是我们看成离散的，有限的集合。所有的**行为**也是离散有限的集合。记作：

$$
\large
\enclose{box}{
\it S: \quad state \quad set \quad \quad \quad\quad\quad\quad\quad\quad\quad\quad S_t \\ 
\it A: \quad action \quad set ,\quad \quad \forall s \in S,A_{(s)} \quad\quad A_t\\
\it R: \quad reward \quad set \quad\quad\quad\quad\quad\quad\quad\quad\quad R_t, R_{(t+1)} \\
}
$$
对于上述公式简单说明，$S_t$ 用来表示某一个时刻的状态。$A_{(s)}$ 表示在**某一个状态**时候的行为 ，这个行为一定是基于某个状态而言的，假设在$t$ 时刻的状态为$S$ 此时的`action`记作 $A_t$ 。$R_t 和 R_{(t+1)}$ 只是记法不同，比如下面的例子：从$S_t$状态经过 $A_t$ 到$S_{t+1}$状态，获得的奖励一般记作$R_{(t+1)}$。 也就是说$S_t$， $A_t$ ，$R_{(t+1)}$ 是配对使用的。

![](https://gitee.com/YJLAugus/pic-go/raw/master/img/coupleR.svg)

### Summary

![](https://gitee.com/YJLAugus/pic-go/raw/master/img/sum01.png)

## MDP动态特性

### Markov Chain

马尔可夫链只有一个量——**状态**。比如 $S\in(s_1,s_2,s_3,s_4,s_5,s_6,s_7,s_8,s_9,s_{10})$ ，在状态集合中一共有十个状态，每个状态之间可以互相转化，即可以从一个状态转移到另外一个状态，当然，“另外的状态” 也有可能是当前状态本身。如下图所示，s1状态到可以转移到s2状态，s1状态也可以转移到自己当前的状态，当然s1也有可能转移到s3，s4，s5，状态，下图中没有给出。

![](https://gitee.com/YJLAugus/pic-go/raw/master/img/recg.svg)

根据上面的例子，我们可以把所有的状态写成矩阵的形式，就成了**状态转移矩阵**。用状态转移矩阵来描述马尔可夫链的**动态特性**。以上面的状态集合$S\in(s_1,s_2,s_3,s_4,s_5,s_6,s_7,s_8,s_9,s_{10})$ 为例，可得一个 $10\times10$ 的矩阵。如下图所示：
$$
\large
\begin{bmatrix}
    s_1s_1 &... & s_1s_{10} \\
    \vdots & \vdots & \vdots \\
  
    s_{10}s_1  & ... & s_{10}s_{10}\\
\end{bmatrix}
$$
由上面的例子可知，在状态转移的过程中，对于下一个状态转移是有概率的，比如说s1转移到到s1状态的概率可能是0.5，s1有0.3的概率转移到s2状态。⻢尔科夫过程是⼀个⼆元组（S， P） ， 且满⾜： **S是有限状态集合， P是状态转移概率。**   可得：
$$
\large
P=
\begin{bmatrix}
    P_{11} &... & P_{1n} \\
    \vdots & \vdots & \vdots \\
  
    P_{n1}  & ... & P_{nn}\\
\end{bmatrix}
$$
一个简单的例子：如图2.2所⽰为⼀个学⽣的7种状态{娱乐， 课程1， 课程2， 课程3， 考过， 睡觉， 论⽂}， 每种状态之间的转换概率如图所⽰。 则该⽣从课程 1 开始⼀天可能的状态序列为：  

![image-20201127124916367](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201127124916367.png)

### MRP

在 MPR 中，打个比喻，更像是随波逐流的小船，没有人为的干预，小船可以在大海中随波逐流。

### MDP

在MDP中，打个比喻，更像是有人划的小船，这里相比较MRP中的小船来说，多了“人划船桨”的概念，可以认为控制船的走向。这里我们看下面的图：



![](https://gitee.com/YJLAugus/pic-go/raw/master/img/dymic1.svg)



s1状态到s2状态的过程，agent从s1发出action A1，使得s1状态转移到s2状态，并从s2状态得到一个R2的奖励。其实就是下图所示的一个过程。这是一个**动态的过程**，由此，引出**动态函数**。

![image-20201127132127738](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201127132127738.png)

* **动态函数：** The function $p$ defines the dynamics of the MDP.  这是书上的原话，也就说，这个动态函数定义了MDP的`动态特性`，动态函数如下：
  $$
  \large
  p(s',r\mid s,a) \dot{=} Pr\lbrace S_{t+1}=s',R_{t+1} = r \mid S_{t} =s,A_{t}=a \rbrace
  $$


* **状态转移函数：** 我们去掉 $r$ ，也就是`reward`，动态函数也就成了状态转移函数。

$$
\large{
p(s'\mid s,a) \dot{=} Pr\lbrace S_{t+1}=s',\mid S_{t} =s,A_{t}=a \rbrace \\
  p(s'\mid s,a) = \sum_{r\in R} p(s'\mid s,a)
  }
$$

* **`reward`的动态性：** 在 s 和 a 选定后，r 也有可能是不同的，即 r 也是随机变量。但是，大多数情况在 s 和 a 选定后 r 是相同的，这里只做简单的介绍。

### Summary

![image-20201127135356269](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201127135356269.png)

## MDP价值函数

### 策略的定义

在MDP中，即马尔可夫决策过程，最重要的当然是**策略（Policy）**，**用 $\pi$ 来表示**。在策略中其主要作用的就是`action`，也即 $A_t$，需要指出的一定是，action 一定是基于某一状态 S 时。看下面的例子：

![](https://gitee.com/YJLAugus/pic-go/raw/master/img/policy.svg)

即，当 $S_t = S$ 状态时，无论 $t$ 取何值，只要遇到 $S$ 状态就 选定 $a_1$ 这个 action ，这就是一种策略，并且是确定性策略。

### 策略的分类

* **确定性策略：**也就是说和时间 $t$ 已经没有关系了，只和这个状态有关，只要遇到这个状态，就做出这样的选择。

* **随机性策略：**与确定性策略相对，当遇到 $S$ 状态时，可能选择 $a_1$ ，可能选择 $a_2$，也可能选择 $a_3$。只是选择action的概率不同。如下图，就是**两种不同**的策略:

  ![](https://gitee.com/YJLAugus/pic-go/raw/master/img/p1.svg)

  从上面两天图中，因为一个策略是基于一个状态而言的，在 $S$ 状态，可能选择 $a_1$ ，可能选择 $a_2$，也可能选择 $a_3$，故三个 `action` 之间是**或**的关系，所以说以上是两个策略，而不要误以为是6个策略。

故策略可分为确定性策略和随机性策略两种。


$$
\large
Policy= \begin{cases} 确定性策略, & \text {a $\dot{=}\pi(s)$} \\ 随机性策略, & \text { $\pi(a\mid s) \dot{=} P \lbrace A_t=a \mid S_t = s \rbrace$} \end{cases}
$$

对于随机性策略而言，给定一个 $s$ ，选择**一个 $a$** ，也就是条件概率了。

> 确定性策略可以看作是一种特殊的随机性策略，以上表-Policy1为例，选择a1的概率为1，选择a2，a3的概率都为0。

### 最优策略

在所有的策略中一定存在至少一个**最优策略**，而且在强化学习中，`reward`的获得有**延迟性（delay）**，举雅达利游戏中，很多游戏只有到结束的时候才会知道是否赢了或者输了，才会得到反馈，也就是`reward`，所以这就是奖励获得延迟。当选定一个状态 $S_t$ 时，选定action $A_t$ ，因为奖励延迟的原因可能对后续的 $S_{t+1}$  等状态都会产生影响。这样，就不能用当前的reward来衡量策略的好坏、优劣。这里引入了回报和价值函数的概念。

#### 回报（$G_t$）

而是后续多个reward的加和，也就是**回报**，用 $G_t$ 表示 $t$ 时刻的回报。




![](https://gitee.com/YJLAugus/pic-go/raw/master/img/optimp.svg)



如上图所示，此时的“回报”可以表示为：$G_t = R_{t+1} + R_{t+2}+ \ ...\ +R_T$

值得注意的是，$T$ 可能是有限的，也有可能是无限的。

举例：张三对李四做了行为，对李四造成了伤害，李四在当天就能感受到伤害，而且，这个伤害明天，后头都还是有的，但是，时间是最好的良药，随着时间的推移，李四对于张三对自己造成的伤害感觉没有那么大了，会有一个wei折扣，用 $\gamma$ 来表示。故**真正的回报**表示为：
$$
\large
G_t = R_{t+1} + \gamma R_{t+2}+\gamma^2 R_{t+3} \ ...\ +\gamma^{T-t-1}R_T = \sum_{i=0}^{\infty}\gamma^i R_{t+i+1} \quad \quad \gamma\in[0,1],\quad (T\rightarrow\infty)
$$
**用 $G_t$ 来衡量一个策略的好坏，$G_t$ 大的策略就好，反之。**

但是使用 $G_t$ 也不能很好的衡量策略的好坏，比如一个最大的问题是，在给定一个状态后，选择一个确定的action后（这里还是在随机策略的角度），进入的下一个状态也是随机的。如下图所示：把左侧的过程放大，只给出a3下的随机状态，a1，a2也是有同样的情况，这里胜率。

![](https://gitee.com/YJLAugus/pic-go/raw/master/img/Gtlim.svg)



举个例子，就像我们给一盆花浇水，水是多了还是少了，对于这盆花来说我们是不得知的，可能会少，也可能会多。这个花的状态变化也就是随机的了。从上面的例子得知，如果还是用 $G_t$ 来对一个策略来进行评估的话，至少有9中情况（随机策略，3个action选一种）。

$G_t$ 只能评估的是一个“分叉”而已（图中`绿色分支`）。而不能更好的进行评估。如下图所示：

![](https://gitee.com/YJLAugus/pic-go/raw/master/img/merch.svg)

因为回报不能很好的对策略进行一个评估，由此引入了另外一个概念——价值函数。

#### 价值函数（Value Function ）

在指定一个状态 $s$ ，采取一个`随机策略 `$\pi$ ，然后加权平均，以上图为例，把9 个分叉($G_t$)加权平均。也就是**期望** $E$。故得出价值函数：
$$
\large
V_\pi(s) = E_\pi[G_t\mid S_t = s]
$$


### Summary

![image-20201128110205095](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201128110205095.png)

## MDP贝尔曼期望方程

### 价值函数分类

上面提到的的价值函数其实是其中的一种，确切的可以称为 `状态价值函数`，**用$v_\pi(s)$ 来表示**只和状态有关系。初次之外还有另外一种价值函数，那就是`状态动作价值函数`，用$q_\pi(s,a)$这里引入的`action`。故价值函数可以分为下面的两种:
$$
Value \quad Function = \begin{cases} v_\pi(s) = E_\pi[G_t\mid S_t = s], & \text {only $s$ is independent variable} \\ q_\pi(s,a) = E_\pi[G_t\mid S_t = s,A_t = a], & \text{Both $s$ and a are independent variable} \end{cases}
$$
从上面的公式中，我们可以得知，在 $v_\pi(s)$ 中，只有 $s$ 是自变量，一个 $\pi$ 其实就是一个状态$s$ 和一个action的一个映射。故，只要$\pi$ 确定了，那么$s,a$ 也就确定了，即此时的 $\pi$ 对状态 $s$ 是有限制作用的。但是，在 $q_\pi(s,a)$ 中，子变量为$s,a$ 两个，这两个自变量之间是没有特定的关系的。也就是说，$s$和$a$ 都在变，无法确定一个映射(策略) $\pi$ ,那么也就是说在 $q_\pi$ 中的$\pi$ 对于$s$ 是没有约束的。

### 两种价值函数之间的关系

#### $v_\pi(s)$ 和 $q_\pi(s,a)$ 之间的关系

![](https://gitee.com/YJLAugus/pic-go/raw/master/img/Gtlim.svg)



还是以上图为例，对于 s 状态，在随机策略中有三种 action 选择，分别是 $\pi(a_1 \mid s)$，$\pi(a_1 \mid s)$，$\pi(a_1 \mid s)$，三种action(行为)对应的价值函数（此时为动作价值函数）为 $q_\pi(s,a_1)$， $q_\pi(s,a_2)$， $q_\pi(s,a_3)$。那么此时的 $v_\pi(s)$ 就等于各个action的动作状态价值函数的加和，即：
$$
v_\pi(s) = \pi(a_1 \mid s)·q_\pi(s,a_1) +  \pi(a_2 \mid s)·q_\pi(s,a_2) +  \pi(a_3 \mid s)·q_\pi(s,a_3)
$$
这样一来我们就得出了 **$v_\pi(s)$ 和 $q_\pi(s,a)$ 之间的关系**，若条件已知，就可以直接计算出 $v_\pi$。
$$
\large
v_\pi(s) = \sum_{a\in A} \pi(a\mid s) ·q_\pi(s,a)
$$
对于某个状态 $s$ 来说，$v_\pi \leq \underset{a}{max}\ q_\pi(s,a)$ ，$v_\pi(s)$ 是一个加权平均，实际上就是一个平均值，当然要小于等于$\ q_\pi(s,a)$的最大值。$v_\pi(s)$只有全部是最大值的时候，两者才有可能相等。比如 5，5，5，平均值是5，最大值也是5；3，4，5而言，平均值为4，但是最大值为5。注意的是，4是乘上权值后的值，换句话说也就是乘了一个概率（$\pi(a\mid s)$）。

#### $q_\pi(s,a)$ 和 $v_\pi(s')$  之间的关系

从下面图中可得，在 $q_\pi(s,a)$ 位置，（一个action）状态转移只能向“箭头”方向转移，而不能向上。如果想从下面的状态转移到上面的状态，那必须还要另外一个action。情况是一样的，就按下图来说明，经过a3后到达状态s'，此时的状态函数就是 $v_\pi(s’）$。

![image-20201128134834753](https://gitee.com/YJLAugus/pic-go/raw/master/img/hss01.svg)

上面的图可知： 在确定了s 后，由随机策略action，引起“分叉”，同理，以a3为例，因为系统状态转移的随机性，也会引起分叉，也就是 s' 的状态也是不确定的。还有一点 r 也又不确定性，如下图蓝色虚线部分。

![image-20201128135658979](https://gitee.com/YJLAugus/pic-go/raw/master/img/hss02.svg)

由我们前面提到的公式也可得知：s' 和 r 都是随机的。比如说s，a ，s' 都是给定的，r也是不确定的。
$$
\large p(s',r\mid s,a) \dot{=} Pr\lbrace S_{t+1}=s',R_{t+1} = r \mid S_{t} =s,A_{t}=a \rbrace
$$
这样一来，可得**一条蓝色通路的回报**：
$$
\large
q_\pi(s,a) = r + \gamma v_\pi(s')  \quad\quad\quad (1)
$$
(1)式是怎么来的呢？以上图为例，在 $q_\pi(s,a)$ 处往下走，选定一个 r ，再往下到达一个状态s'， 此时在往下还是同样的状态，就是俄罗斯套娃，以此类推。关于其中的 $\gamma v_\pi(s') $ ，来自于 $G_t$。看下面的式子：
$$
\large{
G_t = R_{t+1} + \gamma R_{t+2}+\gamma^2 R_{t+3} \ ...\ +\gamma^{T-t-1}R_T \quad\quad  \gamma\in[0,1],\quad (T\rightarrow\infty) \\
G_t = R_{t+1} + \gamma(R_{t+2}+\gamma R_{t+3}+\gamma^2 R_{t+4}+ ...)
}
$$
因为 $v_\pi(s)$ 来自 $G_t$ ，故类比得（1）式。

因为走每条蓝色的通路也都是由概率的，故我们需要乘上概率，同时累加求和，一个是多条蓝色通路另一个是多个s'。故得：**$q_\pi(s,a)$ 和 $v_\pi(s')$ 之间的关系** 如下：
$$
\large
q_\pi(s,a) =\sum_{s',r}P(s',r \mid s,a)[r+ \gamma v_\pi(s')] \quad\quad\quad (2)
$$


### 贝尔曼期望等式（方程）

这样我们得到两个式子：
$$
\large{
v_\pi(s) = \sum_{a\in A} \pi(a\mid s) ·q_\pi(s,a)  \quad\quad\quad\quad\quad\quad\quad\quad\ (3) \\

q_\pi(s,a) =\sum_{s',r}P(s',r \mid s,a)[r+ \gamma v_\pi(s')] \quad\quad\quad (4)
}
$$
（4）式带入（3）得：
$$
\large
v_\pi(s) = \sum_{a\in A} \pi(a\mid s) \sum_{s',r}P(s',r \mid s,a)[r+ \gamma v_\pi(s')] \quad\quad\quad\quad\quad\quad (5)
$$
（3）式带入（4）得：
$$
\large
q_\pi(s,a) =\sum_{s',r}P(s',r \mid s,a)[r+ \gamma \sum_{a'\in A} \pi(a'\mid s')  ·q_\pi(s',a') ] \quad\quad (6)
$$
关于（6）式可以看下图，更容易理解：

![](https://gitee.com/YJLAugus/pic-go/raw/master/img/hss03.svg)

（5）式和（6）式 被称为**贝尔曼期望方程**。

* 一个实例：

  例子是一个学生学习考试的MDP。里面实心圆位置是**起点**，方框那个位置是**终点**。上面的动作有study, Pub, Facebook, Quit, Sleep，每个状态动作对应的即时奖励R已经标出来了。我们的目标是找到最优的状态动作价值函数或者状态价值函数，进而找出最优的策略。

  <img src="https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201129160656460.png" alt="image-20201129160656460" style="zoom:50%;" />

  为了方便，我们假设衰减因子 $\gamma =1, \pi(a|s) = 0.5$ 。对于终点方框位置，由于其没有下一个状态，也没有当前状态的动作，因此其状态价值函数为0，对于其他的状态（圆圈）按照从上到下，从左到右的顺序定义其状态价值函数分别是 $v_1,v_2,v_3,v_4$ ，根据（5）式 :
  $$
  v_\pi(s) = \sum_{a\in A} \pi(a\mid s) \sum_{s',r}P(s',r \mid s,a)[r+ \gamma v_\pi(s')] \quad\quad\quad\quad\quad\quad (5)
  $$
  对于$v_1$位置，我们有：$v_1 = 0.5*(-1+v_1) +0.5*(0+v_2)$
  
  对于$v_2$位置，我们有：$v_2 = 0.5*(-1+v_1) +0.5*(-2+v_3)$
  
  对于$v_3$位置，我们有：$v_3 = 0.5*(0+0) +0.5*(-2+v_4)$
  
  对于$v_4$位置，我们有：$v_4 = 0.5*(10+0) +0.5*(1+0.2*v_2+0.4*v_3+0.4*v_4)$
  
  解出这个方程组可以得到 $v_1=-2.3, v_2=-1.3, v_3=2.7, v_4=7.4$, 即每个状态的价值函数如下图：

<img src="https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201129162749453.png" alt="image-20201129162749453" style="zoom:50%;" />

> 从上面可以看出，针对一个特定状体，状态价值函数计算都是基于下一个状态而言的，通俗的讲，按照“出箭头”的方向计算当前状态的价值函数。


### Summary

![image-20201128152843746](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201128152843746.png)

## MDP贝尔曼最优方程

### 最优价值函数

能够使得 $v$ 达到最大值的那个 $\pi$ ，这个  $\pi$ 被成为最优策略，进而得到**最优状态价值函数**。同理得到最**优状态动作价值函数**。
$$
\large
\begin{cases} v_*(s)\ \dot{=}\ \ \underset{\pi}{max} \ v_\pi(s) & \text{} \\
q_*(s,a)\ \dot{=}\ \ \underset{\pi}{max} \ q_\pi(s,a) & \text{} & \text {} \end{cases}
$$
记 $\pi_* = \underset{\pi}{argmax} \ v_\pi(s) = \underset{\pi}{argmax} \ q_\pi(s,a)$，含义是 $\pi_*$ 可以使得 $ v_*(s)$达到最大值，同样的，也可以使得 

$q_\pi(s,a)$ 达到最大值。

由以上公式得：
$$
\large
\begin{cases}v_*(s)=\underset{\pi}{max}\ v_\pi(s)= v_{\pi_*}(s) & \text{(7)} \\ 
q_*(s,a)=\underset{\pi}{max}\ q_\pi(s,a)= q_{\pi_*}(s,a) & \text{} & \text {(8)} \end{cases}
$$

> 值得注意的一点是$ v_*(s)$ 强调的是，不管你采用的是什么策略，只要状态价值函数达到最大值，而 $v_{\pi_*}(s)$ 则更为强调的是 $\pi$ ，达到最大的状态价值函数所采取的最优的那个 $\pi$

此时，我们再探讨一下$v_{\pi_*}(s)$ 和 $q_{\pi_*}(s,a)$ 的关系。在贝尔曼期望方程中，我们提到过 $v_\pi(s) \leq \underset{a}{max}\ q_\pi(s,a)$ ，那么在这里是不是也由类似的关系$v_{\pi_*}(s)\leq \underset{a}{max}\ q_\pi(s,a)$  成立？我们知道 $v_{\pi_*}(s)$ 是一种策略，并且是最优的策略，$q_{\pi_*}(s,a)$ 是代表一个“分支”，因为 $v_{\pi_*}(s)$ 是一个加权平均值，但同样的，和$v_\pi(s)$ 不同的是，$v_{\pi_*}(s)$ 是最优策略的加权平均，那么是不是可以把小于号去掉，写成下面的形式：
$$
\large
v_{\pi_*}(s)= \underset{a}{max}\ q_\pi(s,a)
$$


假定 $v_{\pi_*}(s)\leq \underset{a}{max}\ q_\pi(s,a)$ 中的 $\pi_*$ 还是一个普通的策略，那么一定满足  $v_{\pi_*}(s)\leq \underset{a}{max}\ q_\pi(s,a)$ ，这一点我们已经提到过，如果说 $v_{\pi_*}(s)< \underset{a}{max}\ q_\pi(s,a)$ ，说明$v_{\pi_*}(s)$ 还有提高的空间，并不是最优策略，这和条件矛盾了。所以这个小于不成立，得证：$v_{\pi_*}(s)= \underset{a}{max}\ q_\pi(s,a)$ 

详细证明过程如下：

![image-20201129190023306](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201129190023306.png)

其实，上面的式子是由  (3)式 
$$
v_\pi(s) = \sum_{a\in A} \pi(a\mid s) ·q_\pi(s,a)  \quad\quad (3)
$$
演变而来的。$v_{\pi_*}(s)$ 直接取最大值时候和 $\underset{a}{max}\ q_\pi(s,a)$ 的最大值是相等的。也就是此时不用加权平均了，直接是 $v_\pi(a) = q_\pi(s,a)$ 。那么从原先的(4)式能不能也得出相似
$$
q_\pi(s,a) =\sum_{s',r}P(s',r \mid s,a)[r+ \gamma v_\pi(s')] \quad\quad (4)
$$
的结论，把求和符号去掉，直接等于最大值呢？答案是否定的，因为$v_{\pi_*}(s)= \underset{a}{max}\ q_\pi(s,a)$  是作用在`action`上的，在公式中也可以看出，换句话说，我们对于下图的a1，a2，a3这里是可以控制的。但是对于下图中的蓝色虚线部分，系统状态转移是无法控制的。

![image-20201128135658979](https://gitee.com/YJLAugus/pic-go/raw/master/img/hss02.svg)

所以，原先的两组公式（3）、（4）并 结合（7）、（8）
$$
\large{
v_\pi(s) = \sum_{a\in A} \pi(a\mid s) ·q_\pi(s,a)  \quad\quad\quad\quad\quad\quad\quad\quad\ (3) \\

q_\pi(s,a) =\sum_{s',r}P(s',r \mid s,a)[r+ \gamma v_\pi(s')] \quad\quad\quad (4)
}
$$
并进行一个推导，得出另外的两组公式（9）、（10）如下：
$$
\large{
v_*(s)=\underset{a}{max}\ q_*(s,a) \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad (9) \\
q_*(s,a)= \sum_{s',r}P(s',r \mid s,a)[r+\gamma v_*(s')] \quad\quad\quad (10)
}
$$

### 贝尔曼最优方程

（10）式带入（9）式得：
$$
\large{
v_*(s)=\underset{a}{max}\sum_{s',r}P(s',r \mid s,a)[r+\gamma v_\pi(s')] \quad\quad(11) \\


}
$$
（9）式带入（10）式得：
$$
\large
q_*(s,a)= \sum_{s',r}P(s',r \mid s,a)[r+\gamma \underset{a'}{max}\ q_*(s',a') ] \quad\quad (12)
$$
（11）、（12）被称为**贝尔曼最优方程**。

* 一个实例：还是以上面的例子讲解，我们这次以动作价值函数 $q_*(s,a)$ 为例来求解 $v_*(s),q_*(s,a) $

  <img src="https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201129160656460.png" alt="image-20201129160656460" style="zoom:50%;" />

根据（12）式

$$
\large
q_*(s,a)= \sum_{s',r}P(s',r \mid s,a)[r+\gamma \underset{a'}{max}\ q_*(s',a') ] \quad\quad (12)
$$
可得方程组如下：

$$
\large{\begin{align}
q_*(s_4, study) & = 10 \\
q_*(s_4, pub) & = 1 + 0.2 * \underset{a'}{max}q_*(s_2, a') + 0.4 * max_{a'}q_*(s_3, a') + 0.4 * \underset{a'}{max}q_*(s_4, a') \\
q_*(s_3, sleep) & = 0  \\
q_*(s_3, study) & = -2 + \underset{a'}{max}q_*(s_4, a') \\
q_*(s_2, study) & = -2 + \underset{a'}{max}q_*(s_3, a') \\
q_*(s_2, facebook) & = -1 + \underset{a'}{max}q_*(s_1, a') \\
q_*(s_1, facebook) & = -1 + \underset{a'}{max}q_*(s_1, a') \\
q_*(s_1, quit) & = 0 + \underset{a'}{max}q_*(s_2, a')
\end{align}}
$$

然后求出所有的 $q_*(s,a)$，然后再利用 $v_*(s) = \underset{a'}{max}q_*(s,a)$，就可以求出所有的  $v_*(s)$，最终结果如下图所示：

<img src="https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201130141108720.png" alt="image-20201130141108720" style="zoom: 67%;" />

详细的计算过程可以看下视频 的简单分析。https://www.bilibili.com/video/BV1Fi4y157vR/

### Summary

![image-20201129210325397](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201129210325397.png)

## 参考文献

https://www.bilibili.com/video/BV1RA411q7wt

https://www.cnblogs.com/pinard/p/9426283.html

https://www.davidsilver.uk/wp-content/uploads/2020/03/MDP.pdf

https://www.cnblogs.com/jsfantasy/p/jsfantasy.html