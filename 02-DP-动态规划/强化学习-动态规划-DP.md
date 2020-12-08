

# 强化学习-动态规划-DP

> 作者：YJLAugus  博客： https://www.cnblogs.com/yjlaugus

这里讲动态规划，主要是用动态规划来解决MDP的中最优的策略，并得出最优的价值函数。这节主要介绍两中动态规划的方法，一个是**策略迭代**方法，另一个是**价值迭代**方法。

## 策略迭代

策略迭代主要分两个，一个是**策略评估**，一个是**策略改进**。定义比较简单，分别如下：

**策略评估**：给定任意的一个 $\pi$ ，求出 状态价值函数 $v_\pi(s)$ 或者状态动作价值函数 $q_\pi(s,a)$。
**策略改进**：依据策略评估得出来的 $v_\pi(s)$ 或者 $q_\pi(s,a)$，从其中构造出一个新的 $\pi'$ ，使得 $\pi' \geq \pi $ ，这样就完成了一次迭代。
 * 换句话说，就是在每个状态下根据$v_\pi(s)$ 或者 $q_\pi(s,a)$选择一个最优的策略，这个策略就是 $\pi'$ 。
 * 这种根据`原策略的价值函数`执行贪心算法，来构造一个更好策略的过程，我们称为策略改进。
**策略迭代**： 就是递归以上两个过程，以上面的例子，得到$\pi'$的价值函数，然后再以$\pi'$的价值函数构造一个新的$\pi''$ ，不断迭代。

### 策略评估-解析解

策略评估，就是已知 MDP，“已知MDP”的意思是知道其动态特性，也就是 $p(s',r \mid s,a)$，给定$\pi$ ，求$v_\pi(s)$。

已知$(\forall s\in S)$ ，即有多少个$s$ 就会有多少$v_\pi(s)$，可得列向量如下：
$$
\large
v_\pi(s)= \begin{pmatrix}
v_\pi(s_1)\\
v_\pi(s_2)\\
\vdots    \\
v_\pi(s_{\lvert S \rvert})

\end{pmatrix} _{\lvert S \rvert \times 1}
$$



上节课我们得到价值函数的公式，并进一步推导得到贝尔曼期望方程：
$$
\large
{\begin{align}  
v_\pi(s) =& E_\pi[G_t \mid S_t = s] \\
=& E_\pi[R_{t+1} + \gamma v_\pi(S_{t+1} \mid S_t=s)]\\
=& v_\pi(s) = \sum_{a\in A} \pi(a\mid s) \sum_{s',r}p(s',r \mid s,a)[r+ \gamma v_\pi(s')] 
\end{align}
}
$$
从上面的列向量中得知，如果我们想要求整个的$v_\pi(s)$ ，就是把$v_\pi(s_1),v_\pi(s_2),v_\pi(s_3)...$ 等所有的价值函数都要求出来。我们这节的目的就是把公式中的累加符号去掉，得到另一个更为简单的式子。接下来我们对下面的式子进行化简如下：
$$
\large{
\begin{align} 

v_\pi(s) =& \sum_{a\in A} \pi(a\mid s) \sum_{s',r}p(s',r \mid s,a)[r+ \gamma v_\pi(s')]\\
= & \underbrace {\sum_{a\in A} \pi(a\mid s) \sum_{s',r}p(s',r \mid s,a) \cdot r }_{①}
+ 
\underbrace{\gamma \sum_{a\in A} \pi(a\mid s) \sum_{s',r}p(s',r \mid s,a) \cdot v_\pi(s')}_{②}


\end{align}
}
$$
**分解①式**：对于①式而言，$s'$ 只出现在函数$p(s',r \mid s,a)$ 中，是可以积分积掉的。故由①得：
$$
\large{
\begin{align}

\sum_{a\in A} \pi(a\mid s) \sum_{s',r}p(s',r \mid s,a) \cdot r  \\
=& \sum_{a\in A}  \pi(a\mid s) \underbrace {\sum_{r}  r\cdot p(r \mid s,a)}_{E_\pi[R_{t+1}  \mid S_t=s,A_t=a]} \quad\quad (1)  \\


\end{align}
}
$$
（1）式正好是期望，此时我们也定义一个函数（记号） $r(s,a) \dot{=} E_\pi[R_{t+1}  \mid S_t=s,A_t=a]$ 。表示$s,a$ 这个二元组的收益。故我们的推导变成如下：
$$
\large{
\begin{align}

\sum_{a\in A} \pi(a\mid s) \sum_{s',r}p(s',r \mid s,a) \cdot r  \\
=& \sum_{a\in A} \pi(a\mid s) \underbrace {\sum_{r}  r\cdot p(r \mid s,a)}_{r(s,a) \dot{=}E_\pi[R_{t+1}  \mid S_t=s,A_t=a]} \quad\quad (1)  \\
=& \sum_{a\in A} \pi(a\mid s) \cdot r(s,a) \quad\quad\quad\quad \quad\quad \quad\quad\ (2)



\end{align}
}
$$
（2）式中，如果把 $a$ 都积分掉，那么（2）式就和$a$ 就没关系了，只和$s$ 有关，这里我们再次引入一个记号$r_\pi(s)$用来表示这个新的关系如下：
$$
\large{
\begin{align}

\sum_{a\in A} \pi(a\mid s) \sum_{s',r}p(s',r \mid s,a) \cdot r  \\
=& \sum_{a\in A} \pi(a\mid s) \underbrace {\sum_{r}  r\cdot p(r \mid s,a)}_{r(s,a) \dot{=}E_\pi[R_{t+1}  \mid S_t=s,A_t=a]} \quad\quad (1)  \\
=& \sum_{a\in A} \pi(a\mid s) \cdot r(s,a) \quad\quad\quad\quad \quad\quad \quad\quad\ (2) \\
\dot{=}&r_\pi(s)  \quad\quad\quad\quad \quad\quad \quad\quad\quad \quad\quad\ \quad\quad \quad\quad\ (3)



\end{align}
}
$$
此时，①式化简至此，需要提到的一点是，$r_\pi(s)$ 应该由多少个呢？在一开始我们就提到过 “$S$中有多少个$s$ 就会有多少$v_\pi(s)$，” ，显然 $r_\pi(s)$ 的数量也是 $\lvert S \rvert$ 个。故可得列向量如下:
$$
\large
r_\pi(s)= \begin{pmatrix}
r_\pi(s_1)\\
r_\pi(s_2)\\
\vdots    \\
r_\pi(s_{\lvert S \rvert})

\end{pmatrix} _{\lvert S \rvert \times 1}
$$
**分解②式**：在 $p(s',r \mid s,a) \cdot v_\pi(s')$ 中，我们发现 $s'$ 在式子中都存在，但是$r$ 只存在于函数 $p(s',r \mid s,a)$ 中，故积分可以积掉，则②式可以进一步写成：
$$
\large{
\begin{align}

\gamma \sum_{a\in A} \pi(a\mid s) \sum_{s',r}p(s',r \mid s,a) \cdot v_\pi(s') 
=& \gamma \sum_{a\in A} \pi(a\mid s) \sum_{s'}\underbrace{p(s' \mid s,a)}_{状态转移函数} \cdot v_\pi(s') \quad\quad (4)


\end{align}
}
$$
在（4）式中，可以发现在两个累加号中都有 $a$（注：虽然也都有 $s$，但是累加号下面是$a$ ，故积分只能积掉 $a$），积分可以积掉，但是一定和 $s,s'$ 有关，故引出一个记号 $P_\pi(s,s')$ 来表示。 故继续推导如下：
$$
\large{
\begin{align}

\gamma \sum_{a\in A} \pi(a\mid s) \sum_{s',r}p(s',r \mid s,a) \cdot v_\pi(s') 
=& \gamma \sum_{a\in A} \pi(a\mid s) \sum_{s'}\underbrace{p(s' \mid s,a)}_{状态转移函数} \cdot v_\pi(s') \quad\quad (4) \\


=& \gamma \sum_{s'} \underbrace{\sum_{a\in A} \pi(a\mid s) p(s' \mid s,a)}_{P_\pi(s,s')} \cdot v_\pi(s') \quad\quad (5) \\

=&  \gamma \sum_{s'} P_\pi(s,s') \cdot v_\pi(s') \quad\quad (6)\\


\end{align}
}
$$
由（6）式可知，对于 $P_\pi(s,s')$ 来说，一共由多少个值呢？很显然，$s$ 和 $s'$ 分别由 $\lvert S \rvert $ 个值，故 $P_\pi(s,s')$ 会有 $\lvert S \rvert \times \lvert S \rvert  $  个。故可得矩阵如下：
$$
\large
P_\pi(s,s')= \begin{pmatrix}
P_\pi(s_1,s'_1) & \cdots & P_\pi(s_1,s'_{\lvert S \rvert})\\
\vdots & \ddots & \vdots \\
P_\pi(s_{\lvert S \rvert},s'_1) & \cdots & P_\pi(s_{\lvert S \rvert},s'_{\lvert S \rvert})\\

\end{pmatrix} _{\lvert S \rvert \times \lvert S \rvert}
$$
于是，结合（3）和（6）式得：
$$
\large{
v_\pi(s) = r_\pi(s) + \gamma \sum_{s'} P_\pi(s,s') \cdot v_\pi(s') \quad\quad\quad (7)
}
$$
令$s_i\dot{=s},s_j \dot{=} s'$ ，则（7） 式得：
$$
\large{
v_\pi(s) = r_\pi(s) + \gamma \underbrace{\sum_{j=1}^{\lvert S \rvert} P_\pi(s_i,s_j) }_{①}\cdot v_\pi(s') \quad\quad\quad (7)
}
$$
由（7）式中的①式可得，正好是矩阵$P_\pi(s,s')$ ，其中的 $P_\pi(s_i,s_j)$正是矩阵的一行。（7）式中$v_\pi(s)，r_\pi(s)，①，v_\pi(s')$ 式（$v_\pi(s')$本质是和$v_\pi(s)$是一样的），都分别对应一个矩阵，也即：
$$
\begin{pmatrix}
v_\pi(s_1)\\
v_\pi(s_2)\\
\vdots    \\
v_\pi(s_{\lvert S \rvert})
\end{pmatrix} _{\lvert S \rvert \times 1}
=

\begin{pmatrix}
r_\pi(s_1)\\
r_\pi(s_2)\\
\vdots    \\
r_\pi(s_{\lvert S \rvert})
\end{pmatrix} _{\lvert S \rvert \times 1}
+ \gamma

\begin{pmatrix}
P_\pi(s_1,s'_1) & \cdots & P_\pi(s_1,s'_{\lvert S \rvert})\\
\vdots & \ddots & \vdots \\
P_\pi(s_{\lvert S \rvert},s'_1) & \cdots & P_\pi(s_{\lvert S \rvert},s'_{\lvert S \rvert})\\
\end{pmatrix} _{\lvert S \rvert \times \lvert S \rvert}
\cdot 

\begin{pmatrix}
v_\pi(s_1)\\
v_\pi(s_2)\\
\vdots    \\
v_\pi(s_{\lvert S \rvert})
\end{pmatrix} _{\lvert S \rvert \times 1}
$$
最终化简到矩阵的运算。故，由（7）式进一步得：
$$
\large{
\begin{align}
v_\pi(s) =& r_\pi(s) + \gamma P_\pi(s,s') \cdot v_\pi(s') \quad\quad\quad \\
v_\pi=& r_\pi + \gamma P_\pi \cdot v_\pi \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\qquad (8)\\
(I-P_\pi)v_\pi =& r_\pi \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\qquad\quad\quad\quad\quad\quad\ (9) \\
v_\pi =& (I-P_\pi)^{-1} r_\pi \qquad\qquad\qquad\qquad\qquad\qquad(10)
\end{align}
}
$$
经过以上推导，得出（10）式。其中在（8）式中$v_\pi(s')$ 其实是 $v_\pi(s)$ 的迭代，所以直接用 $v_\pi(s)$ 替换，在（9）式中$\gamma$ 是个常数，也可忽略。从向量中可以得出复杂度$O({\lvert S \rvert}^3)$

### Summary

![image-20201203135011156](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201203135011156.png)

### 策略评估-迭代解

首先，我们来看如何使用动态规划来求解强化学习的**预测问题**，即求解给定策略的状态价值函数的问题。这个问题的求解过程我们通常叫做策略评估(Policy Evaluation)。

策略评估的基本思路是从**任意一个状态价值函数开始**，依据给定的策略，结合贝尔曼期望方程、状态转移概率和奖励同步迭代更新状态价值函数，直至其**收敛**，得到该策略下最终的状态价值函数。

假设我们在第$k$轮迭代已经计算出了所有的状态的状态价值，那么在第$k+1$轮我们可以利用第$k$轮计算出的**状态价值**计算出第$k+1$轮的状态价值。这是通过贝尔曼方程来完成的，即：
$$
\large{
v_{k+1}(s) = \sum\limits_{a \in A} \pi(a|s)(R_s^a + \gamma \sum\limits_{s' \in S}P_{ss'}^av_{k}(s'))
 \\

 v_{k+1}(s) \dot{=}\sum_{a\in A} \pi(a\mid s) \sum_{s',r}p(s',r \mid s,a)[r+ \gamma v_k(s')]
}
$$
和上一节的式子唯一的区别是由于我们的策略$\pi$已经给定，我们不再写出，对应加上了迭代轮数的下标。我们每一轮可以对计算得到的新的状态价值函数再次进行迭代，直至状态价值的值改变**很小(收敛)**，那么我们就得出了预测问题的解，即给定策略的状态价值函数$v_\pi$。

### Summary

![image-20201203183436771](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201203183436771.png)

### 策略评估-（迭代解）应用

下面我们用一个具体的例子来说明策略评估的过程。

![image-20201203142402216](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201203142402216.png)



这是一个经典的Grid World的例子。我们有一个`4x4`的16宫格。只有左上和右下的格子是终止格子。该位置的价值固定为0，个体如果到达了该2个格子，则停止移动，此后每轮奖励都是0。个体在16宫格其他格的每次移动，得到的即时奖励R都是-1。注意个体每次只能移动一个格子，且只能上下左右4种移动选择，不能斜着走, 如果在边界格往外走，则会直接移动回到之前的边界格。衰减因子我们定义为$\gamma =1$。由于这里每次移动，下一格都是固定的，因此所有可行的的状态转化概率$P =1$。这里给定的策略是随机策略，即每个格子里有25%的概率向周围的4个格子移动。

![image-20201203142716568](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201203142716568.png)

首先我们初始化所有格子的状态价值为0，如上图$k=0$的时候。现在我们开始策略迭代了。由于终止格子的价值固定为0，我们可以不将其加入迭代过程。

**在$k=1$时**，我们利用上面的贝尔曼方程先计算第二行第一个格子的价值：
$$
v_1^{(21)} = \frac{1}{4}[(-1+0) +(-1+0)+(-1+0)+(-1+0)] = -1
$$
第二行第二个格子的价值是：
$$
v_1^{(22)} = \frac{1}{4}[(-1+0) +(-1+0)+(-1+0)+(-1+0)] = -1
$$
其他的格子都是类似的，第一轮的状态价值迭代的结果如上图$k=1$的时候。现在我们第一轮迭代完了。

**在$k=1$时**，还是看第二行第一个格子的价值：
$$
v_2^{(21)} = \frac{1}{4}[(-1+0) +(-1-1)+(-1-1)+(-1-1)] = -1.75
$$
第二行第二个格子的价值是：
$$
v_2^{(22)} = \frac{1}{4}[(-1-1) +(-1-1)+(-1-1)+(-1-1)] = -2
$$
最终得到的结果是上图$k=2$的时候，第二轮迭代完毕。

![image-20201203142745214](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201203142745214.png)

**在$k=3$时**：
$$
v_3^{(21)} = \frac{1}{4}[(-1+0)+(-1-1.7) +(-1-2)+(-1-2)] = -2.425 \\
v_3^{(22)} = \frac{1}{4}[(-1-1.7) +(-1-1.7)+(-1-2)+(-1-2)] = -2.85
$$

> 计算加和的过程 就是 上、下、左、右四个方向，其中无论哪次迭代，都有 $v_k^{11} = 0$。

最终得到的结果是上图$k=3$的时候。就这样一直迭代下去，直到每个格子的策略价值改变很小（收敛）为止。这时我们就得到了所有格子的基于随机策略的状态价值。

可以看到，动态规划的策略评估计算过程并不复杂，但是如果我们的问题是一个非常复杂的模型的话，这个计算量还是非常大的。

### 策略改进-策略改进定理

现在出现了这样一个问题，如果给定一个 $\pi$ 和 $\pi'$ ，我们如何判断哪一策略更好呢？采用我们以前提到的方法，就是分别计算各自对应的价值函数，然后通过判断两个价值函数的大小来判断策略的好坏。虽然能够得出结论，但是这个计算的过程也是会占用 “资源”的，能否有另外一种方式可以实现相应的功能呢？有，那就是**策略改进定理。**

**策略改进定理：** **给定$\pi，\pi'$，如果$\forall(s) \in S,q_\pi(s,\pi'(s)) \geq v_\pi(s)$，那么，则有 $\forall s\in S,v_{\pi'}(s) \geq v_\pi(s)$。**

通过上面的定理可得，$q_\pi(s,a)$ 只要算出来了，那么 $v_\pi(s)$ 自然也会得出（$v_\pi(s)$ 是$q_\pi(s,a)$的加权平均），此时我们不再需要再求得 $v_{\pi'}(s)$，而是直接把$\pi'(s)$ 带入到 $q_\pi(s,a)$ 中的a中即可。

下面进行定理的一个证明：

在 $q_\pi(s,\pi'(s)) \geq v_\pi(s)$ 中，里面的 $q_\pi(s,a)$ 和 $v_\pi(s')$ 有如下关系，我们在第一张MDP已经证明过：
$$
\large{
\begin{align}
q_\pi(s,a) =& \sum_{s',r}P(s',r \mid s,a)[r+ \gamma v_\pi(s')] \\
=&E_\pi[R_{t+1} + \gamma v_\pi(s_{t+1}) \mid S_t = s, A_t=a]
\end{align}
}
$$
其实，这里需要说明一点，上式中 $E_\pi$ 中的$\pi$ 严格意义上是不能带着的，对于$R_{t+1}$ ， 不是$\pi$ 控制的，详情看下图的蓝色虚线（在第一章中，我们称其未系统之间的状态转移），

![image-20201128135658979](https://gitee.com/YJLAugus/pic-go/raw/master/img/hss02.svg)

当然了，对于 $R_{t+2}, R_{t+3}$ 等是需要 $\pi$ 控制的，在公式中的体现就是$v_\pi$。故，这里准确写法应该如下：
$$
\large{
\begin{align}
q_\pi(s,a) =& \sum_{s',r}P(s',r \mid s,a)[r+ \gamma v_\pi(s')] \\
=& E[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t=a]
\end{align}
}
$$
**证明：**$\forall(s) \in S,q_\pi(s,\pi'(s)) \geq v_\pi(s)$，把 $a=\pi'(s)$ 带入得：
$$
\large{
\begin{align}
v_\pi(s) \leq & q_\pi(s,\pi'(s)) \\
=& E[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t=\pi'(s)] \quad\quad\quad(1) \\
=& E_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] \quad\quad\quad\quad\quad\quad\quad\quad (2)


\end{align}
}
$$
上面式子中，在（2）式中，我们把 $\pi'$那个策略定义到 $R_{t+1}$ 上，其余的还是采取 $\pi$ 的策略。在（2）式中出现了 $v_\pi(s_{t+1})$ ，再带入$v_\pi(s) \leq q_\pi(s,\pi'(s))$  得：
$$
\large{
\begin{align}
v_\pi(s) \leq & q_\pi(s,\pi'(s)) \\
=& E[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t=\pi'(s)] \quad\quad\quad(1) \\
=& E_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] \quad\quad\quad\quad\quad\quad\quad\quad\ (2) \\

\leq & E_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1},\pi'(S_{t+1})) \mid S_t = s] \quad\quad\quad\quad\quad\ (3)


\end{align}
}
$$
再把$q_\pi(s_{t+1},\pi'(s_{t+1}))$ 带入（2）式得：
$$
\large{
\begin{align}
v_\pi(s) \leq & q_\pi(s,\pi'(s)) \\
=& E[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t=\pi'(s)] \quad\quad\quad(1) \\
=& E_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] \quad\quad\quad\quad\quad\quad\quad\quad\ (2) \\

\leq & E_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1},\pi'(S_{t+1})) \mid S_t = s] \quad\quad\quad\quad\quad\ (3) \\

=&E_{\pi'}[R_{t+1} + \gamma E_{\pi'}[R_{t+2} + \gamma v_\pi(S_{t+2}) \mid S_{t+1}] \mid S_t=s ]\quad\quad\ (4)


\end{align}
}
$$
有一点注意，在（4）式中 $S_{t+1}$ 没有写成等于多少的形式，这里只是对于$S_{t+2}$ 的前提条件，表明其不能独立出现。然后再对（4）展开得：
$$
\large{
\begin{align}
v_\pi(s) \leq & q_\pi(s,\pi'(s)) \\
=& E[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t=\pi'(s)] \quad\quad\quad(1) \\
=& E_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] \quad\quad\quad\quad\quad\quad\quad\quad\ (2) \\

\leq & E_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1},\pi'(S_{t+1})) \mid S_t = s] \quad\quad\quad\quad\quad\ (3) \\

=&E_{\pi'}[R_{t+1} + \gamma E_{\pi'}[R_{t+2} + \gamma v_\pi(S_{t+2}) \mid S_{t+1}] \mid S_t=s ]\quad\quad\ (4) \\

=&E_{\pi'}[R_{t+1} + \gamma E_{\pi'}[R_{t+2} \mid S_{t+1}] + \gamma^2 E_{\pi'}[v_\pi(S_{t+2}) \mid S_{t+1}] \mid S_t=s ]\quad\quad\ (5)


\end{align}
}
$$
对于（5）式，出现了“期望套期望”，期望的期望还是期望，并且都是 $E{\pi'}$，故（5）式可以进一步化简得：
$$
\large{
\begin{align}
v_\pi(s) \leq & q_\pi(s,\pi'(s)) \\
=& E[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t=\pi'(s)] \quad\quad\quad(1) \\
=& E_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] \quad\quad\quad\quad\quad\quad\quad\quad\ (2) \\

\leq & E_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1},\pi'(S_{t+1})) \mid S_t = s] \quad\quad\quad\quad\quad\ (3) \\

=&E_{\pi'}[R_{t+1} + \gamma E_{\pi'}[R_{t+2} + \gamma v_\pi(S_{t+2}) \mid S_{t+1}] \mid S_t=s ]\quad\quad\ (4) \\

=&E_{\pi'}[R_{t+1} + \gamma E_{\pi'}[R_{t+2} \mid S_{t+1}] + \gamma^2 E_{\pi'}[v_\pi(S_{t+2}) \mid S_{t+1}] \mid S_t=s ]\quad\quad\ (5) \\

=&E_{\pi'}[R_{t+1} + \gamma R_{t+2}  + \gamma^2v_\pi(S_{t+2}) \mid S_t=s ]\quad\quad\ (6)


\end{align}
}
$$
到这里，只走了两步，可以看到，由 $\pi'$ 一开始的只控制 $R_{t+1}$ ，走完两步后，$R_{t+2}$ 也属于 $\pi'$ 控制，接下来继续走的话，$\pi'$ 控制的 $R$ 会越来越多，$\pi$ 控制的 $R$ 越来越少。
$$
\large{
\begin{align}
v_\pi(s) \leq & q_\pi(s,\pi'(s)) \\
=& E[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t=\pi'(s)] \quad\quad\quad(1) \\
=& E_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s] \quad\quad\quad\quad\quad\quad\quad\quad\ (2) \\

\leq & E_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1},\pi'(S_{t+1})) \mid S_t = s] \quad\quad\quad\quad\quad\ (3) \\

=& E_{\pi'}[R_{t+1} + \gamma E_{\pi'}[R_{t+2} + \gamma v_\pi(S_{t+2}) \mid S_{t+1}] \mid S_t=s ]\quad\quad\ (4) \\

=& E_{\pi'}[R_{t+1} + \gamma E_{\pi'}[R_{t+2} \mid S_{t+1}] + \gamma^2 E_{\pi'}[v_\pi(S_{t+2}) \mid S_{t+1}] \mid S_t=s ]\quad\quad\ (5) \\

=& E_{\pi'}[R_{t+1} + \gamma R_{t+2}  + \gamma^2v_\pi(S_{t+2}) \mid S_t=s ]\quad\quad\ (6)\\

\leq & E_{\pi'}[R_{t+1} + \gamma R_{t+2}  +\gamma^2 R_{t+3} + \gamma^3v_\pi(S_{t+3}) \mid S_t=s ]\quad\quad\quad (7)\\

\vdots \\

\leq & E_{\pi'} \underbrace{[R_{t+1} + \gamma R_{t+2}  +\gamma^2 R_{t+3} + \gamma^4 R_{t+4} + \cdots}_{G_t} \mid S_t=s ] \quad\quad\ (8)\\ 

=& v_{\pi'}(s)

\end{align}
}
$$
故得到：$v_\pi'(s) \geq v_\pi(s)$， 得证。

### Summary

![image-20201203210853298](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201203210853298.png)

### 策略改进-贪心策略

这节就是利用策略改进定理提出一种策略改进的方法——**贪心策略**（Greedy Policy）。对于 $\forall s \in S$，定义如下公式：
$$
\large{
\pi'(s) = \underset{a}{argmax}\ q_\pi(s,a) \quad\quad (1)
}
$$
上面式子的意识是说，根据我们上节课所说的，从一个策略$\pi$，经过策略评估得到一些 $\pi'$ ，如下图所示。并从这些 $\pi'$ 中选择一个最大的$q_\pi(s,a)$。$\underset{a}{argmax}$ 表示能够使得表达式的值最大化的 $a$。

![](https://gitee.com/YJLAugus/pic-go/raw/master/img/pe.svg)



由在第一章讲到的 $v_\pi(s)$ 和 $q_\pi(s,a)$ 的 关系得：$v_\pi(s) \leq \underset{a}max \ q_\pi(s,a)$ 。又因由（1）式可得， $q_\pi(s,\pi'(s)) = \underset{a}max \ q_\pi(s,a)$ 故得：
$$
\large{
v_\pi(s) \leq \underset{a}max \ q_\pi(s,a) =q_\pi(s,\pi'(s)) \\
v_\pi(s) \leq q_\pi(s,\pi'(s)) \quad\quad\quad \quad \qquad\quad\quad \qquad\quad\quad
(2)
}
$$
（2）式正好满足上节我们说的策略改进定理。**故由策略定理得知：对于 $\forall s \in S$，$v_{\pi'}(s) \geq v_\pi(s)$。**

如果在某一个时刻，一直迭代，如果出现了 $ v_\pi(s)=v_{\pi'}(s)$ ，这种情况，也就是说明 $v_\pi(s)$ 此时已经不能再好了，并且此时$ v_\pi(s)=v_{\pi'}(s) =v_*$。如下图所示：

![](https://gitee.com/YJLAugus/pic-go/raw/master/img/veq.svg)



> **证明**：如果$v_{\pi'}=v_\pi$ 那么，$ v_\pi(s)=v_{\pi'}(s) =v_*$

证：由 $v_{\pi'}=v_\pi$， 可以得出 $q_{\pi'}=q_\pi。$$\forall s\in S$，由$v_\pi(s) = \sum_{a\in A} \pi(a\mid s) ·q_\pi(s,a) $可得：
$$
\large{
\begin{align}
v_{\pi'}(s) =& \sum_{a\in A} {\pi'}(a\mid s) ·q_{\pi'}(s,a)\\
=& \sum_{a\in A} {\pi'}(a\mid s) ·q_{\pi}(s,a) \quad\quad\quad\ (3) \\  

\end{align}
}
$$
因为前面说的 $\pi'$ 都是选择一个最优的策略，也就是**确定性策略**，如下图所示：

![image-20201128135658979](https://gitee.com/YJLAugus/pic-go/raw/master/img/hss02.svg)

假如我们选择`a3`这个策略为 $\pi'$ 。故在（3）式中的加和可以去掉了，因为是确定性策略，那么选择`a2` 和 `a1`的概率就是0，例缩当然加和后只剩下`a3` 这个策略$\pi'$，故继续推导得：
$$
\large{
\begin{align}
v_{\pi'}(s) =& \sum_{a\in A} {\pi'}(a\mid s) ·q_{\pi'}(s,a)\\
=& \sum_{a\in A} {\pi'}(a\mid s) ·q_{\pi}(s,a) \quad\quad\quad\ (3) \\
=& q_\pi(s,\pi'(s)) \quad\quad\quad\quad\quad\quad\quad\quad\ (4)  \\

\end{align}
}
$$
由（4）式和（2）中$\underset{a}max \ q_\pi(s,a) =q_\pi(s,\pi'(s))$ 得：
$$
\large{
\begin{align}
v_{\pi'}(s) =& \sum_{a\in A} {\pi'}(a\mid s) ·q_{\pi'}(s,a)\\
=& \sum_{a\in A} {\pi'}(a\mid s) ·q_{\pi}(s,a) \quad\quad\quad\ (3) \\
=& q_\pi(s,\pi'(s)) \quad\quad\quad\quad\quad\quad\quad\quad\quad (4)  \\
=& \underset{a}max \ q_\pi(s,a) \qquad\quad\quad\quad\quad\quad\quad\ (5)\\

\end{align}
}
$$
（5）式再由$q_\pi(s,a) =\sum_{s',r}P(s',r \mid s,a)[r+ \gamma v_\pi(s')] $ 可得：
$$
\large{
\begin{align}
v_{\pi'}(s) =& \sum_{a\in A} {\pi'}(a\mid s) ·q_{\pi'}(s,a)\\
=& \sum_{a\in A} {\pi'}(a\mid s) ·q_{\pi}(s,a) \quad\quad\quad\ (3) \\
=& q_\pi(s,\pi'(s)) \quad\quad\quad\quad\quad\quad\quad\quad\quad (4)  \\
=& \underset{a}max \ q_\pi(s,a) \qquad\quad\quad\quad\quad\quad\quad\ (5)\\
=& \underset{a}max \sum_{s',r}P(s',r \mid s,a)[r+ \gamma v_\pi(s')] \quad (6) \\
\end{align}
}
$$
在（6）式中，又因为题设中，$v_{\pi'}=v_\pi$ 故带入得：
$$
\large{
\begin{align}
v_{\pi'}(s) =& \sum_{a\in A} {\pi'}(a\mid s) ·q_{\pi'}(s,a)\\
=& \sum_{a\in A} {\pi'}(a\mid s) ·q_{\pi}(s,a) \quad\quad\quad\ (3) \\
=& q_\pi(s,\pi'(s)) \quad\quad\quad\quad\quad\quad\quad\quad\quad (4)  \\
=& \underset{a}max \ q_\pi(s,a) \qquad\quad\quad\quad\quad\quad\quad\ (5)\\
=& \underset{a}max \sum_{s',r}P(s',r \mid s,a)[r+ \gamma v_\pi(s')] \quad (6) \\
=& \underset{a}max \sum_{s',r}P(s',r \mid s,a)[r+ \gamma v_{\pi’}(s')] \quad (7)
\end{align}
}
$$
故此时我们得到：
$$
\large
v_{\pi'}(s) = \underset{a}max \sum_{s',r}P(s',r \mid s,a)[r+ \gamma v_{\pi’}(s')] \quad(8)
$$
并且我们由贝尔曼最优方程得：
$$
\large{
v_*(s)=\underset{a}{max}\sum_{s',r}P(s',r \mid s,a)[r+\gamma v_\pi(s')] \quad\quad(9) \\


}
$$
故由（8）式和（9）式，以及题设$v_{\pi'}=v_\pi$ 得证：
$$
 \large
 v_\pi(s)=v_{\pi'}(s) =v_*
$$

### Summary

![image-20201205145118572](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201205145118572.png)

## 价值迭代

### 策略迭代的缺点

![image-20201205150228362](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201205150228362.png)

从上图可以发现，在策略迭代中， 其实进行了两次策略循环，第一层是在策略评估中，第二层是策略迭代这一层。故，如果迭代次数过多的化，其实是很低效率的。**策略评估的一个缺点是每一次迭代过程中都涉及了策略评估。**

### 价值迭代

根据以上， 这里我们讨论一种极端的情况，只计算第一次的价值函数 $v_1(s)$， 从此进行截断，后续不在计算，也即是说，只对 $v_1(s)$ 进行策略评估，并把这个算法称为**价值迭代**。如图：

![](https://gitee.com/YJLAugus/pic-go/raw/master/img/jieduan1.svg)



如下图所示，价值迭代只计算 从状态$s$ 到$s'$的一条分支（$v_1(s)$），假定以最右侧分支为例，这是策略评估的**一步**，加上策略改进后，其实只走了“半步”，也就是从$s$ 状态走到a3（假定选择了action a3），此时利用策略改进算法即可用a3处的$q_\pi(s,a)$ ，并得到$v_\pi(s)$ 和 $v_{\pi'}(s)$的大小关系，以下图为例，假设a3为$v_\pi(s)$，则a2或者a1就是$v_{\pi'}(s)$。

也即：如果按照策略迭代的方法，需要计算完所有v1（选择a1的$v_\pi(s)->v_{\pi'}(s)$），v2，v3，如上图所示。但是依据价值迭代 + 策略改进，只需要计算v1的“半步”即可，因为根据策略改进定理，只需要计算出$q_\pi(s,a)$，即可得出$v_\pi(s)$ 和 $v_{\pi'}(s)$的大小关系。

![image-20201128135658979](https://gitee.com/YJLAugus/pic-go/raw/master/img/hss02.svg)



从 **策略评估-（迭代解）应用**小节 中可以知道，这个会有多个v1（注意：v1指的是多个状态，比如$v1^{21}$是第一次迭代中第二行第一个格子的状态，这里的“1”其实指的是迭代次数，并不一个状态）, 在这里选一个最大$v_\pi$作为$v*$ 。即完成了一次迭代。下面的公式，其是正是贝尔曼最优方程（求$v*$ ，$q*$）的 求$v*$的过程，同时也是价值迭代的算法公式：
$$
\large{
 v_{k+1}(s) \dot{=}\underset{a}{max} \sum_{s',r}p(s',r \mid s,a)[r+ \gamma v_k(s') \quad\quad(1)
}
$$
从上面（1）式中，从贝尔曼最优方程的较多来说，这里变成**了一条更新规则**，还是以**策略评估-（迭代解）应用**小节的例子举例，只计算v1，把“每个格子”的数据更新了一次（第一次迭代得到v1），而不需要在根据v1的数据j计算v2，再进行更新了。

除此之外，可以把（1）式和 “**策略评估-迭代解**”的更新策略$ v_{k+1}(s) \dot{=}\sum_{a\in A} \pi(a\mid s) \sum_{s',r}p(s',r \mid s,a)[r+ \gamma v_k(s')]$比较，唯一不同的是，在策略评估-迭代解中，对于 $s$状态的下一步`action` 需要根据概率求得，而在价值迭代中，直接在v1中选择最大的`action`，故概率此时为1。即：**价值迭代是极端情况下的策略迭代**。

根据上面描述的极端情况，其实就是价值迭代，总结如下：

![image-20201205213726436](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201205213726436.png)

### 异步动态规划

异步动态规划，也叫就地迭代，它是基于价值迭代的一种算法。在样本空间比较大的情况下，即使只进行一次迭代，也会花费很长的时间，还是以**策略评估-（迭代解）应用** 小节的例子，比如下图：

![image-20201205220225280](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201205220225280.png)

如果，格子的数量及其多，即使只计算v1，（更新一次），迭代一次，花费时间也是巨大的。所以就出现了一种算法，这种算法，只随机找到“其中的一个格子” 进行更新，这就是**异步动态规划**。然而，为了收敛，异步算法必须不断的更新所有状态的值。

## 广义策略迭代

广义策略迭代，其实就包含了前面所说的一般的策略迭代，价值迭代，还有就地迭代。如下面的图，如果都走到“顶”，那就是一般的策略迭代。如果走不到“顶”就可能是其他的迭代。

![image-20201205221614255](https://gitee.com/YJLAugus/pic-go/raw/master/img/image-20201205221614255.png)

按照下面的例子和类比，就很容易理解了，可以把这个广义策略迭代类比买房：

**策略迭代**：全款买房，以旧换新。

**价值迭代**：首付，以旧换新。

**就地策略迭代**：几乎0首付，以旧换新。

## 参考文献

https://www.cnblogs.com/pinard/p/9463815.html

https://www.bilibili.com/video/BV1nV411k7ve?t=1738

https://www.davidsilver.uk/wp-content/uploads/2020/03/DP.pdf

http://www.incompleteideas.net/book/the-book.html

