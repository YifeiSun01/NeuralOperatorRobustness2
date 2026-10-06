# 常见连续概率分布、多元推广与经典检验的统一关系

本文整理本次讨论中关于一元连续分布、多元分布、`chi-square / Wishart / t / F / Hotelling T²`，以及经典一维统计检验向多维统计推广的全部核心内容。

---

## 1. 最先要分清：分布本身的维度 vs. 构造它时用到的维度

一个概率分布“是一维还是多维”，取决于它最后描述的随机对象是什么。

- 如果随机对象是一个标量：
  [
  Xinmathbb R,
  ]
  那么它是一维分布。
- 如果随机对象是一个向量：
  [
  mathbf Xinmathbb R^p,
  ]
  那么它是 (p) 维分布。
- 如果随机对象是一个矩阵：
  [
  Minmathbb R^{p	imes p},
  ]
  那么它是矩阵值分布。

这和“它是从多少个随机变量构造出来的”不是一回事。

例如：

[
U=sum_{i=1}^{
u}Z_i^2,qquad Z_ioverset{iid}{sim}N(0,1)
]

虽然用了 (
u) 个高斯随机变量，但 (U) 最后只是一个标量，因此

[
Usimchi_
u^2
]

仍然是一维连续分布。

---

## 2. 本次笔记中出现过的主要连续分布

### 2.1 一元连续分布

本次上传的“一元”概率分布笔记中出现过的主要连续分布包括：

- Continuous Bernoulli
- Logistic
- Normal / Gaussian
- Half-normal
- Folded normal
- Laplace / Double exponential
- Continuous uniform
- Exponential
- Gamma
- Inverse-Gamma
- Beta
- Beta prime
- Log-normal
- Lévy
- Cauchy
- Stable distribution
- Gumbel / type-I extreme value
- Fréchet / inverse Weibull / type-II extreme value
- Weibull / type-III extreme value
- Rayleigh
- Pareto
- Bounded Pareto
- Generalized Pareto
- Chi-square
- Noncentral chi-square
- Chi distribution
- Maxwell / Maxwell–Boltzmann
- F distribution
- Student t distribution

### 2.2 多维连续分布

多元笔记中明确出现过的主要多维连续分布包括：

- Bivariate normal
- Multivariate normal
- Wishart
- Inverse-Wishart
- Hotelling (T^2) distribution
- Multivariate t
- Multivariate uniform
- Dirichlet

此外还讨论了：

- Matrix (F) distribution
- Matrix beta type II distribution

其中 Matrix (F) 和 matrix beta type II 在很多参数化下是紧密相关或等价表述。

---

# 3. Chi-square 本质上是“高维高斯半径平方”的一维分布

设

[
mathbf Z=
egin{pmatrix}
Z_1\
dots\
Z_
u
end{pmatrix}
sim N_
u(0,I).
]

那么

[
U=mathbf Z^Tmathbf Z
=sum_{i=1}^{
u}Z_i^2
=|mathbf Z|^2.
]

于是

[
oxed{
Usimchi_
u^2
}
]

这里 (
u) 是自由度，也可以理解为标准高斯向量的维数。

但注意：

[
Uinmathbb R_+.
]

因此不管 (
u=1,2,3,ldots)，(chi_
u^2) 本身始终是一个标量分布。

可以把这个映射理解为：

[
mathbb R^
u
longrightarrow
mathbb R_+,
qquad
mathbf zmapsto|mathbf z|^2.
]

也就是说，高维向量的所有方向信息全部被压掉，只留下半径平方。

---

# 4. Chi-square 开平方后得到 chi distribution

如果

[
Usimchi_
u^2,
]

定义

[
R=sqrt U,
]

那么

[
R=sqrt{Z_1^2+cdots+Z_
u^2}
=|mathbf Z|
]

服从自由度为 (
u) 的 chi distribution。

几个特殊情况非常重要：

### (
u=1)

[
R=|Z_1|
]

得到 Half-normal distribution。

### (
u=2)

[
R=sqrt{Z_1^2+Z_2^2}
]

得到 Rayleigh distribution。

### (
u=3)

[
R=sqrt{Z_1^2+Z_2^2+Z_3^2}
]

得到 Maxwell distribution。

因此可以记成：

[
oxed{
chi_
u^2 = R^2,
qquad
chi_
u = R.
}
]

并且：

[
chi_1	o 	ext{Half-normal},
qquad
chi_2	o 	ext{Rayleigh},
qquad
chi_3	o 	ext{Maxwell}.
]

---

# 5. 高维 Gaussian 的“薄球壳”几何

对

[
mathbf Zsim N_d(0,I),
]

点密度为

[
p(mathbf z)
=(2pi)^{-d/2}
e^{-|mathbf z|^2/2}.
]

在任意维度下，**单位体积的概率密度仍然是原点最高，离原点越远越低**。

但半径在 (r) 到 (r+dr) 的球壳中包含的总概率还要乘上球壳面积。

(d) 维球壳面积的主要尺度为：

[
r^{d-1}.
]

因此径向概率密度满足

[
oxed{
p_R(r)propto
r^{d-1}e^{-r^2/2}.
}
]

其中：

- (r^{d-1})：高维几何体积效应，把概率往外推；
- (e^{-r^2/2})：Gaussian 衰减，把概率往里拉。

二者竞争。

求其峰值：

[
log p_R(r)
=(d-1)log r-rac{r^2}{2}+C.
]

求导：

[
rac{d-1}{r}-r=0,
]

得到

[
oxed{
r_{mathrm{mode}}=sqrt{d-1}.
}
]

所以高维 Gaussian 有一个非常重要的现象：

[
oxed{
	ext{点密度最高处仍然是原点，但绝大部分概率质量位于 }rapproxsqrt d	ext{ 的薄球壳。}
}
]

---

# 6. 为什么 (E(Y_i^2)=1)

如果

[
Y_isim N(0,1),
]

那么

[
E(Y_i)=0,
qquad
operatorname{Var}(Y_i)=1.
]

利用恒等式

[
operatorname{Var}(Y_i)
=
E(Y_i^2)-[E(Y_i)]^2,
]

得到

[
1=E(Y_i^2)-0,
]

因此

[
oxed{
E(Y_i^2)=1.
}
]

更一般地：

[
Xsim N(mu,sigma^2)
]

时，

[
oxed{
E(X^2)=mu^2+sigma^2.
}
]

---

# 7. Student t 为什么自由度大时趋近正态

Student t 的标准构造为：

[
Zsim N(0,1),
qquad
Usimchi_
u^2,
qquad
Zperp U,
]

定义

[
oxed{
T=rac{Z}{sqrt{U/
u}}.
}
]

因为

[
U=sum_{i=1}^{
u}Y_i^2,
qquad
Y_ioverset{iid}{sim}N(0,1),
]

所以

[
rac U
u
=
rac1
u
sum_{i=1}^{
u}Y_i^2.
]

而

[
E(Y_i^2)=1.
]

根据大数定律：

[
rac U
u
	o1.
]

因此

[
sqrt{rac U
u}
	o1,
]

于是

[
T
=
rac Z{sqrt{U/
u}}
	o Z.
]

所以：

[
oxed{
t_
u
Rightarrow
N(0,1),
qquad

u	oinfty.
}
]

## 7.1 几何解释

因为

[
U=|mathbf Y|^2,
qquad
mathbf Ysim N_
u(0,I),
]

所以

[
sqrt{rac U
u}
=
rac{|mathbf Y|}{sqrt
u}.
]

高维 Gaussian 的半径集中在

[
|mathbf Y|approxsqrt
u
]

附近，因此

[
rac{|mathbf Y|}{sqrt
u}approx1.
]

Student t 的随机分母逐渐变成近似常数 (1)，于是 Student t 逐渐变成标准正态。

## 7.2 自由度小时为什么重尾

当 (
u=1) 时：

[
U=Y^2,
]

所以

[
sqrt U=|Y|.
]

于是

[
T=rac Z{|Y|}.
]

分母有机会非常接近 0，因此 (T) 有机会出现非常大的绝对值，形成重尾。

并且

[
oxed{
t_1=	ext{standard Cauchy}.
}
]

---

# 8. Cauchy 的二维几何解释

取两个独立标准正态：

[
X,Yoverset{iid}{sim}N(0,1).
]

二维标准高斯是旋转对称的。

写成极坐标：

[
X=RcosTheta,
qquad
Y=RsinTheta.
]

则

[
rac YX
=
	anTheta.
]

半径 (R) 完全消掉。

由于角度 (Theta) 在圆周方向上均匀，因此其正切服从 Cauchy。

所以可以把 Cauchy 看成：

[
oxed{
	ext{二维圆对称 Gaussian 的两个坐标之比。}
}
]

---

# 9. Chi-square 到 Wishart：从平方变成外积

普通 chi-square：

[
U=sum_{i=1}^{
u}z_i^2,
qquad
z_isim N(0,1).
]

对于标量 (z_i)，

[
z_i^Tz_i=z_i^2.
]

如果把 (z_i) 换成 (p) 维高斯向量：

[
mathbf z_iinmathbb R^p,
qquad
mathbf z_isim N_p(0,Sigma),
]

继续做

[
mathbf z_i^Tmathbf z_i
]

只会得到一个标量：

[
1	imes p
cdot
p	imes1
=
1	imes1.
]

Wishart 做的是相反次序的乘法：

[
oxed{
mathbf z_imathbf z_i^T.
}
]

此时：

[
p	imes1
cdot
1	imes p
=
p	imes p.
]

再求和：

[
oxed{
W=
sum_{i=1}^{
u}
mathbf z_imathbf z_i^T
}
]

得到 Wishart 随机矩阵：

[
oxed{
Wsim W_p(Sigma,
u).
}
]

所以：

[
oxed{
chi^2:quad
sum z_i^2
}
]

对应

[
oxed{
Wishart:quad
summathbf z_imathbf z_i^T.
}
]

---

# 10. Wishart 为什么和样本协方差联系得如此紧

一维正态总体：

[
X_1,ldots,X_n
overset{iid}{sim}
N(mu,sigma^2).
]

样本方差满足

[
oxed{
rac{(n-1)S^2}{sigma^2}
sim
chi^2_{n-1}.
}
]

所以一维里：

[
	ext{sample variance}
leftrightarrow
chi^2.
]

多维正态总体：

[
mathbf X_1,ldots,mathbf X_n
overset{iid}{sim}
N_p(oldsymbolmu,Sigma).
]

样本协方差矩阵为 (S)，则：

[
oxed{
(n-1)S
sim
W_p(Sigma,n-1)
}
]

在常见 Wishart 参数化下成立。

所以多维里：

[
	ext{sample covariance matrix}
leftrightarrow
Wishart.
]

因此最重要的对应关系是：

[
oxed{
	ext{variance}
ightarrow
	ext{covariance matrix}
}
]

和

[
oxed{
chi^2
ightarrow
Wishart.
}
]

当 (p=1) 时：

[
W_1(sigma^2,
u)
]

退化为标量的缩放 chi-square。

---

# 11. Inverse-Wishart

如果

[
Wsim W_p(Sigma,
u),
]

那么矩阵逆 (W^{-1}) 的分布属于 inverse-Wishart 家族（具体 scale 参数会随参数化约定变化）。

在一维里，对应关系是：

[
chi^2
longrightarrow
rac1{chi^2}.
]

在多维里：

[
Wishart
longrightarrow
Wishart^{-1}.
]

Hotelling (T^2) 中会出现 Wishart 的逆，就是因为一维 (t^2) 中本来就存在“除以样本方差”，而多维中“除以协方差矩阵”要通过矩阵逆来表示。

---

# 12. Student t 与 Multivariate t

Student t：

[
T=
rac Z{sqrt{U/
u}},
]

其中：

[
Zsim N(0,1),
qquad
Usimchi_
u^2.
]

最直接的向量推广是把分子 Gaussian 从标量变成向量：

[
mathbf Zsim N_p(0,Sigma).
]

保持分母仍然是同一个标量：

[
Usimchi_
u^2.
]

定义：

[
oxed{
mathbf T
=
oldsymbolmu
+
rac{mathbf Z}{sqrt{U/
u}}.
}
]

那么

[
mathbf T
]

服从 multivariate t distribution。

这个推广非常直接：

[
oxed{
N(0,1)
ightarrow
N_p(0,Sigma)
}
]

而 scalar chi-square 分母保留。

这意味着所有坐标共享一个随机缩放因子：

[
mathbf T=
rac1{sqrt{U/
u}}
egin{pmatrix}
Z_1\
dots\
Z_p
end{pmatrix}.
]

---

# 13. Multivariate t 的 scale matrix 与 covariance matrix

常见参数化下：

[
mathbf T
sim t_
u(oldsymbolmu,Sigma)
]

中的 (Sigma) 通常称为 scale matrix。

当

[

u>2
]

时：

[
oxed{
operatorname{Cov}(mathbf T)
=
rac{
u}{
u-2}Sigma.
}
]

因此 scale matrix 与 covariance matrix 一般不相等。

如果想让某个矩阵 (K) 直接等于 covariance matrix，可以重新参数化：

[
Sigma
=
rac{
u-2}{
u}K.
]

---

# 14. Student t 与 F 的精确关系

如果

[
T=
rac Z{sqrt{U/
u}},
]

那么

[
T^2
=
rac{Z^2}{U/
u}.
]

由于

[
Z^2simchi_1^2,
]

所以

[
T^2
=
rac{chi_1^2/1}{chi_
u^2/
u}.
]

根据 F 分布定义：

[
oxed{
T^2sim F_{1,
u}.
}
]

这就是：

[
oxed{
t_
u^2
=
F_{1,
u}
}
]

在分布意义上的精确关系。

---

# 15. F distribution

设

[
Usimchi_m^2,
qquad
Vsimchi_n^2,
qquad
Uperp V.
]

定义：

[
oxed{
F=
rac{U/m}{V/n}.
}
]

那么

[
Fsim F_{m,n}.
]

如果进一步展开：

[
U=sum_{i=1}^mZ_i^2,
qquad
V=sum_{j=1}^nY_j^2,
]

那么：

[
F
=
rac{rac1msum Z_i^2}
{rac1nsum Y_j^2}.
]

所以 F 可以理解成两个独立 Gaussian 向量的“归一化半径平方之比”。

---

# 16. Matrix F：F 的矩阵值推广

既然：

[
chi^2
ightarrow
Wishart,
]

那么 F 中两个 chi-square 可以同时升级为 Wishart：

[
W_1sim Wishart,
qquad
W_2sim Wishart.
]

因为矩阵不存在唯一的普通除法，一般构造类似：

[
oxed{
W_2^{-1/2}W_1W_2^{-1/2}
}
]

这样的对称正定矩阵。

得到 matrix F / matrix beta type II 类型的分布。

当

[
p=1
]

时所有矩阵退化成标量，回到普通 F distribution。

因此：

[
oxed{
F
=
rac{chi^2}{chi^2}
}
]

对应：

[
oxed{
Matrix F
sim
rac{Wishart}{Wishart}
}
]

其中“矩阵除法”需要借助矩阵逆或平方根定义。

---

# 17. Student t distribution、t statistic 与 t-test

这三个概念要分开。

## 17.1 Student t distribution

这是一个概率分布：

[
T=
rac Z{sqrt{U/
u}}
sim t_
u.
]

## 17.2 t statistic

如果

[
X_1,ldots,X_n
overset{iid}{sim}
N(mu,sigma^2),
]

定义：

[
ar X,
qquad
S^2.
]

检验

[
H_0:mu=mu_0.
]

构造：

[
oxed{
t=
rac{ar X-mu_0}{S/sqrt n}.
}
]

这个 (t) 是由样本构造出来的随机变量，因此也是统计量。

在 (H_0) 下：

[
tsim t_{n-1}.
]

## 17.3 t-test

t-test 是利用上述 t statistic 及其 Student t 分布进行显著性检验的方法。

因此：

[
oxed{
	ext{statistic 是随机变量的一种。}
}
]

在观察数据之前，统计量是随机变量；数据观察后，它变成一个具体数值。

---

# 18. 为什么样本形式的 Student t 和抽象定义完全一样

在 (H_0) 下：

[
Z=
rac{sqrt n(ar X-mu_0)}{sigma}
sim N(0,1),
]

并且：

[
U=
rac{(n-1)S^2}{sigma^2}
simchi^2_{n-1}.
]

同时：

[
Zperp U.
]

因此：

[
rac{ar X-mu_0}{S/sqrt n}
=
rac{
sqrt n(ar X-mu_0)/sigma
}{
sqrt{
[(n-1)S^2/sigma^2]/(n-1)
}
}.
]

即：

[
oxed{
t=
rac Z{sqrt{U/(n-1)}}.
}
]

所以样本统计量里的

- (n)
- (ar X)
- (mu_0)
- (S)

只是把抽象的 (Z) 与 (U) 用样本展开。

---

# 19. Hotelling (T^2)：t² 的多维统计推广

一维：

[
t^2
=
n
rac{(ar X-mu_0)^2}{S^2}.
]

写成乘逆数：

[
t^2
=
n
(ar X-mu_0)
(S^2)^{-1}
(ar X-mu_0).
]

现在把：

[
ar X-mu_0
]

推广成向量：

[
ar{mathbf X}
-
oldsymbolmu_0,
]

把：

[
S^2
]

推广成样本协方差矩阵：

[
S.
]

得到：

[
oxed{
T_H^2
=
n
(ar{mathbf X}-oldsymbolmu_0)^T
S^{-1}
(ar{mathbf X}-oldsymbolmu_0).
}
]

这就是 one-sample Hotelling (T^2)。

它是一个标量。

几何上：

[
(ar{mathbf X}-oldsymbolmu_0)^T
S^{-1}
(ar{mathbf X}-oldsymbolmu_0)
]

就是样本均值向量与假设均值向量之间的 Mahalanobis distance squared。

---

# 20. Hotelling (T^2) 的抽象随机变量定义

可以把 Hotelling 的结构写得和 Student t 非常对称。

取：

[
mathbf Zsim N_p(0,I),
]

以及独立的：

[
Wsim W_p(I,
u).
]

定义：

[
oxed{
H=

u
mathbf Z^TW^{-1}mathbf Z.
}
]

这就是 Hotelling (T^2) 型随机变量。

当 (p=1) 时：

[
Wsimchi_
u^2,
]

所以：

[
H
=

urac{Z^2}{W}
=
rac{Z^2}{W/
u}.
]

而：

[
T=
rac Z{sqrt{W/
u}},
]

因此：

[
oxed{
H=T^2.
}
]

所以：

[
oxed{
Hotelling T^2
	ext{ 是 }t^2	ext{ 的多维推广。}
}
]

---

# 21. Hotelling (T^2) 为什么出现 Wishart 的逆

一维：

[
t^2
=
nrac{(ar X-mu_0)^2}{S^2}.
]

这里本来就在“除以样本方差”。

多维里：

[
S^2
ightarrow
S.
]

“除以协方差矩阵”通过矩阵逆实现：

[
rac1{S^2}
ightarrow
S^{-1}.
]

而多元正态样本的 (S) 是 Wishart 型随机矩阵，所以：

[
S^{-1}
]

自然具有 inverse-Wishart 型结构。

因此：

[
oxed{
chi^2
ightarrow Wishart
}
]

以及：

[
oxed{
rac1{chi^2}
ightarrow
Wishart^{-1}
}
]

共同解释了 Hotelling (T^2) 为什么使用 (S^{-1})。

---

# 22. Multivariate t 与 Hotelling (T^2) 是两条不同的 Student t 推广路线

普通 Student t：

[
T=
rac Z{sqrt{U/
u}}.
]

## 路线 A：把随机变量本身从标量变成向量

[
Z
ightarrow
mathbf Z.
]

得到：

[
oxed{
mathbf T=
rac{mathbf Z}{sqrt{U/
u}}.
}
]

这就是 multivariate t。

输出：

[
mathbf Tinmathbb R^p.
]

## 路线 B：把一维 t-test / t² 统计结构推广到多维

一维：

[
t^2
=

u Z^2U^{-1}.
]

多维把：

[
Z^2
ightarrow
mathbf Z^T(cdot)mathbf Z,
]

同时：

[
Usimchi^2
ightarrow
Wsim Wishart.
]

得到：

[
oxed{
T_H^2
=

u
mathbf Z^TW^{-1}mathbf Z.
}
]

输出仍然是 scalar。

所以：

[
oxed{
Student t
ightarrow
Multivariate t
}
]

是“随机变量的维度推广”。

而：

[
oxed{
t^2	ext{ statistic}
ightarrow
Hotelling T^2
}
]

是“统计检验结构的多维推广”。

---

# 23. Multivariate t 的 Mahalanobis 半径与 F 的关系

设：

[
mathbf T
=
rac{mathbf Z}{sqrt{U/
u}},
qquad
mathbf Zsim N_p(0,Sigma),
]

并且：

[
Usimchi_
u^2.
]

那么：

[
Q
=
mathbf T^T
Sigma^{-1}
mathbf T.
]

代入：

[
Q
=
rac{
mathbf Z^T
Sigma^{-1}
mathbf Z
}{
U/
u
}.
]

而：

[
mathbf Z^T
Sigma^{-1}
mathbf Z
sim
chi_p^2.
]

所以：

[
Q
=
p
rac{chi_p^2/p}{chi_
u^2/
u}.
]

因此：

[
oxed{
rac Qp
sim F_{p,
u}.
}
]

也就是：

[
oxed{
mathbf T^T
Sigma^{-1}
mathbf T
sim
pF_{p,
u}.
}
]

当 (p=1) 时：

[
T^2sim F_{1,
u}.
]

---

# 24. Hotelling (T^2) 与 F 的关系

对 one-sample Hotelling test，在 (p) 维正态总体、样本数 (n) 下：

[
T_H^2
=
n
(ar{mathbf X}-oldsymbolmu_0)^T
S^{-1}
(ar{mathbf X}-oldsymbolmu_0).
]

在原假设成立时：

[
oxed{
rac{n-p}{p(n-1)}
T_H^2
sim
F_{p,n-p}.
}
]

这对应于一维关系：

[
oxed{
t^2
sim
F_{1,n-1}.
}
]

---

# 25. 一个统一的“一维 → 多维”分布对应表

| 一维对象 | 多维对应 | 多维随机对象类型 |
|---|---|---|
| (N(mu,sigma^2)) | Multivariate normal (N_p(mu,Sigma)) | 向量 |
| (chi^2) | Wishart | 矩阵 |
| (1/chi^2) 型 | Inverse-Wishart 型 | 矩阵 |
| Student (t) | Multivariate t | 向量 |
| Cauchy | Multivariate Cauchy | 向量 |
| (F) | Matrix F / Matrix beta II | 矩阵 |
| (t^2) 统计结构 | Hotelling (T^2) | 标量 |

其中：

[
oxed{
chi^2ightarrow Wishart
}
]

是多元统计中最核心的对应之一。

---

# 26. “只要出现 chi-square，就换成 Wishart”需要一个适用条件

这是非常好用的结构直觉，但不能机械套用到所有 chi-square。

它主要适用于：

[
oxed{
	ext{Gaussian quadratic form / sample variance}
}
]

这一条来源。

也就是：

[
	ext{sample variance}
ightarrow
	ext{sample covariance matrix},
]

因此：

[
chi^2
ightarrow
Wishart.
]

但 Pearson goodness-of-fit / contingency table 中的 chi-square 是一种渐近检验分布，其来源不同，不能直接换成 Wishart。

---

# 27. 经典一维统计检验向多维统计推广

很多经典检验都有非常自然的多维对应。

最核心的替换规则是：

[
oxed{
x
ightarrow
mathbf x
}
]

[
oxed{
mu
ightarrow
oldsymbolmu
}
]

[
oxed{
sigma^2
ightarrow
Sigma
}
]

[
oxed{
S^2
ightarrow
S
}
]

[
oxed{
(x-mu)^2
ightarrow
(mathbf x-oldsymbolmu)^T
Sigma^{-1}
(mathbf x-oldsymbolmu)
}
]

以及：

[
oxed{
	ext{scalar sum of squares}
ightarrow
	ext{SSCP matrix}
}
]

其中 SSCP 指：

[
	ext{sum of squares and cross-products}.
]

---

# 28. 单样本 z-test 的多维对应

一维，已知方差：

[
Z
=
rac{sqrt n(ar X-mu_0)}{sigma}.
]

因此：

[
Z^2
=
n
rac{(ar X-mu_0)^2}{sigma^2}
simchi_1^2.
]

多维：

[
mathbf X_i
sim
N_p(oldsymbolmu,Sigma),
]

已知 (Sigma)。

定义：

[
oxed{
Q=
n
(ar{mathbf X}-oldsymbolmu_0)^T
Sigma^{-1}
(ar{mathbf X}-oldsymbolmu_0).
}
]

那么：

[
oxed{
Qsimchi_p^2.
}
]

所以：

[
oxed{
Z^2
ightarrow
	ext{Mahalanobis distance squared}.
}
]

---

# 29. 单样本 t-test → one-sample Hotelling (T^2)

一维：

[
t=
rac{ar X-mu_0}{S/sqrt n}.
]

多维：

[
oxed{
T^2
=
n
(ar{mathbf X}-oldsymbolmu_0)^T
S^{-1}
(ar{mathbf X}-oldsymbolmu_0).
}
]

这就是 Hotelling (T^2)。

---

# 30. 两独立样本 t-test → two-sample Hotelling (T^2)

一维 pooled t-test：

[
t=
rac{ar X_1-ar X_2}
{s_psqrt{1/n_1+1/n_2}}.
]

多维把：

[
ar X_1-ar X_2
]

换成：

[
ar{mathbf X}_1-ar{mathbf X}_2.
]

把 pooled variance 换成 pooled covariance：

[
oxed{
S_p
=
rac{
(n_1-1)S_1+(n_2-1)S_2
}{
n_1+n_2-2
}.
}
]

对应统计量：

[
oxed{
T^2
=
(ar{mathbf X}_1-ar{mathbf X}_2)^T
left[
S_p
left(
rac1{n_1}
+
rac1{n_2}
ight)
ight]^{-1}
(ar{mathbf X}_1-ar{mathbf X}_2).
}
]

在共同协方差假设下，适当缩放以后服从 F 分布。

---

# 31. Paired t-test → paired Hotelling (T^2)

一维配对检验先定义：

[
D_i
=
X_{i,1}-X_{i,2}.
]

然后对 (D_i) 做 one-sample t-test。

多维时：

[
mathbf D_i
=
mathbf X_{i,1}
-
mathbf X_{i,2}.
]

然后检验：

[
H_0:
E(mathbf D)=0.
]

统计量：

[
oxed{
T^2
=
n
ar{mathbf D}^{,T}
S_D^{-1}
ar{mathbf D}.
}
]

因此：

[
oxed{
paired t
ightarrow
paired Hotelling T^2.
}
]

---

# 32. 一维 ANOVA F-test → MANOVA

普通 one-way ANOVA：

[
F=
rac{MS_B}{MS_E}.
]

其中：

- (SS_B)：组间平方和；
- (SS_E)：组内平方和。

如果每个观测从 scalar：

[
Yinmathbb R
]

升级为 vector：

[
mathbf Yinmathbb R^p,
]

那么一维平方：

[
(y-ar y)^2
]

升级成外积：

[
(mathbf y-ar{mathbf y})
(mathbf y-ar{mathbf y})^T.
]

所以：

[
SS_B
ightarrow
H,
]

[
SS_E
ightarrow
E,
]

其中：

- (H)：hypothesis SSCP matrix；
- (E)：error SSCP matrix。

这就是 MANOVA 的核心。

---

# 33. 为什么 MANOVA 没有唯一的一个 “multivariate F”

一维：

[
F=
rac{SS_B/df_B}{SS_E/df_E}
]

是两个标量之比。

多维时：

[
H,E
]

都是矩阵，而矩阵不存在唯一普通除法。

因此通常研究：

[
E^{-1}H
]

的特征值：

[
lambda_1,ldots,lambda_r.
]

然后有四个经典统计量。

## 33.1 Wilks' Lambda

[
oxed{
Lambda
=
rac{|E|}{|H+E|}
=
prod_irac1{1+lambda_i}.
}
]

## 33.2 Pillai's Trace

[
oxed{
V=
sum_i
rac{lambda_i}{1+lambda_i}.
}
]

## 33.3 Hotelling–Lawley Trace

[
oxed{
T=
sum_ilambda_i.
}
]

## 33.4 Roy's Largest Root

[
oxed{
R=
max_ilambda_i.
}
]

因此：

[
oxed{
ANOVA
ightarrow
MANOVA
}
]

以后不再只有一个唯一的 F statistic，而是出现多种把矩阵关系压缩成 scalar statistic 的方法。

---

# 34. 重复测量 ANOVA 的多维对应

如果同一个对象在多个时间点、多个条件下都有响应：

[
Y_{i1},Y_{i2},ldots,Y_{ip},
]

可以直接组成向量：

[
mathbf Y_i
=
egin{pmatrix}
Y_{i1}\
dots\
Y_{ip}
end{pmatrix}.
]

于是重复测量问题可以进入 multivariate repeated-measures MANOVA 框架。

---

# 35. 回归里的 t / F 检验 → multivariate regression tests

普通线性回归：

[
y=Xeta+arepsilon.
]

检验单个系数：

[
H_0:eta_j=0
]

通常用 t-test。

联合检验多个系数：

[
H_0:
eta_{j_1}
=
cdots
=
eta_{j_k}
=0
]

通常用 F-test。

如果响应从 scalar：

[
y
]

升级成向量：

[
mathbf yinmathbb R^p,
]

模型变成：

[
oxed{
Y=XB+E.
}
]

此时检验一个 predictor 是否对所有 response 都没有影响，会再次构造：

[
H,E
]

两个 SSCP 矩阵，并使用：

- Wilks' Lambda
- Pillai's Trace
- Hotelling–Lawley Trace
- Roy's Largest Root

进行联合检验。

所以：

[
oxed{
	ext{univariate regression}
ightarrow
	ext{multivariate multiple regression}.
}
]

---

# 36. 一元方差 chi-square test → covariance matrix test

一维：

[
H_0:
sigma^2=sigma_0^2.
]

利用：

[
rac{(n-1)S^2}{sigma_0^2}
sim
chi_{n-1}^2.
]

多维：

[
H_0:
Sigma=Sigma_0.
]

因为：

[
(n-1)S
sim
Wishart,
]

所以可以利用 Wishart / likelihood-ratio 结构对整个 covariance matrix 进行检验。

这是：

[
oxed{
chi^2
ightarrow
Wishart
}
]

最直接的统计检验对应。

---

# 37. 两总体方差 F-test → covariance matrix equality test

一维：

[
H_0:
sigma_1^2=sigma_2^2.
]

常用：

[
F=
rac{S_1^2}{S_2^2}.
]

多维变成：

[
oxed{
H_0:
Sigma_1=Sigma_2.
}
]

此时：

[
S_1,S_2
]

都是随机矩阵。

概率分布层面可以构造 Matrix F。

实际统计检验中更常见的是：

- likelihood-ratio test；
- Box's (M) test；
- 以及其他 covariance homogeneity tests。

所以：

[
oxed{
	ext{variance equality}
ightarrow
	ext{covariance-matrix equality}.
}
]

---

# 38. Pearson correlation test → canonical correlation

一维：

[
X,Y
]

各一个变量，检验：

[
H_0:ho=0.
]

多维时：

[
mathbf Xinmathbb R^p,
qquad
mathbf Yinmathbb R^q.
]

问题变成：

> 两组变量之间是否存在线性关系？

Canonical Correlation Analysis 寻找：

[
a^Tmathbf X
]

与：

[
b^Tmathbf Y
]

之间最大的相关系数。

得到：

[
ho_1,ho_2,ldots.
]

所以：

[
oxed{
Pearson correlation
ightarrow
canonical correlation.
}
]

对 canonical correlations 的联合显著性检验同样会使用 Wilks' Lambda 等多元统计量。

---

# 39. Pearson chi-square 检验是另一条线

Pearson goodness-of-fit / independence test：

[
X^2
=
sum
rac{(O_i-E_i)^2}{E_i}.
]

这里的 chi-square 来自统计量的渐近分布。

它不是 Gaussian sample variance 的那条 chi-square。

因此不能做：

[
chi^2
ightarrow
Wishart
]

这种替换。

分类变量进一步多维化通常进入：

- multiway contingency tables；
- log-linear models；
- generalized linear models。

---

# 40. 统一的一维检验 → 多维检验表

| 一维检验 | 多维对应 | 核心替换 |
|---|---|---|
| one-sample z | known-covariance mean-vector test | (z^2	o) Mahalanobis distance squared |
| one-sample t | one-sample Hotelling (T^2) | (s^2	o S) |
| two-sample t | two-sample Hotelling (T^2) | pooled variance (	o) pooled covariance |
| paired t | paired Hotelling (T^2) | scalar difference (	o) difference vector |
| one-way ANOVA F | MANOVA | SS (	o) SSCP matrices |
| repeated-measures ANOVA | multivariate repeated-measures MANOVA | repeated scalars (	o) response vector |
| regression coefficient t | multivariate regression joint test | scalar response (	o) response vector |
| regression overall F | multivariate regression / MANOVA tests | scalar SSR/SSE (	o H,E) |
| variance chi-square test | covariance matrix test | (chi^2	o Wishart) |
| two-variance F test | covariance homogeneity test | variance ratio (	o) matrix comparison |
| Pearson correlation test | canonical correlation | one correlation (	o) canonical correlations |
| Pearson contingency-table chi-square | multiway contingency/log-linear | 不走 Wishart 路线 |

---

# 41. 最底层只有三类代数变化

整个经典一维统计向 multivariate statistics 的推广，可以压缩成三个最核心的操作：

## 41.1 数变成向量

[
x
ightarrow
mathbf x.
]

## 41.2 平方变成外积 / 二次型

[
x^2
ightarrow
mathbf xmathbf x^T
]

或：

[
x^2/sigma^2
ightarrow
mathbf x^T
Sigma^{-1}
mathbf x.
]

## 41.3 方差变成协方差矩阵

[
sigma^2
ightarrow
Sigma.
]

由此自然产生：

[
chi^2
ightarrow
Wishart,
]

[
t^2
ightarrow
Hotelling T^2,
]

[
F
ightarrow
Matrix F,
]

[
ANOVA
ightarrow
MANOVA.
]

---

# 42. 最核心的统一框架

整个讨论最后可以用下面这组关系记住：

[
oxed{
N(mu,sigma^2)
longrightarrow
N_p(oldsymbolmu,Sigma)
}
]

[
oxed{
chi^2
longrightarrow
Wishart
}
]

[
oxed{
Student t
longrightarrow
Multivariate t
}
]

[
oxed{
F
longrightarrow
Matrix F
}
]

以及统计检验路线：

[
oxed{
t^2
longrightarrow
Hotelling T^2
}
]

[
oxed{
ANOVA
longrightarrow
MANOVA
}
]

[
oxed{
	ext{variance test}
longrightarrow
	ext{covariance-matrix test}
}
]

[
oxed{
	ext{correlation test}
longrightarrow
	ext{canonical-correlation test}.
}
]

---

# 43. 最后几个容易混淆的点

1. **(chi_
u^2) 的 (
u) 是自由度，不代表 chi-square 本身是 (
u) 维随机变量。**  
   chi-square 的结果始终是 scalar。

2. **Wishart 才是 chi-square 的自然矩阵值推广。**  
   它保留了向量样本的完整二阶结构。

3. **Multivariate t 的分母仍然可以是 scalar chi-square。**  
   因为它表示整个向量共享一个随机尺度。

4. **Hotelling (T^2) 才会把一维样本方差结构完整升级成随机协方差矩阵。**  
   因此其中出现 Wishart / inverse-Wishart。

5. **Multivariate t 和 Hotelling (T^2) 都和 Student t 有深层关系，但推广方向不同。**
   - Multivariate t：随机变量从 scalar 变成 vector。
   - Hotelling (T^2)：t-test / (t^2) statistic 的结构从一维推广到多维。

6. **Matrix F 是 F distribution 的矩阵值推广。**  
   它使用两个 Wishart 随机矩阵。

7. **MANOVA 没有唯一的一个 multivariate F statistic。**  
   因为矩阵比值没有唯一标量化方式，因此出现 Wilks、Pillai、Hotelling–Lawley、Roy 四套经典统计量。

8. **Pearson chi-square 检验里的 chi-square 不应该机械替换成 Wishart。**  
   只有 Gaussian quadratic form / sample variance 这一条线才自然对应 Wishart。

---

# 44. 一句话总结

[
oxed{
	ext{经典多元统计，本质上就是把“标量、平方、方差”系统地升级成“向量、外积/二次型、协方差矩阵”。}
}
]

在这条升级路径上：

[
oxed{
chi^2leftrightarrow Wishart,qquad
tleftrightarrow Multivariate t,qquad
Fleftrightarrow Matrix F,qquad
t^2leftrightarrow Hotelling T^2.
}
]

而 ANOVA、回归、均值检验、方差检验、相关检验，也都沿着同一个思想进入 MANOVA、multivariate regression、covariance tests 和 canonical correlation。

# 45. 后续讨论：Hotelling、MANOVA、矩阵 PDF 与 Wishart 几何

下面继续按对话推进顺序整理后续所有核心内容。前面的内容已经建立了

\[
\chi^2\to Wishart,\qquad
t\to \text{Multivariate }t,\qquad
F\to \text{Matrix }F,
\]

以及

\[
t^2\to \text{Hotelling }T^2,\qquad
ANOVA\to MANOVA
\]

这几条主线。后续讨论主要解决的是：这些对象到底服从什么分布、MANOVA 为什么会出现四套统计量、这些矩阵分布有没有显式 PDF、Wishart 的 support 和正定锥到底是什么意思。

---

# 46. One-sample Hotelling \(T^2\) 和 Two-sample Hotelling \(T^2\) 最终都是普通标量 \(F\)

## 46.1 One-sample Hotelling \(T^2\)

设

\[
\mathbf X_1,\ldots,\mathbf X_n
\overset{iid}{\sim}
N_p(\boldsymbol\mu,\Sigma),
\]

检验

\[
H_0:\boldsymbol\mu=\boldsymbol\mu_0.
\]

定义

\[
\boxed{
T^2
=
n(\bar{\mathbf X}-\boldsymbol\mu_0)^T
S^{-1}
(\bar{\mathbf X}-\boldsymbol\mu_0)
}
\]

则在 \(H_0\) 下

\[
\boxed{
\frac{n-p}{p(n-1)}T^2
\sim
F_{p,n-p}.
}
\]

因此

\[
\boxed{
T^2
\sim
\frac{p(n-1)}{n-p}
F_{p,n-p}.
}
\]

所以 One-sample Hotelling \(T^2\) 是一个 scalar statistic，并不是 Matrix \(F\)。

---

## 46.2 Two-sample Hotelling \(T^2\)

设

\[
\mathbf X_{1i}\sim N_p(\boldsymbol\mu_1,\Sigma),
\qquad
\mathbf X_{2j}\sim N_p(\boldsymbol\mu_2,\Sigma),
\]

两个样本独立，并假定共同协方差矩阵 \(\Sigma\)。

检验

\[
H_0:\boldsymbol\mu_1=\boldsymbol\mu_2.
\]

定义 pooled covariance

\[
\boxed{
S_p
=
\frac{
(n_1-1)S_1+(n_2-1)S_2
}{
n_1+n_2-2
}.
}
\]

Hotelling 统计量

\[
\boxed{
T^2
=
\frac{n_1n_2}{n_1+n_2}
(\bar{\mathbf X}_1-\bar{\mathbf X}_2)^T
S_p^{-1}
(\bar{\mathbf X}_1-\bar{\mathbf X}_2).
}
\]

则

\[
\boxed{
\frac{
n_1+n_2-p-1
}{
p(n_1+n_2-2)
}
T^2
\sim
F_{
p,\,
n_1+n_2-p-1
}.
}
\]

因此 Two-sample Hotelling \(T^2\) 同样最终对应普通 scalar \(F\)。

---

## 46.3 Paired Hotelling \(T^2\)

若每个对象有两个 \(p\)-维观测

\[
\mathbf X_{i1},\mathbf X_{i2},
\]

先构造差向量

\[
\mathbf D_i
=
\mathbf X_{i1}-\mathbf X_{i2}.
\]

然后对

\[
H_0:E(\mathbf D)=0
\]

做 One-sample Hotelling：

\[
\boxed{
T^2
=
n\bar{\mathbf D}^{\,T}
S_D^{-1}
\bar{\mathbf D}.
}
\]

因此配对 \(t\)-test 的多维推广就是对差向量做 One-sample Hotelling \(T^2\)。

---

# 47. Hotelling \(T^2\) 为什么不是 Matrix \(F\)

设

\[
\mathbf Z\sim N_p(0,I),
\qquad
W\sim W_p(I,\nu),
\qquad
\mathbf Z\perp W.
\]

抽象 Hotelling 结构可以写成

\[
\boxed{
H
=
\nu\mathbf Z^TW^{-1}\mathbf Z.
}
\]

虽然 \(W^{-1}\) 是矩阵，但左右被向量夹住：

\[
(1\times p)(p\times p)(p\times1)
=
1\times1.
\]

因此

\[
\boxed{
H\in\mathbb R.
}
\]

而且

\[
\boxed{
\frac{\nu-p+1}{p\nu}
H
\sim
F_{p,\nu-p+1}.
}
\]

所以 Hotelling \(T^2\) 的最后随机对象仍然是 scalar。

相比之下，Matrix \(F\) 保留整个矩阵：

\[
\boxed{
M
=
\left(\frac{W_2}{n}\right)^{-1/2}
\left(\frac{W_1}{m}\right)
\left(\frac{W_2}{n}\right)^{-1/2}
\in\mathbb S_{++}^p.
}
\]

因此

\[
\boxed{
\text{Hotelling }T^2
\neq
\text{Matrix }F.
}
\]

核心区别：

\[
\boxed{
Wishart/Wishart
\to
Matrix\ F
}
\]

保留矩阵；

而

\[
\boxed{
N_p^T Wishart^{-1}N_p
\to
Hotelling\ T^2
\to
\text{scaled ordinary }F
}
\]

最后压成一个标量。

---

# 48. “三层对象”统一框架

整个讨论可以分成三层。

## 48.1 第一层：基础随机对象

一维：

\[
Z\sim N(0,1),
\qquad
U_\nu\sim\chi_\nu^2.
\]

多维：

\[
\mathbf Z\sim N_p(0,\Sigma),
\qquad
W_\nu\sim W_p(\Sigma,\nu).
\]

对应关系：

\[
\boxed{
N\to N_p
}
\]

以及

\[
\boxed{
\chi^2\to Wishart.
}
\]

---

## 48.2 第二层：由基础分布组合出的标准分布

### Student \(t\)

\[
\boxed{
T
=
\frac{Z}{\sqrt{U_\nu/\nu}}
\sim t_\nu.
}
\]

### Multivariate \(t\)

\[
\boxed{
\mathbf T
=
\boldsymbol\mu
+
\frac{\mathbf Z}{\sqrt{U_\nu/\nu}}
}
\]

其中

\[
\mathbf Z\sim N_p(0,\Sigma).
\]

所以

\[
\boxed{
\frac{N_p}{\sqrt{\chi^2/\nu}}
\to
\text{Multivariate }t.
}
\]

这里分母仍然是 scalar \(\chi^2\)。

### Scalar \(F\)

\[
\boxed{
F
=
\frac{U_m/m}{V_n/n}
\sim F_{m,n}.
}
\]

### Matrix \(F\)

把两个 \(\chi^2\) 换成两个 Wishart：

\[
W_1\sim W_p(\Sigma,m),
\qquad
W_2\sim W_p(\Sigma,n).
\]

构造矩阵比值：

\[
\boxed{
M
=
\left(\frac{W_2}{n}\right)^{-1/2}
\left(\frac{W_1}{m}\right)
\left(\frac{W_2}{n}\right)^{-1/2}.
}
\]

因此

\[
\boxed{
\frac{\chi^2/m}{\chi^2/n}
\to F,
\qquad
\frac{Wishart/m}{Wishart/n}
\to Matrix\ F.
}
\]

---

## 48.3 第三层：为了 hypothesis test 再压成 scalar

一维：

\[
T^2
=
\nu Z^2U_\nu^{-1}.
\]

多维 Hotelling：

\[
\boxed{
T_H^2
=
\nu\mathbf Z^TW_\nu^{-1}\mathbf Z.
}
\]

最后

\[
\boxed{
\text{scaled }T_H^2\sim F.
}
\]

MANOVA 则保留两个 SSCP 矩阵 \(H,E\)，先形成矩阵比结构，再取特征值，最后用不同方式压成 scalar statistics。

---

# 49. 多元 \(t\) 的 Mahalanobis 平方

定义

\[
\mathbf T
=
\frac{\mathbf Z}{\sqrt{U_\nu/\nu}},
\qquad
\mathbf Z\sim N_p(0,\Sigma),
\qquad
U_\nu\sim\chi_\nu^2.
\]

取 Mahalanobis 平方：

\[
\boxed{
Q
=
\mathbf T^T\Sigma^{-1}\mathbf T.
}
\]

代入：

\[
Q
=
\frac{
\mathbf Z^T\Sigma^{-1}\mathbf Z
}{
U_\nu/\nu
}.
\]

因为

\[
\mathbf Z^T\Sigma^{-1}\mathbf Z
\sim
\chi_p^2,
\]

所以

\[
Q
=
\frac{\chi_p^2}{\chi_\nu^2/\nu}
=
p
\frac{\chi_p^2/p}{\chi_\nu^2/\nu}.
\]

因此

\[
\boxed{
\frac{Q}{p}
\sim
F_{p,\nu}.
}
\]

即

\[
\boxed{
\frac1p
\mathbf T^T\Sigma^{-1}\mathbf T
\sim
F_{p,\nu}.
}
\]

当 \(p=1\) 时：

\[
\boxed{
T^2\sim F_{1,\nu}.
}
\]

---

# 50. 多元 \(t\) 的平方与 Hotelling \(T^2\) 是否是同一个东西

一般情况下不是。

## 50.1 Multivariate \(t\)

\[
\mathbf T
=
\underbrace{
\frac1{\sqrt{U/\nu}}
}_{\text{scalar random scale}}
\mathbf Z.
\]

整个向量共享一个 scalar random scale。

## 50.2 Hotelling 结构

令

\[
\mathbf Y
=
W^{-1/2}\mathbf Z.
\]

则

\[
\boxed{
H
=
\nu\mathbf Y^T\mathbf Y
=
\nu\|\mathbf Y\|^2.
}
\]

这里随机缩放是一个矩阵：

\[
\boxed{
W^{-1/2}.
}
\]

若

\[
W
=
Q\Lambda Q^T,
\]

则

\[
W^{-1/2}
=
Q\Lambda^{-1/2}Q^T,
\]

它会对不同方向进行不同的随机缩放。

因此：

\[
\boxed{
\text{Multivariate }t:
\quad
\text{scalar random scaling}
}
\]

而

\[
\boxed{
\text{Hotelling structure}:
\quad
\text{matrix random scaling}.
}
\]

两者最终的 \(F\) 也不同：

\[
\boxed{
\frac1p
\mathbf T^T\Sigma^{-1}\mathbf T
\sim
F_{p,\nu}
}
\]

而

\[
\boxed{
\frac{\nu-p+1}{p\nu}
H
\sim
F_{p,\nu-p+1}.
}
\]

只有 \(p=1\) 时：

\[
Wishart_1
=
\chi^2,
\]

矩阵缩放退化成 scalar 缩放，从而

\[
\boxed{
H=T^2.
}
\]

---

# 51. Hotelling 与 multivariate \(t\) 更深一层的联系

虽然 \(p>1\) 时二者不是同一个构造，但

\[
\mathbf Y
=
W^{-1/2}\mathbf Z
\]

把 \(W\) 积分掉以后，\(\mathbf Y\) 本身仍属于 multivariate \(t\)-type 家族。

设

\[
r=\nu-p+1.
\]

在常见 scale 参数化下，可以写成

\[
\boxed{
\mathbf Y
\sim
t_{p,r}
\left(
0,\frac1r I
\right).
}
\]

于是

\[
\frac{r\|\mathbf Y\|^2}{p}
\sim
F_{p,r}.
\]

而

\[
H
=
\nu\|\mathbf Y\|^2,
\]

因此

\[
\boxed{
\frac{r}{p\nu}H
\sim
F_{p,r},
\qquad
r=\nu-p+1.
}
\]

这就是 Hotelling 的 scaled-\(F\) 关系。

---

# 52. MANOVA 从 ANOVA 怎么得到

普通 ANOVA 每个观测是 scalar：

\[
Y_{gi}\in\mathbb R.
\]

平方和分解：

\[
SS_T
=
SS_H+SS_E.
\]

其中 \(H\) 表示组间 hypothesis variation，\(E\) 表示组内 error variation。

多维时每个观测变成

\[
\mathbf Y_{gi}\in\mathbb R^p.
\]

标量平方

\[
(y-\bar y)^2
\]

升级成外积

\[
(\mathbf y-\bar{\mathbf y})
(\mathbf y-\bar{\mathbf y})^T.
\]

于是：

\[
\boxed{
H
=
\sum_g
n_g
(\bar{\mathbf Y}_g-\bar{\mathbf Y})
(\bar{\mathbf Y}_g-\bar{\mathbf Y})^T
}
\]

以及

\[
\boxed{
E
=
\sum_g\sum_i
(\mathbf Y_{gi}-\bar{\mathbf Y}_g)
(\mathbf Y_{gi}-\bar{\mathbf Y}_g)^T.
}
\]

二者都是 \(p\times p\) SSCP 矩阵。

总矩阵：

\[
\boxed{
T=H+E.
}
\]

---

# 53. SSCP 矩阵到底包含什么

二维时

\[
E
=
\begin{pmatrix}
E_{11}&E_{12}\\
E_{12}&E_{22}
\end{pmatrix}.
\]

其中

\[
E_{11}
\]

是第一个响应变量的组内平方和，

\[
E_{22}
\]

是第二个响应变量的组内平方和，

\[
E_{12}
\]

是两个响应变量之间的组内 cross-product。

因此 SSCP 的全称是

\[
\boxed{
\text{Sum of Squares and Cross Products}.
}
\]

它是 scalar sum-of-squares 的自然多维推广。

---

# 54. MANOVA 为什么出现 Matrix \(F\)-like 对象

在标准 Gaussian MANOVA 的 \(H_0\) 下，可以把

\[
H
\]

和

\[
E
\]

看成独立 Wishart 型矩阵：

\[
\boxed{
H\sim W_p(\Sigma,\nu_H),
\qquad
E\sim W_p(\Sigma,\nu_E),
\qquad
H\perp E.
}
\]

这与普通 ANOVA 中两个 chi-square 型平方和完全对应。

一维：

\[
\boxed{
\frac{\chi^2/\nu_H}{\chi^2/\nu_E}
\to F.
}
\]

多维：

\[
\boxed{
\frac{Wishart/\nu_H}{Wishart/\nu_E}
\to
Matrix\ F\text{-like structure}.
}
\]

定义

\[
\widetilde H
=
\frac H{\nu_H},
\qquad
\widetilde E
=
\frac E{\nu_E}.
\]

可以形成

\[
\boxed{
M
=
\widetilde E^{-1/2}
\widetilde H
\widetilde E^{-1/2}
=
\frac{\nu_E}{\nu_H}
E^{-1/2}HE^{-1/2}.
}
\]

这就是 MANOVA 底层的矩阵 ratio 结构。

---

# 55. MANOVA 的广义特征值

通常研究

\[
E^{-1}H.
\]

它与

\[
E^{-1/2}HE^{-1/2}
\]

具有相同的非零特征值。

定义

\[
\boxed{
\lambda_1,\ldots,\lambda_s
=
\operatorname{eig}(E^{-1}H),
}
\]

其中

\[
s\le\min(p,\operatorname{rank}H).
\]

等价于广义特征值问题：

\[
\boxed{
H\mathbf a_i
=
\lambda_iE\mathbf a_i.
}
\]

沿方向 \(\mathbf a\) 把多维响应压成

\[
Z=\mathbf a^T\mathbf Y,
\]

则组间 variation：

\[
\mathbf a^TH\mathbf a,
\]

组内 variation：

\[
\mathbf a^TE\mathbf a.
\]

对应 ratio：

\[
\boxed{
R(\mathbf a)
=
\frac{\mathbf a^TH\mathbf a}
{\mathbf a^TE\mathbf a}.
}
\]

最大化这个 Rayleigh quotient 得到最大广义特征值：

\[
\boxed{
\lambda_1
=
\max_{\mathbf a\neq0}
\frac{\mathbf a^TH\mathbf a}
{\mathbf a^TE\mathbf a}.
}
\]

因此

\[
\boxed{
\lambda_i
=
\text{第 }i\text{ 个最佳多维方向上的 signal/error ratio}.
}
\]

---

# 56. 四个经典 MANOVA statistics

底层共同信息是

\[
\lambda_1,\ldots,\lambda_s.
\]

不同统计量只是采用不同的 scalarization。

## 56.1 Hotelling–Lawley Trace

\[
\boxed{
U
=
\operatorname{tr}(E^{-1}H)
=
\sum_{i=1}^s\lambda_i.
}
\]

解释：直接把所有 signal/error roots 相加。

---

## 56.2 Roy's Largest Root

\[
\boxed{
R
=
\lambda_{\max}
=
\lambda_1.
}
\]

解释：只看最强的 discriminant direction。

---

## 56.3 Pillai's Trace

定义

\[
\boxed{
V
=
\operatorname{tr}
\left[
H(H+E)^{-1}
\right].
}
\]

若

\[
\theta_i
=
\frac{\lambda_i}{1+\lambda_i},
\]

则

\[
\boxed{
V
=
\sum_{i=1}^s
\frac{\lambda_i}{1+\lambda_i}
=
\sum_i\theta_i.
}
\]

每个

\[
0\le\theta_i<1.
\]

如果把

\[
\lambda_i
=
\frac{\text{signal}}{\text{error}},
\]

则

\[
\theta_i
=
\frac{\lambda_i}{1+\lambda_i}
=
\frac{\text{signal}}
{\text{signal}+\text{error}}.
\]

所以 Pillai 可以理解为各 canonical direction 的 explained fraction 之和。

---

## 56.4 Wilks' Lambda

定义

\[
\boxed{
\Lambda
=
\frac{|E|}{|E+H|}.
}
\]

因为

\[
|E+H|
=
|E|\,|I+E^{-1}H|,
\]

所以

\[
\Lambda
=
\frac1{|I+E^{-1}H|}.
\]

由 determinant 等于特征值乘积：

\[
\boxed{
\Lambda
=
\prod_{i=1}^s
\frac1{1+\lambda_i}.
}
\]

又因为

\[
\theta_i
=
\frac{\lambda_i}{1+\lambda_i},
\]

所以

\[
\boxed{
\Lambda
=
\prod_i(1-\theta_i).
}
\]

因此 Wilks 可以理解为各 canonical direction 上“剩余 error fraction”的乘积。

---

# 57. 四个 MANOVA statistics 的统一表

| Statistic | 公式 | 对 \(\lambda_i\) 的操作 |
|---|---|---|
| Wilks | \(\displaystyle \Lambda=\prod_i(1+\lambda_i)^{-1}\) | 乘 residual fraction |
| Pillai | \(\displaystyle V=\sum_i\lambda_i/(1+\lambda_i)\) | 加 bounded explained fraction |
| Hotelling–Lawley | \(\displaystyle U=\sum_i\lambda_i\) | 直接加 signal/error |
| Roy | \(\displaystyle R=\max_i\lambda_i\) | 只取最强方向 |

所以：

\[
\boxed{
H,E
\to
E^{-1}H
\to
\{\lambda_i\}
\to
\{Wilks,Pillai,HL,Roy\}.
}
\]

---

# 58. 为什么四个统计量在多维时会产生不同结论倾向

如果只有一个强方向：

\[
\lambda=(10,0,0),
\]

那么

\[
HL=10,
\qquad
Roy=10,
\]

\[
Pillai
=
\frac{10}{11}
\approx0.909,
\]

\[
Wilks
=
\frac1{11}
\approx0.091.
\]

如果效应均匀分布在三个方向：

\[
\lambda=(3,3,3),
\]

那么

\[
HL=9,
\qquad
Roy=3,
\]

\[
Pillai
=
3\frac34
=
2.25,
\]

\[
Wilks
=
\frac1{4^3}
=
0.015625.
\]

Roy 对“一个特别强方向”最敏感；Pillai 和 Wilks 更能反映效应是否分布在多个方向。

---

# 59. 一维 ANOVA 为什么没有四套不同 statistics

当 \(p=1\) 时只有一个 root：

\[
\lambda=\frac HE.
\]

于是：

\[
\boxed{
Wilks
=
\frac1{1+\lambda},
}
\]

\[
\boxed{
Pillai
=
\frac{\lambda}{1+\lambda},
}
\]

\[
\boxed{
HL=\lambda,
}
\]

\[
\boxed{
Roy=\lambda.
}
\]

它们全部是同一个 \(\lambda\) 的单调函数，因此检验排序完全一致。

普通 ANOVA：

\[
\boxed{
F
=
\frac{H/\nu_H}{E/\nu_E}
=
\frac{\nu_E}{\nu_H}\lambda.
}
\]

所以在一维情形：

\[
\boxed{
Wilks
\Longleftrightarrow
Pillai
\Longleftrightarrow
HL
\Longleftrightarrow
Roy
\Longleftrightarrow
F.
}
\]

多维后有多个 \(\lambda_i\)，才真正产生不同的 scalarization。

---

# 60. MANOVA statistics 最后服从什么分布

MANOVA 底层的

\[
H,E
\]

在标准 Gaussian \(H_0\) 下是 Wishart 型，因此

\[
E^{-1}H
\]

是 Matrix-\(F\)/matrix-beta 类型结构。

对应特征值

\[
\lambda_1,\ldots,\lambda_s
\]

有已知的联合 latent-root distribution。

然后

\[
Wilks=f_1(\lambda_1,\ldots,\lambda_s),
\]

\[
Pillai=f_2(\lambda_1,\ldots,\lambda_s),
\]

\[
HL=f_3(\lambda_1,\ldots,\lambda_s),
\]

\[
Roy=f_4(\lambda_1,\ldots,\lambda_s).
\]

它们一般不再简单地等于某个单一

\[
F_{a,b}
\]

或

\[
\chi_k^2.
\]

实际统计软件通常使用：

\[
\boxed{
F\text{-transformation / }F\text{-approximation}
}
\]

或者某些 exact multivariate calculation 来获得 \(p\)-value。

所以：

\[
\boxed{
\text{底层分布已知}
\neq
\text{最终 scalar statistic 必须属于简单的 }t/F/\chi^2\text{ 家族}.
}
\]

---

# 61. 两方差 \(F\)-test 的多维对应

一维：

\[
H_0:\sigma_1^2=\sigma_2^2.
\]

在正态条件下：

\[
\frac{(n_1-1)S_1^2}{\sigma^2}
\sim\chi^2_{n_1-1},
\]

\[
\frac{(n_2-1)S_2^2}{\sigma^2}
\sim\chi^2_{n_2-1}.
\]

所以方差比是普通 \(F\)。

多维：

\[
S_1^2\to S_1,
\qquad
S_2^2\to S_2,
\]

而

\[
\chi^2\to Wishart.
\]

所以底层矩阵 ratio：

\[
\boxed{
Wishart/Wishart
\to
Matrix\ F.
}
\]

但是实际检验

\[
H_0:\Sigma_1=\Sigma_2
\]

并没有唯一一种 scalar test。

常见做法包括：

\[
\boxed{
\text{Box's }M,
}
\]

likelihood-ratio tests，以及基于 determinant、trace、eigenvalues 的其他 covariance-homogeneity tests。

因此：

\[
\boxed{
\text{Matrix }F
}
\]

是底层矩阵分布的自然对应；

而

\[
\boxed{
\text{covariance equality test}
}
\]

是基于这些矩阵关系进一步构造的统计检验。

---

# 62. 前面提到的主要分布及其多维推广关系

| 一维分布/对象 | 多维对应 | 随机对象类型 |
|---|---|---|
| Gaussian | Multivariate Gaussian | 向量 |
| \(\chi^2\) | Wishart | 对称半正定/正定矩阵 |
| Student \(t\) | Multivariate \(t\) | 向量 |
| Cauchy \(=t_1\) | Multivariate Cauchy | 向量 |
| \(F\) | Matrix \(F\) / Matrix beta-II | 正定矩阵 |
| \(t^2\) statistic | Hotelling \(T^2\) | 标量 |
| inverse-\(\chi^2\)-type | Inverse-Wishart | 正定矩阵 |

最重要的三条“分布本身的多维推广”：

\[
\boxed{
\chi^2\to Wishart
}
\]

\[
\boxed{
t\to Multivariate\ t
}
\]

\[
\boxed{
F\to Matrix\ F.
}
\]

而

\[
\boxed{
t^2\to Hotelling\ T^2
}
\]

和

\[
\boxed{
ANOVA\to MANOVA
}
\]

属于统计量/检验结构的多维推广。

---

# 63. 中国本科概率统计里的“三大抽样分布”

中国本科《概率论与数理统计》里通常说的“三大抽样分布”就是：

\[
\boxed{
\chi^2,\qquad
t,\qquad
F.
}
\]

它们在多元正态统计中的自然对应是：

\[
\boxed{
\begin{array}{ccc}
\chi^2
&\longrightarrow&
Wishart\\[2mm]
t
&\longrightarrow&
Multivariate\ t\\[2mm]
F
&\longrightarrow&
Matrix\ F
\end{array}
}
\]

再往假设检验层面走：

\[
\boxed{
t^2
\to
Hotelling\ T^2,
}
\]

\[
\boxed{
ANOVA
\to
MANOVA.
}
\]

---

# 64. 各种分布不仅有生成式定义，也有解析 PDF

前面大量使用

\[
Z,U\to T,
\]

\[
\mathbf Z_i\to W,
\]

\[
W_1,W_2\to Matrix\ F
\]

这样的生成式定义，是为了看清分布之间的关系。

但大多数经典分布本身都有显式概率密度：

\[
\boxed{
\text{scalar PDF},\quad
\text{joint vector PDF},\quad
\text{matrix PDF}.
}
\]

连续随机变量在某个精确点的概率都是零：

\[
P(X=x)=0,
\]

\[
P(\mathbf X=\mathbf x)=0,
\]

\[
P(W=W_0)=0.
\]

PDF 给的是局部概率浓度。

---

# 65. 一维 Gaussian PDF

\[
X\sim N(\mu,\sigma^2)
\]

有

\[
\boxed{
f_X(x)
=
\frac1{\sqrt{2\pi}\sigma}
\exp
\left[
-\frac{(x-\mu)^2}{2\sigma^2}
\right].
}
\]

---

# 66. Multivariate Gaussian 的 joint PDF

\[
\mathbf X\sim N_p(\boldsymbol\mu,\Sigma)
\]

有

\[
\boxed{
f_{\mathbf X}(\mathbf x)
=
\frac1{
(2\pi)^{p/2}
|\Sigma|^{1/2}
}
\exp
\left[
-\frac12
(\mathbf x-\boldsymbol\mu)^T
\Sigma^{-1}
(\mathbf x-\boldsymbol\mu)
\right].
}
\]

这就是

\[
f_{X_1,\ldots,X_p}(x_1,\ldots,x_p).
\]

---

# 67. Chi-square PDF

\[
U\sim\chi_\nu^2
\]

有

\[
\boxed{
f_U(u)
=
\frac1{
2^{\nu/2}\Gamma(\nu/2)
}
u^{\nu/2-1}
e^{-u/2},
\qquad
u>0.
}
\]

---

# 68. Chi distribution 与 Chi-square 的区别

设

\[
Z_1,\ldots,Z_k\overset{iid}{\sim}N(0,1).
\]

定义 Gaussian 半径：

\[
\boxed{
R
=
\sqrt{
Z_1^2+\cdots+Z_k^2
}.
}
\]

则

\[
\boxed{
R\sim\chi_k.
}
\]

其平方：

\[
\boxed{
Q=R^2
=
Z_1^2+\cdots+Z_k^2
\sim\chi_k^2.
}
\]

因此

\[
\boxed{
\chi_k^2
=
(\chi_k)^2
}
\]

是在随机变量变换意义上的关系。

Chi distribution 的 PDF：

\[
\boxed{
f_R(r)
=
\frac{
2^{1-k/2}
}{
\Gamma(k/2)
}
r^{k-1}
e^{-r^2/2},
\qquad r>0.
}
\]

---

# 69. Half-normal、Rayleigh、Maxwell 都是 Chi distribution 的特殊情况

\[
\boxed{
\chi_1
=
\text{Half-normal}
}
\]

因为

\[
R=|Z_1|.
\]

\[
\boxed{
\chi_2
=
\text{Rayleigh}
}
\]

因为

\[
R
=
\sqrt{Z_1^2+Z_2^2}.
\]

\[
\boxed{
\chi_3
=
\text{Maxwell}
}
\]

因为

\[
R
=
\sqrt{
Z_1^2+Z_2^2+Z_3^2
}.
\]

经典 Maxwell–Boltzmann speed distribution 就是 scaled \(\chi_3\)。

若

\[
V_x,V_y,V_z
\overset{iid}{\sim}
N(0,\sigma^2),
\]

那么速度大小

\[
V
=
\sqrt{
V_x^2+V_y^2+V_z^2
}
\]

满足

\[
\boxed{
\frac V\sigma
\sim
\chi_3.
}
\]

---

# 70. Student \(t\) 也可以用 Chi distribution 表示

因为

\[
U_\nu\sim\chi_\nu^2
\]

意味着

\[
R=\sqrt{U_\nu}\sim\chi_\nu.
\]

所以

\[
T
=
\frac{Z}{\sqrt{U_\nu/\nu}}
\]

可以写成

\[
\boxed{
T
=
\frac{\sqrt\nu\,Z}{R},
\qquad
R\sim\chi_\nu.
}
\]

即 Student \(t\) 的分母本质上是一个 \(\nu\)-维标准 Gaussian 向量的随机半径。

---

# 71. Student \(t\) PDF

标准 Student \(t\)：

\[
T\sim t_\nu
\]

有

\[
\boxed{
f_T(t)
=
\frac{
\Gamma\left(\frac{\nu+1}{2}\right)
}{
\sqrt{\nu\pi}
\Gamma\left(\frac{\nu}{2}\right)
}
\left(
1+\frac{t^2}{\nu}
\right)^{-(\nu+1)/2}.
}
\]

这是标准化参数：

\[
\mu=0,
\qquad
\sigma=1.
\]

---

# 72. 一般 location-scale Student \(t\)

若 location 为 \(\mu\)，scale 为 \(\sigma\)，则

\[
\boxed{
f(x)
=
\frac{
\Gamma\left(\frac{\nu+1}{2}\right)
}{
\Gamma\left(\frac{\nu}{2}\right)
\sqrt{\nu\pi}\,\sigma
}
\left[
1+
\frac{(x-\mu)^2}{\nu\sigma^2}
\right]^{-(\nu+1)/2}.
}
\]

注意 \(\sigma^2\) 是 scale，不一定等于 variance。

当 \(\nu>2\)：

\[
\boxed{
\operatorname{Var}(X)
=
\frac{\nu}{\nu-2}\sigma^2.
}
\]

---

# 73. Multivariate \(t\) 的解析 joint PDF

若

\[
\mathbf X
\sim
t_{p,\nu}(\boldsymbol\mu,\Sigma),
\]

则

\[
\boxed{
f_{\mathbf X}(\mathbf x)
=
\frac{
\Gamma\left(\frac{\nu+p}{2}\right)
}{
\Gamma\left(\frac{\nu}{2}\right)
(\nu\pi)^{p/2}
|\Sigma|^{1/2}
}
\left[
1+
\frac1\nu
(\mathbf x-\boldsymbol\mu)^T
\Sigma^{-1}
(\mathbf x-\boldsymbol\mu)
\right]^{-(\nu+p)/2}.
}
\]

标准化情形

\[
\boldsymbol\mu=0,
\qquad
\Sigma=I_p
\]

时：

\[
\boxed{
f(\mathbf x)
=
\frac{
\Gamma\left(\frac{\nu+p}{2}\right)
}{
\Gamma\left(\frac{\nu}{2}\right)
(\nu\pi)^{p/2}
}
\left(
1+\frac{\|\mathbf x\|^2}{\nu}
\right)^{-(\nu+p)/2}.
}
\]

---

# 74. Multivariate Gaussian 与 Multivariate \(t\) 的 PDF 几何关系

两者都只通过 Mahalanobis radius

\[
r^2
=
(\mathbf x-\boldsymbol\mu)^T
\Sigma^{-1}
(\mathbf x-\boldsymbol\mu)
\]

依赖 \(\mathbf x\)。

Gaussian：

\[
\boxed{
f_G(\mathbf x)
\propto
e^{-r^2/2}.
}
\]

Multivariate \(t\)：

\[
\boxed{
f_t(\mathbf x)
\propto
\left(
1+\frac{r^2}{\nu}
\right)^{-(\nu+p)/2}.
}
\]

因此二者具有相同的椭球层集结构，区别主要在 radial decay：

\[
\boxed{
\text{Gaussian: exponential tail}
}
\]

\[
\boxed{
t: polynomial heavy tail.
}
\]

---

# 75. Multivariate \(t\) 的 scale matrix 与 covariance

在常见参数化下：

\[
\mathbf X
\sim
t_{p,\nu}(\boldsymbol\mu,\Sigma)
\]

中的 \(\Sigma\) 是 scale matrix。

当

\[
\nu>2
\]

时：

\[
\boxed{
\operatorname{Cov}(\mathbf X)
=
\frac{\nu}{\nu-2}\Sigma.
}
\]

因此

\[
\boxed{
\Sigma
\neq
\operatorname{Cov}(\mathbf X)
}
\]

一般成立。

若希望某个矩阵 \(K\) 直接是 covariance：

\[
K
=
\operatorname{Cov}(\mathbf X),
\]

则需要取

\[
\boxed{
\Sigma
=
\frac{\nu-2}{\nu}K.
}
\]

---

# 76. F distribution PDF

\[
F\sim F_{m,n}
\]

有

\[
\boxed{
f_F(x)
=
\frac{
\Gamma\left(\frac{m+n}{2}\right)
}{
\Gamma(m/2)\Gamma(n/2)
}
\left(\frac mn\right)^{m/2}
x^{m/2-1}
\left(
1+\frac mnx
\right)^{-(m+n)/2},
\qquad x>0.
}
\]

---

# 77. Wishart 的矩阵 PDF

设

\[
W\sim W_p(\Sigma,\nu),
\qquad
\Sigma\succ0.
\]

非退化密度写成

\[
\boxed{
f_W(W)
=
\frac{
|W|^{(\nu-p-1)/2}
\exp\left[
-\frac12
\operatorname{tr}(\Sigma^{-1}W)
\right]
}{
2^{\nu p/2}
|\Sigma|^{\nu/2}
\Gamma_p(\nu/2)
}
\mathbf 1_{\{W\succ0\}}.
}
\]

其中 multivariate gamma function：

\[
\boxed{
\Gamma_p(a)
=
\pi^{p(p-1)/4}
\prod_{j=1}^{p}
\Gamma
\left(
a-\frac{j-1}{2}
\right).
}
\]

---

# 78. Wishart 的 support 如何写进 PDF

indicator：

\[
\mathbf 1_{\{W\succ0\}}
=
\begin{cases}
1,&W\succ0,\\
0,&W\not\succ0.
\end{cases}
\]

所以完整 PDF 也可以写成分段：

\[
\boxed{
f_W(W)
=
\begin{cases}
\displaystyle
\frac{
|W|^{(\nu-p-1)/2}
e^{-\frac12\operatorname{tr}(\Sigma^{-1}W)}
}{
2^{\nu p/2}
|\Sigma|^{\nu/2}
\Gamma_p(\nu/2)
},
&
W\succ0,
\\[5mm]
0,
&
\text{otherwise}.
\end{cases}
}
\]

因此 support 并不是必须由前面的代数表达式自动“看出来”。

很多教材省略 indicator，只写

\[
W\succ0
\]

作为公式旁边的条件。

---

# 79. 为什么不能只靠 determinant 判断正定

\[
|W|>0
\]

并不足以推出

\[
W\succ0.
\]

例如：

\[
W=
\begin{pmatrix}
-1&0\\
0&-1
\end{pmatrix}.
\]

有

\[
|W|=1>0,
\]

但

\[
W\prec0.
\]

因此

\[
|W|^\alpha
\]

本身不能决定 Wishart support。

正定条件必须单独作为 support 写进去。

---

# 80. Wishart 为什么天然至少是半正定

构造：

\[
\boxed{
W
=
\sum_{i=1}^{\nu}
\mathbf z_i\mathbf z_i^T.
}
\]

对任意 \(\mathbf a\)：

\[
\mathbf a^TW\mathbf a
=
\sum_i
(\mathbf a^T\mathbf z_i)^2
\ge0.
\]

因此：

\[
\boxed{
W\succeq0.
}
\]

所以 Wishart 不可能产生 indefinite matrix。

---

# 81. 为什么 Wishart 不一定总是正定

令数据矩阵

\[
X
=
\begin{pmatrix}
\mathbf z_1^T\\
\vdots\\
\mathbf z_\nu^T
\end{pmatrix}
\in\mathbb R^{\nu\times p}.
\]

则

\[
\boxed{
W=X^TX.
}
\]

因此

\[
\boxed{
\operatorname{rank}(W)
=
\operatorname{rank}(X)
\le
\min(\nu,p).
}
\]

如果

\[
\nu<p,
\]

则

\[
\operatorname{rank}(W)
\le\nu<p,
\]

所以

\[
\boxed{
|W|=0
}
\]

必然 singular。

若

\[
\nu\ge p
\]

且 \(\Sigma\succ0\)，Gaussian 样本连续，则随机向量以概率 1 张成整个 \(\mathbb R^p\)，因此

\[
\boxed{
P(W\succ0)=1.
}
\]

---

# 82. 为什么非退化 Wishart 密度要求 \(\nu>p-1\)

如果 \(\nu\) 是整数样本数：

\[
\nu>p-1
\iff
\nu\ge p.
\]

这与 rank 条件完全一致。

对于非整数自由度的 analytic extension，条件仍然是

\[
\boxed{
\nu>p-1.
}
\]

从 normalization constant 也能直接看见：

\[
\Gamma_p(\nu/2)
=
\pi^{p(p-1)/4}
\prod_{j=1}^p
\Gamma
\left(
\frac{\nu-j+1}{2}
\right).
\]

最后一项：

\[
\Gamma
\left(
\frac{\nu-p+1}{2}
\right)
\]

要求

\[
\frac{\nu-p+1}{2}>0,
\]

即

\[
\boxed{
\nu>p-1.
}
\]

---

# 83. Bartlett decomposition 对 \(\nu>p-1\) 的解释

若

\[
W\sim W_p(I,\nu),
\]

可写成

\[
\boxed{
W=LL^T,
}
\]

其中 \(L\) 为下三角矩阵，对角元满足：

\[
L_{11}^2\sim\chi_\nu^2,
\]

\[
L_{22}^2\sim\chi_{\nu-1}^2,
\]

一直到

\[
\boxed{
L_{pp}^2
\sim
\chi_{\nu-p+1}^2.
}
\]

而

\[
|W|
=
|L|^2
=
\prod_{j=1}^pL_{jj}^2.
\]

为了得到正定矩阵，需要最后一个 chi-square 自由度仍然为正：

\[
\nu-p+1>0.
\]

所以：

\[
\boxed{
\nu>p-1.
}
\]

---

# 84. Wishart PDF 中 determinant 幂为什么出现

Scalar chi-square：

\[
q
=
\sum_{i=1}^{\nu}z_i^2
=
\|\mathbf z\|^2.
\]

Gaussian density：

\[
f(\mathbf z)
\propto
e^{-\|\mathbf z\|^2/2}.
\]

球坐标 Jacobian：

\[
d\mathbf z
=
r^{\nu-1}dr\,d\Omega.
\]

令

\[
q=r^2,
\qquad
dr=\frac1{2\sqrt q}dq.
\]

于是

\[
r^{\nu-1}dr
=
\frac12
q^{\nu/2-1}dq.
\]

因此 chi-square PDF 中出现

\[
\boxed{
q^{\nu/2-1}
}
\]

来自径向变量变换的 Jacobian / volume factor。

矩阵版本：

\[
X\in\mathbb R^{\nu\times p},
\qquad
W=X^TX.
\]

对 \(X\) 做 matrix polar / QR decomposition，把 orientation 积掉以后，Jacobian 产生：

\[
\boxed{
|W|^{(\nu-p-1)/2}.
}
\]

所以 determinant power 是 scalar radial volume factor 的矩阵对应。

---

# 85. 为什么 determinant 是自然的矩阵体积量

线性变换

\[
\mathbf x\mapsto A\mathbf x
\]

使体积元满足：

\[
\boxed{
dV
\mapsto
|\det A|\,dV.
}
\]

因此 determinant 天生度量矩阵对体积的缩放。

这就是为什么标量中的 power-law volume factor，到了矩阵空间会自然出现 determinant 的幂。

---

# 86. Wishart exponent 为什么是 \(\operatorname{tr}(\Sigma^{-1}W)\)

每个

\[
\mathbf z_i
\sim
N_p(0,\Sigma)
\]

的 Gaussian exponent 是

\[
-\frac12
\mathbf z_i^T
\Sigma^{-1}
\mathbf z_i.
\]

多个独立样本联合起来：

\[
\exp
\left[
-\frac12
\sum_i
\mathbf z_i^T
\Sigma^{-1}
\mathbf z_i
\right].
\]

利用恒等式：

\[
\boxed{
\mathbf z_i^T
\Sigma^{-1}
\mathbf z_i
=
\operatorname{tr}
\left(
\Sigma^{-1}
\mathbf z_i\mathbf z_i^T
\right).
}
\]

所以

\[
\sum_i
\mathbf z_i^T
\Sigma^{-1}
\mathbf z_i
=
\operatorname{tr}
\left[
\Sigma^{-1}
\sum_i
\mathbf z_i\mathbf z_i^T
\right].
\]

而

\[
W
=
\sum_i
\mathbf z_i\mathbf z_i^T.
\]

因此：

\[
\boxed{
\sum_i
\mathbf z_i^T
\Sigma^{-1}
\mathbf z_i
=
\operatorname{tr}(\Sigma^{-1}W).
}
\]

所以 Wishart exponent：

\[
\boxed{
e^{-\frac12\operatorname{tr}(\Sigma^{-1}W)}
}
\]

直接来自原始 Gaussian quadratic form，并不是人为猜出来的。

---

# 87. 一维 scaled chi-square 与 Wishart 的完全对应

若

\[
X_i\sim N(0,\sigma^2),
\]

则

\[
W=\sum_iX_i^2
\]

满足

\[
\frac W{\sigma^2}
\sim\chi_\nu^2.
\]

其 PDF：

\[
\boxed{
f_W(w)
=
\frac{
w^{\nu/2-1}
}{
2^{\nu/2}
\Gamma(\nu/2)
(\sigma^2)^{\nu/2}
}
e^{-w/(2\sigma^2)}.
}
\]

Wishart 中令

\[
p=1,
\qquad
\Sigma=\sigma^2,
\qquad
W=w,
\]

有

\[
|\Sigma|=\sigma^2,
\qquad
|W|=w,
\]

\[
\operatorname{tr}(\Sigma^{-1}W)
=
\frac w{\sigma^2},
\]

\[
\Gamma_1(a)=\Gamma(a).
\]

因此 Wishart PDF 完全退化成上面的 scaled chi-square PDF。

---

# 88. Inverse-Wishart PDF

若

\[
S\sim IW_p(\nu,\Psi),
\]

则

\[
\boxed{
f(S)
=
\frac{
|\Psi|^{\nu/2}
}{
2^{\nu p/2}
\Gamma_p(\nu/2)
}
|S|^{-(\nu+p+1)/2}
\exp
\left[
-\frac12
\operatorname{tr}(\Psi S^{-1})
\right]
\mathbf 1_{\{S\succ0\}}.
}
\]

和 Wishart 对比：

\[
Wishart:
\quad
|S|^{(\nu-p-1)/2}
e^{-\frac12\operatorname{tr}(\Sigma^{-1}S)},
\]

\[
Inverse\text{-}Wishart:
\quad
|S|^{-(\nu+p+1)/2}
e^{-\frac12\operatorname{tr}(\Psi S^{-1})}.
\]

---

# 89. Matrix \(F\) / Matrix Beta II 的 PDF

取

\[
A\sim W_p(I,m),
\qquad
B\sim W_p(I,n),
\qquad
A\perp B.
\]

先定义 unscaled matrix ratio：

\[
\boxed{
X
=
B^{-1/2}AB^{-1/2}.
}
\]

则 \(X\succ0\)，其 matrix beta type II density：

\[
\boxed{
f_X(X)
=
\frac{
\Gamma_p\left(\frac{m+n}{2}\right)
}{
\Gamma_p(m/2)
\Gamma_p(n/2)
}
|X|^{(m-p-1)/2}
|I+X|^{-(m+n)/2}
\mathbf 1_{\{X\succ0\}}.
}
\]

也可以定义 multivariate beta function：

\[
\boxed{
B_p(a,b)
=
\frac{
\Gamma_p(a)\Gamma_p(b)
}{
\Gamma_p(a+b)
}.
}
\]

则：

\[
\boxed{
f_X(X)
=
\frac1{B_p(m/2,n/2)}
|X|^{(m-p-1)/2}
|I+X|^{-(m+n)/2}
\mathbf 1_{\{X\succ0\}}.
}
\]

---

# 90. 按普通 scalar \(F\) 归一化的 Matrix \(F\)

为了让 \(p=1\) 时精确退回标准

\[
F_{m,n},
\]

定义：

\[
\boxed{
F_M
=
\frac nm
B^{-1/2}AB^{-1/2}.
}
\]

其 PDF：

\[
\boxed{
f_{F_M}(F)
=
\frac{
\Gamma_p\left(\frac{m+n}{2}\right)
}{
\Gamma_p(m/2)
\Gamma_p(n/2)
}
\left(\frac mn\right)^{mp/2}
|F|^{(m-p-1)/2}
\left|
I+\frac mnF
\right|^{-(m+n)/2}
\mathbf 1_{\{F\succ0\}}.
}
\]

当 \(p=1\) 时：

\[
|F|=f,
\]

\[
\left|
I+\frac mnF
\right|
=
1+\frac mnf,
\]

\[
\Gamma_1=\Gamma,
\]

于是：

\[
\boxed{
f_{F_M}(f)
=
f_{F_{m,n}}(f).
}
\]

所以：

\[
\boxed{
Matrix\ F
\xrightarrow{p=1}
F.
}
\]

---

# 91. Wishart PDF 与 Matrix \(F\) PDF 的 scalar-to-matrix 对照

Chi-square：

\[
\boxed{
f(x)
\propto
x^{\nu/2-1}
e^{-x/2}.
}
\]

Wishart：

\[
\boxed{
f(W)
\propto
|W|^{(\nu-p-1)/2}
e^{-\frac12\operatorname{tr}(\Sigma^{-1}W)}.
}
\]

Scalar \(F\)：

\[
\boxed{
f(x)
\propto
x^{m/2-1}
\left(
1+\frac mnx
\right)^{-(m+n)/2}.
}
\]

Matrix \(F\)：

\[
\boxed{
f(F)
\propto
|F|^{(m-p-1)/2}
\left|
I+\frac mnF
\right|^{-(m+n)/2}.
}
\]

这里的规律不是简单机械替换，而是：

\[
\boxed{
\text{scalar volume factor}
\to
\text{determinant power},
}
\]

以及

\[
\boxed{
\text{Gaussian quadratic form}
\to
\operatorname{tr}(\Sigma^{-1}W).
}
\]

---

# 92. Wishart 为什么只有 \(\frac{p(p+1)}2\) 个独立坐标

Wishart：

\[
W
=
\sum_i
\mathbf z_i\mathbf z_i^T.
\]

每一项满足：

\[
(\mathbf z_i\mathbf z_i^T)^T
=
\mathbf z_i\mathbf z_i^T.
\]

所以：

\[
\boxed{
W^T=W.
}
\]

因此

\[
w_{ij}=w_{ji}.
\]

一般 \(p\times p\) 矩阵有

\[
p^2
\]

个坐标。

对称矩阵中：

- 对角线有 \(p\) 个；
- 严格上三角有

\[
\frac{p(p-1)}2
\]

个；
- 下三角由上三角完全决定。

因此独立坐标总数：

\[
\boxed{
p+\frac{p(p-1)}2
=
\frac{p(p+1)}2.
}
\]

这里最好称为“矩阵空间维数”或“独立坐标数”，不要和 Wishart 的 degrees of freedom \(\nu\) 混淆。

---

# 93. 对称矩阵空间 \(\mathbb S^p\)

定义：

\[
\boxed{
\mathbb S^p
=
\{
A\in\mathbb R^{p\times p}:A^T=A
\}.
}
\]

这是一个向量空间，维数：

\[
\boxed{
\dim(\mathbb S^p)
=
\frac{p(p+1)}2.
}
\]

例如 \(p=2\)：

\[
W
=
\begin{pmatrix}
a&b\\
b&c
\end{pmatrix}
\]

对应坐标：

\[
(a,b,c)\in\mathbb R^3.
\]

因此：

\[
\boxed{
\mathbb S^2\cong\mathbb R^3.
}
\]

---

# 94. 正定矩阵空间 \(\mathbb S_{++}^p\)

定义：

\[
\boxed{
\mathbb S_{++}^p
=
\{
W\in\mathbb S^p:
\mathbf x^TW\mathbf x>0
\quad
\forall \mathbf x\neq0
\}.
}
\]

所以：

\[
\mathbb S_{++}^p
\subset
\mathbb S^p.
\]

对 \(2\times2\)：

\[
W=
\begin{pmatrix}
a&b\\
b&c
\end{pmatrix}.
\]

正定等价于：

\[
\boxed{
a>0,
\qquad
ac-b^2>0.
}
\]

这两个条件会自动推出 \(c>0\)。

因此在 \((a,b,c)\in\mathbb R^3\) 中：

\[
\boxed{
\mathbb S_{++}^2
=
\{
(a,b,c):
a>0,\;
ac>b^2
\}.
}
\]

---

# 95. 什么叫“正定锥”

数学中的 cone 指满足正数缩放闭合的集合。

若

\[
W\in C,
\qquad
t>0,
\]

则

\[
tW\in C.
\]

对正定矩阵：

\[
W\succ0
\]

意味着

\[
\mathbf x^TW\mathbf x>0
\quad
\forall\mathbf x\neq0.
\]

于是对任何 \(t>0\)：

\[
\mathbf x^T(tW)\mathbf x
=
t\mathbf x^TW\mathbf x
>0.
\]

因此：

\[
\boxed{
W\succ0
\Longrightarrow
tW\succ0.
}
\]

所以 \(\mathbb S_{++}^p\) 是一个 cone。

另外若

\[
A\succ0,
\qquad
B\succ0,
\]

则：

\[
\mathbf x^T(A+B)\mathbf x
=
\mathbf x^TA\mathbf x
+
\mathbf x^TB\mathbf x
>0.
\]

因此：

\[
\boxed{
A+B\succ0.
}
\]

所以正定锥还是凸锥：

\[
\boxed{
\mathbb S_{++}^p
\text{ is an open convex cone}.
}
\]

---

# 96. \(2\times2\) 正定锥真的可以写成普通三维圆锥

设：

\[
W=
\begin{pmatrix}
a&b\\
b&c
\end{pmatrix}.
\]

定义新坐标：

\[
u
=
\frac{a+c}{\sqrt2},
\]

\[
v
=
\frac{a-c}{\sqrt2},
\]

\[
w
=
\sqrt2\,b.
\]

则：

\[
ac-b^2
=
\frac12
(u^2-v^2-w^2).
\]

正定条件变成：

\[
u>0,
\]

以及

\[
u^2>v^2+w^2.
\]

即：

\[
\boxed{
u>\sqrt{v^2+w^2}.
}
\]

这在 \(\mathbb R^3\) 中就是一个真正的圆锥内部。

---

# 97. Positive-semidefinite cone 与 Positive-definite cone

定义：

\[
\boxed{
\mathbb S_+^p
=
\{
W\in\mathbb S^p:
W\succeq0
\}.
}
\]

它包含边界上的 singular matrices。

严格正定：

\[
\mathbb S_{++}^p
\]

是其内部：

\[
\boxed{
\mathbb S_{++}^p
=
\operatorname{int}
(\mathbb S_+^p).
}
\]

闭包关系：

\[
\boxed{
\overline{\mathbb S_{++}^p}
=
\mathbb S_+^p.
}
\]

二维时：

\[
ac>b^2
\]

是内部；

\[
ac=b^2
\]

是 rank-deficient 边界。

---

# 98. Wishart 密度“定义域”和严格拓扑 support 的区别

非退化 Wishart 的 density interior 是：

\[
\boxed{
W\in\mathbb S_{++}^p.
}
\]

而严格拓扑意义上的 support 通常取密度正区域的闭包：

\[
\boxed{
\operatorname{supp}(W)
=
\mathbb S_+^p.
}
\]

这和 chi-square 很像：

密度公式通常写在

\[
x>0
\]

上，

但拓扑 support 是：

\[
\boxed{
[0,\infty).
}
\]

边界本身通常概率为 0，但仍属于 support 的闭包。

---

# 99. PDF、support 与 probability 的关系

连续标量：

\[
x\mapsto f(x).
\]

连续向量：

\[
\mathbf x\mapsto f(\mathbf x).
\]

连续矩阵：

\[
W\mapsto f(W).
\]

都可以直接把一个具体值代入 PDF 得到密度。

但必须区分：

\[
\boxed{
f(W)
\neq
P(\mathbf W=W).
}
\]

连续分布单点概率：

\[
\boxed{
P(\mathbf W=W_0)=0.
}
\]

真正的概率由区域积分得到：

\[
\boxed{
P(\mathbf W\in A)
=
\int_A f(W)\,dW.
}
\]

对于对称矩阵，可以把 Lebesgue 元写成：

\[
\boxed{
dW
=
\prod_{i\le j}dw_{ij}.
}
\]

所以 Wishart 本质上就是在

\[
\frac{p(p+1)}2
\]

维对称矩阵空间上的普通连续密度，只是 support 被限制在正定锥。

---

# 100. Support 本来就是分布定义的一部分

一般连续分布可以统一写成：

\[
\boxed{
f(x)
=
(\text{解析表达式})
\times
\mathbf 1_{\{x\in \text{support region}\}}.
}
\]

例如 chi-square：

\[
\boxed{
f(x)
=
\frac{
x^{\nu/2-1}e^{-x/2}
}{
2^{\nu/2}\Gamma(\nu/2)
}
\mathbf 1_{\{x>0\}}.
}
\]

Gamma：

\[
f(x)
=
C
x^{\alpha-1}e^{-\lambda x}
\mathbf 1_{\{x>0\}}.
\]

Beta：

\[
f(x)
=
C
x^{a-1}(1-x)^{b-1}
\mathbf 1_{\{0<x<1\}}.
\]

Wishart：

\[
\boxed{
f(W)
=
C
|W|^{(\nu-p-1)/2}
e^{-\frac12\operatorname{tr}(\Sigma^{-1}W)}
\mathbf 1_{\{W\succ0\}}.
}
\]

所以“解析公式里看不出 support”并不是问题；完整 PDF 可以通过 indicator 把 support 明确写进去。

---

# 101. 本轮完整关系总图

\[
\boxed{
\begin{array}{ccccc}
&&\textbf{基础分布}&&\\[2mm]
N&\longrightarrow&N_p\\[1mm]
\chi^2&\longrightarrow&Wishart\\[5mm]

&&\textbf{组合分布}&&\\[2mm]
\displaystyle
\frac{N}{\sqrt{\chi^2/\nu}}
&\longrightarrow&
t
\\[4mm]
\displaystyle
\frac{N_p}{\sqrt{\chi^2/\nu}}
&\longrightarrow&
Multivariate\ t
\\[4mm]
\displaystyle
\frac{\chi^2/m}{\chi^2/n}
&\longrightarrow&
F
\\[4mm]
\displaystyle
\frac{Wishart/m}{Wishart/n}
&\longrightarrow&
Matrix\ F
\\[6mm]

&&\textbf{检验统计量}&&\\[2mm]
t^2
&\longrightarrow&
Hotelling\ T^2
&\longrightarrow&
scaled\ scalar\ F
\\[5mm]
ANOVA
&\longrightarrow&
MANOVA
&\longrightarrow&
H,E
\\[2mm]
&&&
\downarrow
\\[-1mm]
&&&
E^{-1}H
\\[2mm]
&&&
\downarrow
\\[-1mm]
&&&
\lambda_1,\ldots,\lambda_s
\\[2mm]
&&&
\downarrow
\\[-1mm]
&&&
\begin{cases}
Wilks=\prod_i(1+\lambda_i)^{-1}\\[1mm]
Pillai=\sum_i\lambda_i/(1+\lambda_i)\\[1mm]
HL=\sum_i\lambda_i\\[1mm]
Roy=\max_i\lambda_i
\end{cases}
\end{array}
}
\]

---

# 102. 最终压缩总结

整套经典一维到多维统计的骨架可以压缩成：

\[
\boxed{
\text{scalar}
\to
\text{vector/matrix}
}
\]

\[
\boxed{
x^2
\to
\mathbf x\mathbf x^T
}
\]

\[
\boxed{
\sigma^2
\to
\Sigma
}
\]

\[
\boxed{
\chi^2
\to
Wishart
}
\]

\[
\boxed{
t
\to
Multivariate\ t
}
\]

\[
\boxed{
F
\to
Matrix\ F
}
\]

\[
\boxed{
t^2\text{ statistic}
\to
Hotelling\ T^2
\to
\text{scaled scalar }F
}
\]

\[
\boxed{
ANOVA
\to
MANOVA
\to
Wishart/Wishart
\to
Matrix\ F\text{-like roots}
\to
Wilks/Pillai/HL/Roy.
}
\]

对 Wishart 本身：

\[
\boxed{
W=
\sum_i\mathbf z_i\mathbf z_i^T
\in
\mathbb S_+^p.
}
\]

非退化时：

\[
\boxed{
W\in\mathbb S_{++}^p
\quad\text{a.s.}
}
\]

而

\[
\boxed{
\dim\mathbb S^p
=
\frac{p(p+1)}2.
}
\]

Wishart density：

\[
\boxed{
f_W(W)
\propto
|W|^{(\nu-p-1)/2}
e^{-\frac12\operatorname{tr}(\Sigma^{-1}W)}
\mathbf 1_{\{W\succ0\}}.
}
\]

其中 determinant power 来自矩阵变量变换的 Jacobian，trace quadratic term 来自原始 Gaussian exponent。

Matrix \(F\) density：

\[
\boxed{
f_F(F)
\propto
|F|^{(m-p-1)/2}
\left|
I+\frac mnF
\right|^{-(m+n)/2}
\mathbf 1_{\{F\succ0\}}.
}
\]

因此标量、向量、矩阵随机对象都可以拥有直接的解析密度；区别主要在随机对象所在空间以及对应的 support。
