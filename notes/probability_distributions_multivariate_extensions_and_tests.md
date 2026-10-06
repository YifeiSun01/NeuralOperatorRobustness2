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
