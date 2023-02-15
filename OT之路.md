## OT 综述
https://new.qq.com/rain/a/20220613A02N0700

## 知乎OT近几年进展总结
https://www.zhihu.com/column/DCF-tracking

## VOT 竞赛介绍PPT
[链接](C:/Users/Xiaodong/Desktop/文献/OT/vot2020-st.pdf)

## 2019 年图形图像学报上的一篇中文综述

http://www.cjig.cn/html/jig/2019/12/weixin/20191201.htm


## paper阅读专业词汇

MI(互信息)：

![](https://img-blog.csdnimg.cn/0e77893134d9408b884ef9a2e43a254e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAU3VrdXJh44CBeGlh,size_20,color_FFFFFF,t_70,g_se,x_16)

两个图片或者向量重叠的信息，代表两个图片相似度的。互信息不受强度的感染，单纯从信息论的角度度量图片差异。

SIFT 特征：
[CSDN](https://visionary.blog.csdn.net/article/details/118794045?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-118794045-blog-104771096.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-118794045-blog-104771096.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=1)

## SURF特征

[CSDN](https://blog.csdn.net/bblingbbling/article/details/112749705)


## FHOG
https://cloud.tencent.com/developer/article/1327453


# Matlab Benchmark 的使用



# Matlab KCF 代码释义


## CF类Tracking 整理


CF类Tracking指Correlation Filter Tracking。 一般被认为是从发表在CVPR2010上的MOSSE算法开始的。

追踪任务是指以第一帧指定的选定框开始，连续追踪该物体在之后帧的位置。

### MOSSE算法：

该算法将其看作是一个判别问题，即尝试训练一个模板H，该模板可以将目标与背景分开。

即：

$$
G=F \odot H^*
$$

$
F，G，H均是f,g,h 经2DFFT变换后的频域图。\odot 代表元素间的乘积。f是图片，h是模板，g是响应（一般是以目标框中心为原点的2D高斯响应）。
$


放到频域上是为了加速计算。


因为这里的h是一个和原图f是一样大小的，并且f和g要做卷积。为什么f，g要做卷积，而不是直接相乘，如果f，h直接相乘，得到的g矩阵是没有意义的，可以想象滤波器的第一列和图像第一行相乘到g的左上角的值，这个值能说明什么那。

如果f，h做卷积，相当于f，h在左上角做一次相关运算(对应位置元素相乘后再加在一起，实际上卷积和相关运算有一个模板旋转180°的差异，但是这里简略说成相关运算无大碍)，得到一个值，该值可以代表在模板和左上角这个f的相关性。

很显然，图像卷积在边界上会有一些问题，计算f，g在左上角这个像素的卷积结果时，会出现f的行列负数索引，这是没有定义的，如果以0为边界定义卷积，相当于不同位置的卷积操作的大小是不一样的，为了解决这个问题，在CNN里，是用padding解决的，因为CNN的filter普遍是1，3，5，7,只有边界存在上述问题，所以padding几圈像素就OK了。

但是这里h需要和f一样大，我们希望h能考虑到所有f的像素，换CNN的话说，就是感受野越大越好，因为目标往往在第一帧里会占据很大的面积，模板h应该学习到整个目标才可以。当然如果第一帧框的很大的，目标在中心的面积较小，那么h可以小于f，所以后面会看到，所有算法一般会进行padding，即在目标框选的大小基础上扩大1.5倍。(实际上如果h比f小了，那后面还会有问题)

这样的话就相当于我有个模板h，从f的左上角开始扫描，从左到右，从上到下，得到每个位置的相关性大小，这个结果g应该最好是一个高斯响应，即中间的相关性最大，周围逐渐降低。


现在我有了g，有了f，要算出h，算出h，就相当于得到了一个可以区分前景(即目标)和背景的滤波器，将h应用到新的一帧f上时，会得到一个新的响应g'，这个g'上做大的位置就可以当作下一帧的位置。


上述这种卷积，乘法次数是 $W_f*H_f*W_h*W_h，W、H是f，h的宽高$。

所以要放在频域上进行，这样可以将卷积变成频域的元素间乘法，快了很多倍。这里有一个点，就是为什么CNN不放在频域上进行？因为CNN的核太小了，没有意义，一个复数运算需要4次乘法，如果CNN的核小于原图的1/3，那么卷积定理的收益就相当于没有了。

这个H其实可以通过下式解出了。

$$
H_i^*=\frac{G_i}{F_i}
$$

但是MOSSE里作者还提到了ASEF和UMACE两种生成模板H的算法。


ASEF算法，即Average Synthetic Exact Filters ，就是取多个f，然后制作对应的g，得到不同的Hi，最后取这些Hi的平均：

$$
H_{\text {final }}=\frac{\sum_{i=0}^N H_i}{N}
$$

![](./ASEFfig1.png)

ASEF是做眼睛位置定位的，所以上图所有g的高峰都标在眼睛上了，最后得到的h也是个眼睛。


基于这个算法的启发，为了增加鲁棒性，可以对原图做很多小改变，如旋转一点之类的，得到多个f和g。构造下述问题：

$$
\min _{H^*} \sum_i\left|F_i \odot H^*-G_i\right|^2
$$


得到了一个闭式解：

$$
H^*=\frac{\sum_i G_i \odot F_i^*}{\sum_i F_i \odot F_i^*}
$$


![](./MOSSEfig2.png)

可以看到MOSSE，ASEF得到的模板其实很像，但是MOSSE训练起来会更好。
因为mosse把多个响应结果相加作为loss，得到的H是在分母上做加法，

而ASEF的求解是下式：

$$
H^*=\frac{1}{N} \sum_i \frac{G_i \odot F_i^*}{F_i \odot F_i^*}
$$

万一$Fi \odot Fi^* =0或约等于0$，那么解就不稳定了，相反MOSSE的求解分母是相加，就不容易出现这种问题。



而且，看上去ASEF和MOSSE提出来的filter没有naive的和UMACE方法的更像一个鱼，但是其实前两者鲁棒性更强，因为太明显的特征特异性太强，但凡物体有一些变化响应就会锐减，这也是为什么，MOSSE一定要选择多个f，构造最小平方误差loss的原因——增强泛化能力。
__因为强大的卷积定理，MOSSE运行速度极快，达到了600+FPS__

### 思考：

之前说的H可以比F小其实并不能实现，因为FFT要求两个序列是等长的，不等长的会给短的补0，也就是说，用了卷积定理，算出来的H还是和F等大的，从$
H^*=\frac{\sum_i G_i \odot F_i^*}{\sum_i F_i \odot F_i^*}
$就能看出来。

其次，FFT要求周期性，所以实际上，相当于是和下图做卷积。红框是卷积核大小，和中心的原图一样大，最后得到的g的尺寸也和原图一样大。但是g肯定不应该能是高斯响应，应该是四个角和中心都高，其他地方小的响应。MOSSE，包括后来的CF方法都是靠给原图加余弦窗解决的，就是让图像的周围逐渐变成0. 但是这会导致边界的背景信息学不到了。———— _这就是CF类方法普遍存在的边界问题_

### MOSSE之后，最重要的CF算法莫过于CSK

很多文章都在说CSK或者KCF的关键在于循环矩阵的引入，但实际上从前述MOSSE的分析来看，循环矩阵是被隐含在卷积定理中的，作者自己也说:
>>
 Patnaik and Casasent [16] investigate this
 problem, and show that, given the Fourier representation of an image, many
classical filters cannot be kernelized. Instead, they propose a kernelized filter
that is trained with a single subwindow (called Kernel SDF).

We believe that the method we propose achieves this goal. We are able to de-
vise Kernel classifiers with the same characteristics as correlation filters, namely
their ability to be trained and evaluated quickly with the FFT.

因此，实际上CSK的关键应该是把核技巧带到了CF框架中。


CSK中，作者把问题建模为：
$$
\min _{\mathbf{w}, b} \sum_{i=1}^m L\left(y_i, f\left(\mathbf{x}_i\right)\right)+\lambda\|\mathbf{w}\|^2
$$

$L代表loss，y_i 代表理想输出，f(x_i)代表对x_i的预测输出，w是滤波器$，可以发现比MOSSE多了一个正则项。

并且这里的LOSS可以选择，如果使用$L(y, f(\mathbf{x}))=\max (0,1-y f(\mathbf{x}))$，则是SVM模型，如果是
$L(y, f(\mathbf{x}))=(y-f(\mathbf{x}))^2$，则是RLS岭回归模型。二者的性能表现差不多。RLS也可以看作是带L1正则的最小平方差和loss，好处是能减少w的过拟合问题。

如果预测模型$f$是线性的,即$f(x)=w^TX$，则上述问题有个很好的闭式解：
$$
w=(X^TX+\lambda I)^{-1}X^Ty)
$$


如果$f$是非线性的，__任何一个非线性问题都可以看作是高维的线性问题__，因此可以通过非线性映射$\phi(x)$映射到一个高维线性空间，把问题变成$f(x)=w^T \phi(X)$

$$
w=(\phi(X^T)\phi(X)+\lambda I)^{-1}\phi(X^T)y)

$$

而由表示定理$w=\alpha^T\phi(x)$,


$$
\boldsymbol{\alpha}=(K+\lambda I)^{-1} \mathbf{y},K是\phi(X^T)\phi(X)
$$

同时$f(x)$可以写成：
$$

f(x)=\alpha \phi(X)^T\phi(X)=\alpha K(X,X)

$$

这里的K叫核矩阵，可以用核技巧比较简单的计算得到，但不一定严格等于$\phi(X^T)\phi(X)$，对应的$\phi$可以是高斯核也可以是线性核等。


但是这里要求逆，求逆的运算量是$O(n^3)$,相当于$\alpha$计算量还是$O(n^4)$.


但是如果X是循环采样的，那么求逆就可以变成频域的元素间运算。

现在设$u是一个n维行向量$，那么其循环矩阵定义为
$$
C(\mathbf{u})=\left[\begin{array}{ccccc}
u_0 & u_1 & u_2 & \cdots & u_{n-1} \\
u_{n-1} & u_0 & u_1 & \cdots & u_{n-2} \\
u_{n-2} & u_{n-1} & u_0 & \cdots & u_{n-3} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
u_1 & u_2 & u_3 & \cdots & u_0
\end{array}\right]
$$

现在有一个同样长的向量v，那么$C(u)v$虽然是乘积，但就是u，和v的卷积定义，就可以应用卷积定理：
$$
C(\mathbf{u}) \mathbf{v}=\mathcal{F}^{-1}\left(\mathcal{F}^*(\mathbf{u}) \odot \mathcal{F}(\mathbf{v})\right),
$$

上述$X$取成循环矩阵就可以了。

但是没用啊，还得求逆。要是$(K+\lambda I)$是个循环矩阵，那么就可以变换到频域上，用元素间除法了,$I$是对角阵，则K循环的是就都是循环的。

CSK作者给了很好的证明，如下：

Theorem 1. The matrix $K$ with elements $K_{i j}=\kappa\left(P^i \mathbf{x}, P^j \mathbf{x}\right)$ is circulant if $\kappa$ is a unitarily invariant kernel.

Proof. A kernel $\kappa$ is unitarily invariant if $\kappa\left(\mathbf{x}, \mathbf{x}^{\prime}\right)=\kappa\left(U \mathbf{x}, U \mathbf{x}^{\prime}\right)$ for any unitary matrix $U$. Since permutation matrices are unitary, $K_{i j}=\kappa\left(P^i \mathbf{x}, P^j \mathbf{x}\right)=$ $\kappa\left(P^{-i} P^i \mathbf{x}, P^{-i} P^j \mathbf{x}\right)=\kappa\left(\mathbf{x}, P^{j-i} \mathbf{x}\right)$. Because $K_{i j}$ depends only on $(j-i) \bmod n$, $K$ is circulant.

则就可以用下式元素间的除法计算$\alpha$了
$$
\boldsymbol{\alpha}=\mathcal{F}^{-1}\left(\frac{\mathcal{F}(\mathbf{y})}{\mathcal{F}(\mathbf{k})+\lambda}\right)
$$

有了模板$\alpha$,还可以快速计算对应于新的一帧$z$的响应：

$$
\hat{\mathbf{y}}=\mathcal{F}^{-1}(\mathcal{F}(\overline{\mathbf{k}}) \odot \mathcal{F}(\boldsymbol{\alpha})),
$$
where $\overline{\mathbf{k}}$ is the vector with elements $\bar{k}_i=\kappa\left(\mathbf{z}, P^i \mathbf{x}\right)$. We provide an extended proof in Appendix A.2. Just like the formula for KRLS training, the complexity is bound by the FFT operations and is only $\mathcal{O}\left(n^2 \log n\right)$ for 2 D images.

如果不用核函数，也就是说假设原问题是纯线性的，那么得到的解w和MOSSE完全一样。

$$
\mathbf{w}=\mathcal{F}^{-1}\left(\frac{\mathcal{F}(\mathbf{x}) \odot \mathcal{F}^*(\mathbf{y})}{\mathcal{F}(\mathbf{x}) \odot \mathcal{F}^*(\mathbf{x})+\lambda}\right) .
$$
This is a kind of correlation filter that has been proposed recently, called Minimum Output Sum of Squared Error (MOSSE) [12, 15], with a single training image. It is remarkably powerful despite its simplicity.


综上，CSK的主要作用是引入了核技巧，并且揭示了卷积定理和循环矩阵的之间的本质关系。

看看实验结果，其实引入核技巧，提升并没有很明显。

![](CSK%E5%AE%9E%E9%AA%8C%E7%BB%93%E6%9E%9C.png)

### KCF

KCF是CSK作者本人在2014年提出来的新算法，主要是不在使用灰度特征了，而是使用FHOG特征，FHOG是TPAMI上2010年的一篇行人检测文章，引用量高达10000+。 基本原理没有区别，而KCF原文对FHOG也是一笔带过，可能因为是CSK发在ECCV2012上，而KCF发在TPAMI上，所以KCF需要再叙述一遍基础原理。

FHOG特征:

传统的HOG特征，是将图片以cell(可以是n*n,一般是4*4)为单位，统计每个cell里的加权梯度方向直方图，即如果是9个bins，就是指将360°分为九个区间，统计cell里所有像素梯度方向的分布，并用梯度幅度进行加权，得到一个9维的向量。把所有cell的9维向量串起来就是HOG特征描述。

FHOG特征，显示计算了27=18+9个bins，18个方向敏感和9个方向不敏感。

$$
\begin{aligned}
& B_1(x, y)=\operatorname{round}\left(\frac{p \theta(x, y)}{2 \pi}\right) \bmod p \\
& B_2(x, y)=\operatorname{round}\left(\frac{p \theta(x, y)}{\pi}\right) \bmod p
\end{aligned}
$$

$B1是方向敏感，p=19，B2是方向不敏感，p=9$，这里分成18+9的原因是，有的时候同一个物体的轮廓可能 存在梯度反转现象，如红外和RGB图的很可能梯度是反的(比如人的区域亮度比较低，RGB上人物轮廓的梯度应该是指向外侧的，但是红外图上人物区域可能比较亮，轮廓梯度是指向内侧的，但这无疑都是代表一个人的特征)，18个在360°的是可以区分内外梯度指向的，剩下9个是无法区分的，分成梯度敏感和不敏感是为了增加鲁棒性。

同时还可以做三线性插值，对每个bin进行邻域插值。

这里作者还提出了像素级的特征表示，和CFOG很像。而CFOG用3D高斯核做线性插值，速度更快，效果可能差不多。

![](./FHOG%E9%87%8C%E7%9A%84%E5%83%8F%E7%B4%A0%E7%BA%A7%E7%89%B9%E5%BE%81.png)

然后每个cell和周围左上，右上，左下，右下四个方向的各四个cell进行归一化和截断（截断：大于某个阈值的固定为上限）

![](./FHOG%E6%88%AA%E6%96%AD.png)

这里的归一化截断是为了减少光照（区域变强或变弱）的影响的同时能表示更大范围的梯度特征。这样一个cell可以变成一个36维的特征。

之后作者通过PCA分析，发现这36个特征主成分基本全部集中在前11个（这显然易见，四次方向的归一化截断，一定存在大量相关的维度），然后通过分析发现，按照其可以变成更有计算意义的13维特征，这13维和11维效果基本一致。

将36维特征表示为4*9的矩阵，则这13维新的特征可以表示为，每列的和9个，每行和合4个，分表代表了四个方向的总能量，以及9个bin梯度的平均分布。