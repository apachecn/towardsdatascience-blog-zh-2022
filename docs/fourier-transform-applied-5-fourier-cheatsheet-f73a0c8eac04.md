# 傅立叶变换，应用(5):傅立叶备忘单

> 原文：<https://towardsdatascience.com/fourier-transform-applied-5-fourier-cheatsheet-f73a0c8eac04>

## Numpy 的傅里叶变换函数之旅

为了结束这个系列，让我们回顾一下最常见的数字傅里叶变换函数。

查看该系列的前几篇文章！

*   [https://towards data science . com/the-Fourier-transform-1-ca 31 ADB FB 9 ef](/the-fourier-transform-1-ca31adbfb9ef)
*   [https://towards data science . com/the-Fourier-transform-2-understanding-phase-angle-a 85 ad 40 a 194 e](/the-fourier-transform-2-understanding-phase-angle-a85ad40a194e)
*   [https://towards data science . com/the-Fourier-transform-3-magnitude and-phase-encoding-in-complex-data-8184 e2ef 75 f 0](/the-fourier-transform-3-magnitude-and-phase-encoding-in-complex-data-8184e2ef75f0)
*   [https://towards data science . com/the-Fourier-transform-4-put-the-FFT-to-work-38dd 84 DC 814](/the-fourier-transform-4-putting-the-fft-to-work-38dd84dc814)

![](img/b0341e5529be4627822b841e8ec8a089.png)

照片由[阿尔方斯·莫拉莱斯](https://unsplash.com/@alfonsmc10?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

# 微妙的区别

到目前为止，在我们的讨论中，我已经交替使用了“傅立叶变换”和“快速傅立叶变换(FFT)”。在这一点上，值得注意的区别！FFT 是“离散”傅立叶变换(DFT)的有效算法实现。“离散”表示我们可以将变换应用于一系列点，而不是完整、连续的信号。在数据科学应用中，我们通常有一组样本，而不是连续的输入函数，所以我们通常对 DFT 感兴趣！

# 傅立叶变换函数

## np.fft.fft

> FFT 将时域或空域中的实数或复数信号作为输入，并返回离散傅立叶变换。如前所述，np.abs 允许我们恢复频率成分的幅度，np.angle 允许我们恢复相位。

我们以前见过这个函数(见本系列早期的故事)！我们会回到这个注意，但它是足够重要的说两次！傅立叶变换可以应用于复杂的输入信号。对于复数输入，傅里叶变换返回的负频率项是完全重构信号所必需的。对于实际输入——就像我们在本系列中讨论过的输入——只需要正频率项。您仍然可以对实值信号使用完整的 FFT，只需知道您可以使用重复值更少的 RFFT(参见下文)。

**索引 0:** 第一个值有时被称为“DC”项(来自电气工程领域的“直流”)，并用作偏移项。它不振荡，相当于 0Hz(即 DC)。它只是信号的总和！

**索引 1 到((N/2) -1)如果 N 是偶数，否则索引 1 到((N-1)/2):** 正频率分量按递增的正顺序排列。

**索引(N/2)到(N-1)如果 N 是偶数，否则((N+1)/2)到(N-1):** 负频率分量按负的递减(即正的递增)顺序排列。

> 提示！如果您对哪些元素对应于哪些频率感到困惑，请查看 np.fft.fftfreq！它会告诉你频率的顺序，例如 np.fft.fftfreq(5)告诉我们频率仓中心为 0。、0.2、0.4、-0.4 和-0.2。

## np.fft.ifft

> IFFT 将傅立叶变换作为输入，并返回时域或空域中的实数或复数重构信号。

如前所述，逆 FFT 允许我们从频域转换回时间/空间域。不出所料，如果我们将 IFFT 应用于信号的 FFT，我们又回到了起点。

IFFT(FFT(x)) ≈ x，逆性质成立！

## np.fft.fftshift

> FFT 变换将傅立叶变换作为输入，并将值从“标准”顺序重新排序为“自然”顺序:最负到零到最正。

如果组件按自然顺序排序，那么对 FFT 结果进行可视化和推理会容易得多。标准订单可能会非常混乱！

FFTSHIFT 从最负到最正排列频率中心。

## np.fft.ifftshift

> FFT 变换将傅立叶变换作为输入，并将值从“自然”顺序重新排序为“标准”顺序:DC 项，然后是正频率，然后是负频率。

IFFTSHIFT 恢复“标准”顺序。

## np.fft.rfft

> RFFT 将时域或空域中的实信号作为输入，并返回离散傅立叶变换。

本文前面提到过，实信号的 FFT 有一个有趣的特性:正负频率分量互为镜像。形式上，这意味着实信号的傅立叶变换是“[厄米变换](https://en.wikipedia.org/wiki/Hermitian_function)”这其中的数学原因非常有趣，我希望在以后的文章中对此进行详细阐述。不过现在，我们只需要注意到这是真的。RFFT 允许我们跳过这些多余的术语！

RFFT 利用厄米对称性来跳过重复的负频率分量。

## np.fft.irfft

> IRFFT 将实值函数的傅立叶变换作为输入，并在时域或空域中返回实重构信号。

IRFFT(RFFT(x)) ≈ x，逆性质成立！

## NP . FFT . FFT T2

> FFT2 将时域或空域中的实数或复数 2D 信号作为输入，并返回离散傅立叶变换。如前所述，np.abs 允许我们恢复频率成分的幅度，np.angle 允许我们恢复相位。

在[之前的文章](/the-fourier-transform-4-putting-the-fft-to-work-38dd84dc814)中，我们展示了我们可以使用相同的逻辑来分解 2D 信号中频率成分的幅度和角度(比如图像！).np.fft.fft2 允许我们这样做:计算输入的二维快速傅立叶变换。

## np.fft.ifft2

> IFFT 将傅立叶变换作为输入，并返回时域或空域中的实数或复数 2D 重构信号。

正如所料，我们有一个类似的二维输入的逆变换！

IFFT2(FFT2(x)) ≈ x，2D 逆性质成立！

## np.fft.fftn

我们可以将 FFT 扩展到三维、四维、五维输入！事实上，对于一个 *n* 维输入，有一个 *n-* 维 FFT…

## np.fft.ifftn

…以及相应的一个 *n* 维逆变换。

IFFTN(FFTN(x)) ≈ x，n-D 逆性质成立！

感谢您的参与，我们已经完成了这个系列！接下来我想写一些关于傅立叶变换实现的内容。如果你感兴趣，或者有其他概念需要我解释，请在下面留下评论。

你可能也会对我其他一些关于傅立叶直觉的文章感兴趣！