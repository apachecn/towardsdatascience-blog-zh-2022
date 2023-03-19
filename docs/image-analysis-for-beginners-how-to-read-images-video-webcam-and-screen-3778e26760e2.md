# 如何在 OpenCV 中读取图像、视频、网络摄像头和屏幕——初学者图像分析

> 原文：<https://towardsdatascience.com/image-analysis-for-beginners-how-to-read-images-video-webcam-and-screen-3778e26760e2>

## 如何读取 OpenCV 要处理的图像的分步指南

![](img/3961fb933683c409b9797d066b1266b2.png)

计算机视觉！(图片由 [Wolfgang Hasselmann](https://unsplash.com/@wolfgang_hasselmann) 在 [unsplash](https://unsplash.com/photos/JAmijzPKF20) 上拍摄)

OpenCV 是从图像中获取信息的一个很好的工具。在这个系列中，我们将学习如何处理和分析图像，以便我们可以检测运动、模式、模板甚至阅读文本。然而，在这之前，我们需要图像来分析，这也是本文的主题。

我们将为 OpenCV 检查 4 种获取图像的方法，并一步一步地研究它们。最终，您将能够:

*   加载图像
*   加载视频文件
*   阅读您的网络摄像头
*   阅读你的屏幕
*   调整颜色

我们将在本系列的下一部分中使用这些知识来创建一些漂亮的分析。

## 系列

本文是关于 OpenCV 图像处理系列的一部分。查看其他文章:

*   *阅读图像、视频、您的屏幕和网络摄像头(📍你在这里！)*
*   [*检测和模糊人脸*](https://mikehuls.medium.com/image-analysis-for-beginners-detect-and-blur-faces-with-a-simple-function-60ba60753487)
*   [*用模板匹配破坏猎鸭:在图像中寻找图像*](https://mikehuls.medium.com/image-analysis-for-beginners-destroying-duck-hunt-with-opencv-e19a27fd8b6)
*   [*创建运动检测器*](https://mikehuls.medium.com/image-analysis-for-beginners-creating-a-motion-detector-with-opencv-4ca6faba4b42)
*   *检测没有 AI 的形状(在建；即将推出)*
*   *从图像中检测和读取文本*(正在建设中；即将推出)

# 设置

在这一部分中，我们将安装我们的依赖项，并启动和运行 OpenCV。目标是有一个我们可以使用的图像。我们只需要安装两个依赖项:OpenCV 和 PIL (Python 图像库)。将它们安装在:

```
pip install opencv-python pillow
```

![](img/ed5f15d9f55d70f02358dbf995a1e508.png)

阅读时间(图片由[西格蒙德](https://unsplash.com/@sigmund)在 [unsplash](https://unsplash.com/photos/t-da_md1qMc) 拍摄)

# 读取图像

我们将从简单地从文件中读取图像开始。接下来我们开始“拍摄”我们的屏幕。我们可以用它来分析我们在屏幕上看到的图像(我们稍后会谈到这一点)。最后，我们将开始从网络摄像头和视频文件中读取图像。

## 1.加载文件

首先，我们在第 1 行导入 OpenCV(cv2 是 OpenCV 在 Python 实现中的模块名)。

```
import cv2img = cv2.imread(filename="c:/path/to/image.png")
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

在第 3 行中，我们加载了一个图像，在下一行中，我们告诉 cv2 向我们显示它。第 5 行告诉 cv2 等待按钮按下(任何按钮)。如果没有这一行，图像将在 cv2 显示后立即关闭。最后一行用于清理所有打开的窗口。这是结果:

![](img/0fdb1ba9244d583fa2e49fcb6b6026f2.png)

这幅美丽的风景图片的 BGR 版本(图片由 [Hendrik Conrlissen](https://unsplash.com/@the_bracketeer) 在 [Unsplash](https://unsplash.com/photos/-qrcOR33ErA) 上拍摄)

颜色有点不对，不是吗？那条河和天空不应该是美好的蓝色吗？原因是 OpenCV 使用 BGR 而不是 RGB(这在当时的相机制造商和软件中很流行)。这将我们带到我们的第一个图像操作:**从 BGR 转换到 RGB:**

```
img = cv2.imread(filename="c:/path/to/image.png")
img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
cv2.imshow('image', img)
cv2.waitKey(0)
```

通过添加一行，我们转换了图像，这样我们就得到它应该有的样子:

![](img/0a70b2c60c2854a9b4d3ee5b86bbbc5e.png)

好多了(图片由 [Hendrik Conrlissen](https://unsplash.com/@the_bracketeer) 在 [Unsplash](https://unsplash.com/photos/-qrcOR33ErA) 上拍摄)

## 2.拍摄您的屏幕

想为一个游戏创建一个 AI 或者分析视频数据？然后，你可能想阅读你的屏幕，以分析图像。这非常类似于加载文件:

```
import cv2
import numpy as np
from PIL import ImageGrabwhile (True):
    screen = np.array(ImageGrab.grab())
    screen = cv2.cvtColor(src=screen, code=cv2.COLOR_BGR2RGB)
    cv2.imshow('my_screen', screen)

    # press escape to exit
    if (cv2.waitKey(30) == 27):
       break
cv2.destroyAllWindows()
```

在第 6 行我们使用了 PIL 的 ImageGrab.grab()方法；这将返回我们的屏幕。我们必须将它转换成一个 Numpy 数组，这样 OpenCV 才能使用它。然后，在接下来的几行中，我们做和以前一样的事情:转换成 RGB 并告诉 OpenCV 向我们显示图像。然而，在第 9 行和第 10 行，当我们按下 escape 键时，我们告诉 OpenCV 中断循环；这就是`waitKey(30) == 27`的意思。最后一行又是大扫除。

## 3.读取网络摄像头

让我们加快速度；代码与上一部分非常相似。

```
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2RGB)
    cv2.imshow('webcam', frame)# press escape to exit
    if (cv2.waitKey(30) == 27):
       break
cap.release()
cv2.destroyAllWindows()
```

在第 1 行中，我们使用 OpenCV 方法来捕获我们的网络摄像头。然后，我们使用第 3 行中的`read()`方法来接收两个变量:`ret`，如果我们可以读取帧，则为真；以及`frame`。接下来，我们再次转换我们的颜色(这一次是灰色)，显示帧并等待我们的退出键进行清理。

## 4.读取视频文件

读取一个视频文件与读取你的网络摄像头是一样的，唯一的区别是我们必须传递文件的位置:

```
cap = vc2.VideoCapture('c:/path/to/my/file.mp4
```

# 结论

在本系列的第一部分中，我们已经处理了将图像导入 OpenCV 的绝对基础。在接下来的部分，我们将做一些漂亮的预处理和分析。

> 不要忘记查看本系列的其他文章！

如果你有建议/澄清，请评论，以便我可以改进这篇文章。同时，看看我的其他关于各种编程相关主题的文章，比如:

*   [Python 为什么慢，如何加速](https://mikehuls.medium.com/why-is-python-so-slow-and-how-to-speed-it-up-485b5a84154e)
*   [Python 中的高级多任务处理:应用线程池和进程池并进行基准测试](https://mikehuls.medium.com/advanced-multi-tasking-in-python-applying-and-benchmarking-threadpools-and-processpools-90452e0f7d40)
*   [编写自己的 C 扩展来加速 Python x100](https://mikehuls.medium.com/write-your-own-c-extension-to-speed-up-python-x100-626bb9d166e7)
*   【Cython 入门:如何在 Python 中执行>每秒 17 亿次计算
*   [用 FastAPI 用 5 行代码创建一个快速自动归档、可维护且易于使用的 Python API](https://mikehuls.medium.com/create-a-fast-auto-documented-maintainable-and-easy-to-use-python-api-in-5-lines-of-code-with-4e574c00f70e)
*   [创建并发布你自己的 Python 包](https://mikehuls.medium.com/create-and-publish-your-own-python-package-ea45bee41cdc)
*   [创建您的定制私有 Python 包，您可以从您的 Git 库 PIP 安装该包](https://mikehuls.medium.com/create-your-custom-python-package-that-you-can-pip-install-from-your-git-repository-f90465867893)
*   [面向绝对初学者的虚拟环境——什么是虚拟环境以及如何创建虚拟环境(+示例)](https://mikehuls.medium.com/virtual-environments-for-absolute-beginners-what-is-it-and-how-to-create-one-examples-a48da8982d4b)
*   [通过简单升级，显著提高数据库插入速度](https://mikehuls.medium.com/dramatically-improve-your-database-inserts-with-a-simple-upgrade-6dfa672f1424)

编码快乐！

—迈克

页（page 的缩写）学生:比如我正在做的事情？[跟着我！](https://mikehuls.medium.com/membership)

<https://mikehuls.medium.com/membership> 