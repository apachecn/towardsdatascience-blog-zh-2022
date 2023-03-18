# Python 项目:如何在图像中隐藏秘密数据

> 原文：<https://towardsdatascience.com/python-project-how-to-hide-secret-data-in-images-2a835ed6b9b1>

## 开发 Python 项目来加密和解密隐藏在图像中的信息

![](img/a06b907b232c93f77f782861ca555f1d.png)

斯蒂芬·斯坦鲍尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

安全和隐私是现代世界的两大关注点。如果你想把某种秘密信息传递给你的朋友或家人，但又不想让其他人知道，最好的方法是什么？—在本文中，我们将开发一个有趣的 Python 项目，它可以安全地加密图像中的文本信息，以便它可以成功地传输给接收者，接收者可以相应地解码它。

通过加密编码在图像中隐藏信息和秘密数据的技术通常被称为**隐写术**。一旦秘密信息被编码在图像中，只有专家或接收带有秘密密钥的数据的终端用户才能解码(或解密)该秘密信息。下面是来自 [wiki](https://en.wikipedia.org/wiki/Steganography) 的更正式的隐写术定义

> **隐写术**是将一条消息隐藏在另一条消息或物理对象中的做法。在计算/电子环境中，计算机文件、消息、图像或视频隐藏在另一个文件、消息、图像或视频中。

在本文中，我们将致力于构建一个有趣的 Python 项目，从中我们可以了解隐写术的基础知识以及如何在图像中隐藏秘密信息。只要发送方和接收方(目的地)共享公共密钥，您就可以加密您的秘密数据，并与您的朋友一起解密所需的信息。在下一节中，我们将详细了解如何构建这个项目。

# 图像中数据隐藏的应用；

在我们开始隐写术项目之前，让我们看看一些关键的应用和用例，您可能会发现隐藏在图像中的数据很有用。

1.  两个用户之间的机密通信
2.  数据变更保护
3.  您还可以使用图像隐写术在图像中隐藏图像
4.  为了确保机密数据的存储
5.  用于数字内容分发的访问控制系统

现在我们已经对隐写术的一些用例有了一个简单的概念，我们可以进入下一部分，开始开发这个项目。

# 发展隐写术项目:

![](img/c48c8e9389a3cd1ed71890d1e1b82d3a.png)

Anukrati Omar 在 [Unsplash](https://unsplash.com/s/photos/scenary?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片

在文章的这一部分，我们将学习如何通过隐藏在图像中来加密和解密秘密的文本数据和信息。我们将把这个部分分成几个部分，以便更容易理解成功开发这个项目需要遵循的所有基本步骤。在我们开始之前，请确保将上面的图像下载到您的工作目录中，并在。jpg 格式。现在，事不宜迟，让我们开始隐写术项目。

## 导入基本库:

该项目的第一步是导入完成任务所需的所有基本库。最重要的库模块是 Open-CV (cv2 ),用于计算机视觉应用，如图像操作、读取和写入图像以及其他关键操作。字符串库可用于处理字符串数据和 ASCII 值。OS 库对于管理与操作系统相关的功能，以便在 Windows 或其他类似平台中保存或创建文件非常有用。

```
# Importing the essential libraries
import cv2
import string
import os
```

如果您不完全熟悉 Open-CV 库，我强烈推荐您通过下面提供的链接查阅我以前的一篇文章，这篇文章涵盖了这个概念的全部基础知识。掌握这个库的知识是掌握 Python 中计算机视觉的最重要的步骤之一。

[](/opencv-complete-beginners-guide-to-master-the-basics-of-computer-vision-with-code-4a1cd0c687f9) [## OpenCV:用代码掌握计算机视觉基础的完全初学者指南！

### 包含代码的教程，用于掌握计算机视觉的所有重要概念，以及如何使用 OpenCV 实现它们

towardsdatascience.com](/opencv-complete-beginners-guide-to-master-the-basics-of-computer-vision-with-code-4a1cd0c687f9) 

## 声明和定义所需的变量和参数:

在本文的这一部分，我们将声明计算项目所需的所有变量和参数。首先，我们将声明两个字典数据结构，如下面的代码块所示。第一个字典将 ASCII 对象及其各自的 id 作为键存储，而第二个字典执行类似的任务，反之亦然。你可以打印这两本字典来帮助弄清楚它们是如何工作的。

在同一个代码块中，我们可以继续读取之前存储在工作目录中的图像“test_image.jpg”。分析图像的参数通常是一种健康的做法，比如高度、宽度和通道数。图像的通道数定义了它是灰度图像还是 RGB 图像。下面是执行以下操作的代码块。

```
# Declaring the essential Characters
dict1 = {}
dict2 = {}for i in range(255):
    dict1[chr(i)]=i
    dict2[i]=chr(i)# print(dict1)
# print(dict2)# Reading and analyzing our image
img = cv2.imread("test_image.jpg")height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]print(f"Height: {height}, Width: {width}, Number of channels: {channels}")
```

## 输出:

```
Height: 3585, Width: 5378, Number of channels: 3
```

如果您已经按照教程学习了这么久，那么在运行下面的代码时，您应该会获得下面的输出参数。如果用户不太熟悉数据结构和字典的概念，我建议查看我以前的一篇文章，通过下面的链接更熟悉这个概念。

[](/mastering-dictionaries-and-sets-in-python-6e30b0e2011f) [## 掌握 Python 中的字典和集合！

### 通过代码和示例理解 Python 中的字典和集合的概念

towardsdatascience.com](/mastering-dictionaries-and-sets-in-python-6e30b0e2011f) 

## 加密:

现在我们已经定义了所有需要的变量和参数，我们可以继续在我们想要的图像中添加一些加密的文本数据。我们将输入一个共同的密钥，最终用户必须知道这个密钥，这样他们才能读取加密和解密的数据。您可以添加您的关键和各自的文本，你想隐藏在图像中。

这项任务的加密算法非常简单。我们利用实际输入文本和之前声明的字典值提供的键之间的按位操作。我们定义的 x、y 和 z 值用于确定文本在图像中是垂直、水平还是对角隐藏。在每次模运算迭代之后，密钥也被修改。

```
# Encryption
key = input("Enter Your Secret Key : ")
text = input("Enter text to hide In the Image : ")kl=0
tln=len(text)
x = 0 # No of rows
y = 0 # no of columns
z = 0 # plane selectionl=len(text)for i in range(l):
    img[x, y, z] = dict1[text[i]] ^ dict1[key[kl]]
    y = y+1
    x = x+1
    x = (x+1)%3
    kl = (kl+1)%len(key)

cv2.imwrite("encrypted_img.jpg", img) 
os.startfile("encrypted_img.jpg")
print("Data Hiding in Image completed successfully.")
```

最后，一旦我们执行了加密算法，我们就可以将加密的图像保存在工作目录中。但是，一旦图像打开，您会发现几乎不可能注意到原始图像和加密图像之间的任何差异。文本隐藏得很好，图像中的秘密信息可以与解密过程所需的目的地共享。

## 解密:

![](img/ef32dca06e6dca9d657c6868424be034.png)

作者加密的图像

一旦信息在相应的图像中被加密，我们就可以继续执行解密。请注意，只有当发送方和接收方具有相同的输入密钥时，才能成功解密。如果目的地的接收者输入了正确的密钥，则执行下面代码块中的解密算法。此任务的方法类似于加密代码，因为我们用第二个和第一个字典的按位操作来反转前面的操作。

```
# Decryption
kl=0
tln=len(text)
x = 0 # No of rows
y = 0 # no of columns
z = 0 # plane selectionch = int(input("\nEnter 1 to extract data from Image : "))if ch == 1:
    key1=input("\n\nRe-enter secret key to extract text : ")
    decrypt=""if key == key1 :
        for i in range(l):
            decrypt+=dict2[img[x, y,z] ^ dict1[key[kl]]]
            y = y+1
            x = x+1
            x = (x+1)%3
            kl = (kl+1)%len(key)print("Encrypted text was : ", decrypt)
    else:
        print("Enter Key doesn't match the original records.")
else:
    print("Exiting the code...")
```

如果代码(密钥)输入正确，图像中的文本信息将被相应地解密和解码。您将收到最初在相应映像中加密的数据。但是，如果您没有正确的密钥，程序将终止，并让您知道输入了错误的密钥。

## 最终结果、改进和未来工作:

![](img/b765a21aabfe8245c073abcf965a7f62.png)

作者图片

上面的截图显示了下面的隐写术项目的工作方法。上图显示了加密和解密过程，以及相应的密钥和文本数据。请随意尝试各种变化，测试他们的工作程序。我强烈推荐查看下面的 GitHub [链接](https://github.com/pypower-codes/Steganography/blob/master/stagenography_1.py)，这段代码的大部分都是从这里开始的。

为了进一步改进，将这种想法扩展到图像之外会很棒，比如 pdf 文件、文本文件、gif 等等。通过使用 GUI 界面，您还可以添加几个额外的项目来使这个项目更具展示性。最后，您可以扩展代码，使加密和解密过程变得更加复杂。

密码学是一个庞大的课题，还有很多安全措施对于用户在网上或现实生活中维护他们的隐私和安全是必不可少的。虽然这样的项目是一个很好的起点，但是检查其他类似的项目来增加您的安全措施也是一个很好的主意。我建议通过下面提供的链接查看我的另一个关于使用 Python 生成高度安全的密码的项目。

[](/highly-secure-password-generation-with-python-852da86565b9) [## 使用 Python 生成密码

### 在大约 5 分钟内为所有重要文件和网上交易生成安全密码

towardsdatascience.com](/highly-secure-password-generation-with-python-852da86565b9) 

# 结论:

![](img/89745f79558259f783139f54cec91c2f.png)

摄影:[飞:D](https://unsplash.com/@flyd2069?utm_source=medium&utm_medium=referral) 上 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

> “我已经变得喜欢保密。对我们来说，这似乎是能使现代生活变得神秘或不可思议的一件事。最普通的东西是令人愉快的，如果只有一个隐藏它。”
> *——****王尔德***

在现代社会，安全和隐私至关重要。为了确保每个人都能够安全可靠地使用众多日常生活设施，例如互联网、信用卡或借记卡、电话和其他类似设备，人们正在研究大量的发展来保持这种安全性。使用 Python 编码的隐写术是众多这类有趣项目中的一个，你可以在图像中隐藏一些秘密信息并共享数据。

在本文中，我们了解了图像数据隐藏和隐写术的一些基本概念。我们还获得了执行这样一个项目所需的计算机视觉库的知识。我建议尝试这个项目的许多变化，通过有趣的实验和探索来测试各种可能性。

如果你想在我的文章发表后第一时间得到通知，请点击下面的[链接](https://bharath-k1297.medium.com/subscribe)订阅邮件推荐。如果你希望支持其他作者和我，请订阅下面的链接。

[](https://bharath-k1297.medium.com/membership) [## 通过我的推荐链接加入媒体

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

bharath-k1297.medium.com](https://bharath-k1297.medium.com/membership) 

如果你对这篇文章中提到的各点有任何疑问，请在下面的评论中告诉我。我会尽快给你回复。

看看我的一些与本文主题相关的文章，你可能也会喜欢阅读！

[](/7-python-programming-tips-to-improve-your-productivity-a57802f225b6) [## 提高生产力的 7 个 Python 编程技巧

### 通过修正一些常见的不良编程实践，使您的 Python 编码更加有效和高效

towardsdatascience.com](/7-python-programming-tips-to-improve-your-productivity-a57802f225b6) [](/develop-your-own-calendar-to-track-important-dates-with-python-c1af9e98ffc3) [## 使用 Python 开发您自己的日历来跟踪重要日期

### 开发一个日历 GUI 界面来管理您 2022 年及以后的计划

towardsdatascience.com](/develop-your-own-calendar-to-track-important-dates-with-python-c1af9e98ffc3) [](/develop-your-weather-application-with-python-in-less-than-10-lines-6d092c6dcbc9) [## 用 Python 开发不到 10 行的天气应用程序

### 使用 Python 构建我们的天气电视广播应用程序，以接收所需位置的更新

towardsdatascience.com](/develop-your-weather-application-with-python-in-less-than-10-lines-6d092c6dcbc9) 

谢谢你们坚持到最后。我希望你们都喜欢这篇文章。祝大家有美好的一天！