# 使用 PyScript 构建之前要知道的 3 件事

> 原文：<https://towardsdatascience.com/3-things-you-must-know-before-building-with-pyscript-245a0a82f2c3>

## 在最近遇到一些障碍、错误和怪癖之后，我想用 PyScript、Python 和 HTML 做一个构建指南

![](img/ccc42abbf59af54152ee7f8cfbcf47ea.png)

戴维·克洛德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

对于那些还没有听说过[的人来说，在](https://pyscript.net/) [PyCon 2022](https://us.pycon.org/2022/) 上首次亮相的 PyScript 是一个嵌入浏览器的 python 环境，构建在一个名为 [Pyodide](https://pyodide.org/en/stable/) 的现有项目之上。这个项目让长期的 Python 爱好者和 web 开发人员感到震惊，它在一个双向环境中无缝地融合了(*好得几乎是* ) JavaScript 和 Python，允许开发人员在浏览器中使用 Python staples，如**[**NumPy**](https://numpy.org/)**或** [**熊猫**](https://pandas.pydata.org/) **。****

**玩了几天这个项目后，我想分享一些我在掌握 PyScript 的过程中遇到的学习和 gotchya 的经验。**

****前奏**:py script
**1**中的[速成班。](#a820)[包装压痕至关重要！](#e1f7)
2。[本地文件访问](#4958)
**3** 。 [DOM 操作](#2e74)**

## **PyScript 速成班**

**要开始使用 PyScript，我们首先必须将我们的 HTML 文件与 PyScript 脚本链接起来，就像我们处理任何普通 javascript 文件一样。此外，我们可以链接 PyScript 样式表来提高可用性。**

```
**<head>**
    <link rel="stylesheet" href="https://pyscript.net/alpha/pyscript.css" />
    <script defer src="https://pyscript.net/alpha/pyscript.js"></script>
**</head>**
```

**在 HTML 文件头中导入 PyScript 后，我们现在可以利用 HTML 主体中的 **< py-script >** 标签来编写 python 代码。**

```
**<body>**
    <py-script>
        for i in ["Python", "in", "html?"]:
            print(i)
    </py-script>
**</body>**
```

**没错。开始真的就这么简单。现在，事情在哪里变得棘手？**

## **包装压痕很重要**

**使用 PyScript 的一个很大的优势是能够导入 Python 库，比如 NumPy 或 Pandas，这首先在*头*中使用 *< py-env >* 标签完成，然后在 *< py-script >* 标签内部完成，就像在普通 Python 中一样。**

```
**<head>**
    <link rel="stylesheet" href="https://pyscript.net/alpha/pyscript.css" />
    <script defer src="https://pyscript.net/alpha/pyscript.js"></script> <py-env>
- numpy
- pandas
    </py-env>
**</head>****<body>**
    <py-script>
        **import pandas as pd**
    </py-script>
**</body>**
```

**从表面上看，这似乎很简单**，但是请注意 *< py-env >* 中的包**的缩进。**

```
 <py-env>
**- numpy
- pandas**
    </py-env>
```

**原来，如果有[任何缩进](https://github.com/pyscript/pyscript/issues/136)，您将收到一个***ModuleNotFoundError****:没有名为‘pandas’*或****ModuleNotFoundError****:没有名为‘numpy’)*的模块用于 PyScript。这个错误一开始让我措手不及，因为 Python 中的缩进非常重要。***

## ***本地文件访问***

***与 Python 相比，JavaScript 处理文件访问的方式非常不同……鉴于 web 开发与隐私和安全之间的关系，这是理所应当的。因此普通 JavaScript 不能直接访问本地文件。因为 PyScript 项目是建立在 JavaScript **之上的，所以您的 Python 代码将不能像您可能习惯的那样访问本地文件**。***

***PyScript 确实在<py-env>标签中提供了文件访问的解决方案。除了导入包之外，还可以导入 CSV 或 XLSXs 等文件。</py-env>***

```
 *<py-env>
- numpy
- pandas
**- paths:
    - /views.csv**
    </py-env>*
```

***再次**注意缩进**，因为在这种情况下，CSV 必须相对于路径缩进。***

***有了包含在路径中的文件，您可以在您的<py-script>代码中读取它。</py-script>***

```
*<py-script>
    import pandas as pd
    df = pd.read_csv("**views.csv**")
</py-script>*
```

## ***DOM 操作***

***对于任何从事过 web 开发的人来说，您应该熟悉 DOM 或文档对象模型。DOM 操作在大多数 web 应用程序中很常见，因为开发人员通常希望他们的网站与用户交互，读取输入并响应按钮点击。在 PyScript 的例子中，这提出了一个有趣的问题:按钮和输入字段如何与 Python 代码交互？***

***PyScript 对此也有一个解决方案，但是，它可能不是您所期望的。下面是 PyScript 具有功能的几个例子:***

1.  ***对于按钮，可以包含*pys-onClick = " your _ function "*参数，在点击时触发 python 函数。***
2.  ***用于从 *< py-script >* 标签*document . getelementbyid(' input _ obj _ id ')中检索用户输入。值*可以检索输入值。***
3.  ***最后*py script . write(" output _ obj _ id "，data)* 可以从 *< py-script >* 标签内将输出写入标签。***

***我们可以看到这三种 DOM 操作技术被放在一个 web 应用程序中，该应用程序允许用户检查 CSV 是否已被添加到 PyScript 路径中:***

```
*<body>
   <form onsubmit = 'return false'>
   <label for="fpath">filepath</label>
   <input type="text" id="fpath" name="filepath" placeholder="Your name..">
   <input **pys-onClick="onSub"** type="submit" id="btn-form" value="submit">
    </form><div **id="outp"**></div> <py-script>
        import pandas as pd def onSub(*args, **kwargs):
            file_path = **document.getElementById('fpath').value**
            df = pd.read_csv(file_path)
            **pyscript.write("outp",df.head())**
    </py-script>
</body>*
```

***这些例子并不全面，因为该项目还支持[可视组件标签](https://github.com/pyscript/pyscript/blob/main/docs/tutorials/getting-started.md)。***

## *****结论*****

***PyScript 将一些优秀的 Python 包引入 web 开发领域，是朝着正确方向迈出的精彩一步。尽管如此，它仍然有一点成长要做，在该项目被广泛采用之前，还有许多需要改进的地方。***

> ***对在这个【https://github.com/pyscript】棒极了的项目中工作的团队表示一些支持***

*****留下您在使用 PyScript 时可能遇到的任何其他见解或 gotchya 的评论，我将制作第 2 部分。*****

***![](img/7516a80072309e14ab26b667ae9e80a7.png)***

***Jan Kahánek 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片***