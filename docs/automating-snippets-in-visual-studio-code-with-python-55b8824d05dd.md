# 用 Python 自动化 Visual Studio 代码中的代码片段

> 原文：<https://towardsdatascience.com/automating-snippets-in-visual-studio-code-with-python-55b8824d05dd>

![](img/86add70a6a436931914796d73f159447.png)

克里斯里德在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

# 用 Python 自动化 Visual Studio 代码中的代码片段

## 使用 Python 和 VSCode 加速片段工作流

每一个好的开发工作流都涉及某种代码片段的管理，在这里，您不断地存储和检索代码片段，以解决编程例程中的各种任务。

***在本文中，我将向您展示如何使用 Python 自动创建 VScode 代码片段，只需一个终端命令就可以直接从剪贴板保存它们。***

# 从代码到格式化的片段

在 VScode 中，代码片段文件的格式如下:

```
{
"Title": {"prefix": "the prefix of the snippet",
"body": ["the body of the snippet", "as a list of strings".
         "where each list element corresponds to a line of code",
"description": "a description of the snippet"}
}
```

VScode 文本编辑器应该利用这种结构，以便当您调用所需代码片段的前缀时，您会得到一段符合您当前特定需求的格式完美的代码。

这里的问题是，创建这样一个片段的过程可能有点麻烦，所以我们最终谷歌了几乎所有的东西，这有时会有点烦人。

那么，为什么不用 Python 来自动化这个过程呢？这样，当你写一些代码，你想保存它以后，你所要做的就是把它保存到剪贴板，然后在终端上运行一个命令。

# 从剪贴板保存片段的步骤

做到这一点的步骤将是:

1.  使用代码片断文件夹的路径指定一个变量。
2.  编写代码以获取剪贴板的内容。
3.  编写代码，以正确的格式将内容保存在预先指定的 snippets 文件(在调用脚本时确定)中。
4.  将最终的 Python 脚本保存在文件夹中。
5.  编写一个别名，以便在终端的任何地方调用该脚本。

现在，让我们一个接一个地完成每个步骤。

1.  **用 snippets 文件夹的路径指定一个变量**

```
global_snippets_folder = “/home/username/.config/Code/User/snippets/”
```

VScode 通常有这样一个路径的 snippets 文件夹，根据您的具体情况进行更改。

**2。编写代码以获取剪贴板的内容**

```
import clipboardclipboard.paste()clipboard_content = clipboard.paste()
```

这里，我们使用`clipboard`包来获取剪贴板的内容。

**3。编写代码，以正确的格式将内容保存在预先指定的片段文件(在调用脚本时确定)中**

这段代码将剪贴板的内容保存到指定的文件夹中，文件夹的名称由用户在调用脚本时指定，并且具有稍后在 VScode 中调用代码段所需的格式。

**4。将最终的 Python 脚本保存在一个文件夹中**

在这里你可以把它保存在你电脑里任何你想要的文件夹里。

**5。编写一个别名，从终端的任何地方调用该脚本**

要做到这一点，你所要做的就是打开你的`.bashrc`文件并写下:

```
alias save_snippet_from_clipboard="python /path/to/python_script.py"
```

然后，保存修改后的文件并运行`source .bashrc`来更新您的终端。

现在，您可以直接从剪贴板中保存代码片段了！

图片来自 https://giphy.com/

# 做同样事情的许多方法

值得注意的是，这只是一种方法，有很多方法可以保存和检索代码片段，最后，你应该尽可能多地尝试各种工具，看看哪些对你有用。

由于从剪贴板保存代码的问题，我在这里描述的方法在保证保存的代码片段的格式不变方面有一些缺点，但最终，我觉得每次我需要保存或检索某个代码片段时节省的几秒钟是值得的。

如果你愿意，你可以在这里观看我的 Youtube 视频:

如果你喜欢这个帖子，[加入媒体](https://lucas-soares.medium.com/membership)，[关注](https://lucas-soares.medium.com/)，[订阅我的简讯](https://lucas-soares.medium.com/subscribe)。还有，订阅我的 [youtube 频道](https://www.youtube.com/channel/UCu8WF59Scx9f3H1N_FgZUwQ)在 [Tiktok](https://www.tiktok.com/@enkrateialucca?lang=en) 、[推特](https://twitter.com/LucasEnkrateia)、 [LinkedIn](https://www.linkedin.com/in/lucas-soares-969044167/) 、 [Instagram](https://www.instagram.com/theaugmentedself/) 上和我联系！谢谢，下次再见！:)