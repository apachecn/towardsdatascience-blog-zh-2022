# PyScript v. Flask:如何在浏览器或服务器上创建 Python 应用程序

> 原文：<https://towardsdatascience.com/pyscript-v-flask-how-to-create-a-python-app-in-the-browser-or-on-a-server-2cfe4dd0df9d>

## PyScript 允许您在不需要服务器的情况下用 Python 创建 web 应用程序。Flask 是一个 Python web 应用程序框架，用于制作基于服务器的应用程序。我们使用两者编写相同的简单应用程序。

![](img/14841a8ba5d8f8a1ac24f4d64e0036be.png)

多梅尼科·洛亚在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

PyScript 是浏览器中的 Python，它承诺了一种编写 web 应用程序的新方法。目前它还处于 *alpha* 阶段，但已经是一个可用的系统了。

但是我们需要新的方法吗？的确，我可以用 PyScript 创建一个全功能的 web 应用程序(借助一些 HTML，可能还有一点 Javascript ),但是我已经可以用 Flask 做到了！

使用 Flask，Python 代码在服务器上运行，并根据需要更新网页。使用 PyScript，Python 代码在浏览器中运行，并直接与网页交互。

那么，使用 PyScript 比使用基于服务器的应用程序更好还是更容易呢？我们将比较两个类似的应用程序，一个用 PyScript 编写，另一个使用基于 Python 的应用程序框架 Flask。

在浏览器中运行代码至少有一个好处，那就是部署。一个基于浏览器的应用程序只需要被复制到一个网络服务器上，它就可以工作了。然而，基于服务器的应用程序可能需要更多的努力才能部署到 Heroku、Azure 或 AWS 等平台上。但是一旦你习惯了部署基于服务器的应用，这并不需要太多的努力。

我在这里使用的 PyScript 应用程序在以前的文章中已经介绍过了，所以您可以看看这些文章来更深入地了解 PyScript 是如何工作的(参见下面的注释)。

我们将首先查看 PyScript 应用程序，然后看看我们如何使用 Flask 构建类似的东西。

该应用程序本身相当简单，但它包含了仪表板类型应用程序的所有基础:它允许用户交互，加载远程数据，显示交互式图表，并使用 Bootstrap UI 框架使一切看起来很好。

具体来说，该应用程序显示了代表 2020 年英国伦敦天气状况的四个图表之一。这些图表是最高和最低月气温、降雨量和日照时间，它们使用了我在 Github 上的英国历史天气报告中的数据。

用户可以从下拉菜单中选择一个图表，新的图表将会显示出来。这两个版本的应用程序都不会刷新页面:在 PyScript 应用程序中，对 Python 代码的调用会读取数据并直接更新图表，而在 Flask 应用程序中，会对服务器进行异步回调，服务器会使用用于更新图表的数据进行响应。

让我们看看代码。

# PyScript 应用程序

PyScript 应用程序是一个 HTML 网页，其结构如下:

标签包含了你能在网页上找到的所有常见的东西，还包含了对 PyScript CSS 和 Javascript 文件的引用。`<body>`标签包含页面的 HTML 和任何所需的 Javascript，而`<py-script>`包含——谁会想到呢——py script 代码。

PyScript 部分可以引用一个外部文件，但是由于 CORS 限制，在本地运行该文件时它将不起作用，它必须在一个服务器上运行(当然，那个服务器也可以在您的本地机器上运行)。

这里是对头部的近距离观察:

我们首先引用运行 PyScript 所需的 PyScript 文件，然后引用将在 HTML 中使用的 Boostrap 和 Plotly CDNs。

`<py-env>`简单地列出了将在`<py-script>`部分使用的 Python 库。

现在让我们看看正文的第一部分，HTML 和 Javascript:

不需要太多的细节，我们从一个自举超大屏幕元素开始，它充当一个标题。接下来是一个下拉菜单，允许选择要显示的图表。

然后我们有一个`<div>`作为图表的容器，最后是一个简短的 Javascript 函数，它获取图表数据和容器的 id，并使用 Plotly 库绘制图表。这个函数将直接从 Python 代码中访问。

这就是整理出来的用户界面。剩下的工作是加载远程数据，过滤它以获得我们想要的特定数据，并创建图表数据。所有这些都是用 Python 实现的。

这里是包含 Python 代码的 PyScript 部分。

首先，与任何 Python 程序一样，我们导入所有需要的库。

为了获得数据，我们需要使用 Pyodide 包中的函数`open_url`(PyScript 基于 Pyodide，其库集成到了 py script 中)。然后我们可以从这些数据中创建一个熊猫数据框架。

接下来，我们过滤数据。数据框架目前包含几十年的数据，但我们只打算使用 2020 年的数据。这就是下面的代码将为我们做的。

```
df = df[df['Year']==2020]
```

代码的剩余部分主要是函数定义。

函数`plot()`创建 Plotly 图表，并使用 Javascript 函数在其容器中绘制图表。它接受一个用于选择要显示的图表的参数，用 Plotly Python 包创建图表数据，最后调用我们前面看到的 Javascript 函数在其容器中显示图表。

请注意调用 Javascript 函数是多么容易。我们只需导入`js`库，所有的 Javascript 函数都可供我们使用。

接下来的两个函数依赖于它们前面的`imports`。`selectChange()`是当下拉菜单中的值改变时将被调用的函数。它从`<select>`标签中读取新值，并调用我们刚刚看到的 Python `plot()`函数来显示所选图表。

注意，使用`js` 包中的`document`可以让我们以与内置 Javascript 函数完全相同的方式查询 DOM。

接下来，我们需要将用户更改选定值所生成的事件与该函数联系起来。这就是`setup()`的工作。首先，这个函数为 Python 函数`selectChange()`创建一个代理，Javascript 可以使用这个代理直接调用这个函数。代理叫做`change_proxy`。

接下来，我们设置一个事件监听器来检测下拉菜单中的变化，并指示它通过我们刚刚设置的代理调用`selectChange`函数。

最后两行简单地运行`setup()`来设置代理和事件监听器，并调用`plot("Tmax")`来默认显示一个图表。

对于那些习惯于传统网页开发或 Python 编程的人来说，这个应用程序可能感觉不太直观。但是 PyScript 应用程序的设计非常优雅:用户界面由 HTML 和 CSS 定义(在 Javascript 的帮助下),而应用程序的逻辑几乎完全在 PyScript 部分定义。

在我看来，这是为每项工作使用正确的工具。HTML 和 CSS 都是关于如何显示内容的，而 Python 是用于编程的，这两者应该并且确实是分开的。

# 烧瓶应用程序

这种分离也存在于基于服务器的应用程序中。但是在这种情况下，Python 逻辑在服务器上运行。

一个基本的 Flask 应用程序由两个组件组成，基于服务器的 Python 代码和 HTML 模板。当应用程序被调用时，它将 HTML 模板作为网页返回。它还可能用 Python 代码中的值填充模板中的字段，尽管我们在这里没有使用这种功能。

如果你不熟悉 Flask 应用，我在文章“[如何创建和运行 Flask 应用](https://alan-jones.medium.com/how-to-create-and-run-a-flask-app-533b7b101c86)”中写了一个简单的介绍，在 [Flask 网站](https://flask.palletsprojects.com/en/2.1.x/quickstart/)上有更全面的介绍。

简而言之，Flask 应用程序的 Python 部分定义了端点和 Python 函数，这些函数决定了当这些端点被寻址时会发生什么。这个应用程序定义了两个端点，一个对应于根，另一个调用回调函数。

根端点简单地返回 HTML 模板，而回调端点加载所需的数据并返回 Plotly 图表数据以显示在 web 页面上。

让我们先来看看 HTML。

该文件在功能上与 PyScript 应用程序的 HTML 部分相同。事实上，除了缺少 PyScript 部分之外，大部分代码都是相同的。

此外，事件监听器和它调用的函数现在是用 Javascript 编写的，当然，不需要代理。

最大的变化是访问 Python 代码的方式。这是通过调用服务器上的回调函数来完成的。

我将在这里重复这段代码:

这个函数是异步的，这意味着在它被调用后，应用程序的执行不会停止，而是允许该函数在后台继续执行，直到它完成。这是调用服务器函数的理想方法，因为我们不能确定服务器需要多长时间来响应，也不想在等待响应时冻结应用程序。

服务器上的回调端点称为`callback`，它期望看到一个称为`data`的参数，该参数将保存下拉菜单中的选择值。

服务器将以 JSON 格式的图表数据进行响应，这将用于绘制图表，就像以前一样，使用 Plotly Javascript 库。

这将 Python 代码留在了服务器上。

我们从导入开始，然后我们可以看到两个端点的定义。代码的格式如下:

```
@app.route('/')
def index():
    *# some code that returns a web page or other data*
```

第一个端点是根，使用 Flask 函数`render_template()`返回我们上面看到的 HTML 模板。

第二个端点是回调函数。这不返回网页，只返回网页自我更新所需的数据。它调用函数`getGraph()`，该函数的工作与 PyScript 版本相同，它加载并过滤数据，然后创建图表数据，该数据返回给异步 Javascript 函数。

我希望你能看到，Flask 应用程序和 PyScript 版本做的完全一样。

那么你为什么要选择一种方法而不是另一种呢？

# 结论

两个应用程序都可以工作，而且看起来一样。那么，如何选择使用哪种方法呢？

首先，我们应该意识到这是一个非常简单的应用程序。更复杂的应用程序可能会下载更多的数据，可能需要更多的处理，或者可能需要更复杂的用户界面。

但是这些例子说明了这种类型的应用程序所需的基本操作。

我在一台典型的家用笔记本电脑上运行了这两个应用程序，它们都运行得很好。PyScript 应用程序的加载时间要长得多，但响应时间可能会稍微快一些。坦率地说，在您等待 PyScript 版本加载(需要几秒钟)之后，它们之间并没有太大的区别。

这里有几个变量:

*   互联网连接的速度将决定向服务器发出请求的速度以及数据返回的速度
*   服务器的力量。大多数服务器比任何台式机或笔记本电脑都更强大
*   用户计算机的能力。当然，一个较弱的硬件会比一个更强的硬件慢，但这将对 PyScript 应用程序产生更大的影响，因为处理是在本地完成的。

# 获胜者是…

这两种技术都是有效和有用的。PyScript 可能还有很长的发展路要走，毫无疑问还会改进，但是，即使是现在，它也是轻量级应用的一个好的解决方案。

如果有大量的处理工作要做，那么目前基于服务器的解决方案可能仍然是最好的方法，但是随着硬件变得更强大(它总是这样)和 PyScript 的改进，这种情况可能会改变。

PySript 是工具箱中另一个看起来很有前途的工具。我怀疑它会完全取代基于服务器的应用程序，但对于后端处理在典型主机能力范围内的应用程序，PyScript 可能会找到自己的位置。

一如既往，感谢阅读。要下载代码或找到更多关于 PyScript、Python、Flask 和其他数据科学相关主题的文章，请参见我的 [Github 网页](http://alanjones2.github.io)。

<https://alanjones2.github.io>  

你也可以订阅我的时事通讯 [Technofile](http://technofile.substack.com) 来了解我正在做的事情。

<https://technofile.substack.com>  

# 笔记

1.  文章[用 PyScript 和 Pandas 创建一个交互式 web 应用](/create-an-interactive-web-app-with-pyscript-and-pandas-3918ad2dada1)包含了使用 Pandas 绘图的原始 Web 应用。[如何结合 PyScript 使用 Plotly](https://medium.com/technofile/how-to-use-ploty-with-pyscript-578d3b287293)展示了如何使用 Plotly 创建一个类似的应用程序。
2.  天气数据来自我的报告[英国历史天气](https://github.com/alanjones2/uk-historical-weather)，并来自英国气象局[历史气象站数据](https://www.metoffice.gov.uk/research/climate/maps-and-data/historic-station-data)。它是根据[英国开放政府许可证](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/)发布的，可以在相同的条件下使用。