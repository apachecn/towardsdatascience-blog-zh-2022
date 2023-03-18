# 用 Python 快速构建和部署 Web 应用程序

> 原文：<https://towardsdatascience.com/rapidly-building-and-deploying-a-web-app-in-python-10e0021596a5>

![](img/dd63c4d79b9cbc2692270f49e76de980.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的 [ThisisEngineering RAEng](https://unsplash.com/@thisisengineering?utm_source=medium&utm_medium=referral)

我最近用 Python 写了几个 [web 开发框架，今天我将介绍使用 Streamlit 构建和部署一个非常简单的新冠肺炎仪表板。Streamlit 旨在通过围绕您的 Python 代码构建一个流畅的 UI 来轻松创建以数据为中心的应用程序，不需要任何 web 开发经验(尽管它确实有所帮助)。](https://python.plainenglish.io/a-very-brief-overview-ofweb-development-with-python-8898dd00801c)

在开始之前，我想解释一下为什么我要写这篇文章。我和一位技术高超的数据科学家同事通了电话。他在寻找利用他的 Python 技能来构建可共享的应用程序的方法。我向他介绍了 Streamlit，他很快就开发出了一个很棒的应用。然而，当要分享他的应用程序时，他完全不知道该怎么做。作为一名数据分析师，我也经历过他的情况，如果不能与他人分享自己辛苦编写的代码，会非常沮丧。我希望通过这篇文章来弥补这一差距，解释让你在适当的时候轻松分享应用程序所需的所有步骤。

我们将首先为我们的项目建立一个虚拟环境。然后，我们将编写一些 Python 代码，作为应用程序的引擎。第三，我们将使用 Streamlit 库来帮助为我们的 Python 代码创建 UI，最后我们将部署我们的应用程序。我希望你能更好地理解 Python 中 web 开发的端到端过程。

## 1.设置虚拟环境

当您开始一个项目时，设置虚拟环境是您应该采取的第一步。你可以把虚拟环境想象成你的应用程序所在的地方，它与外界完全隔离。它不知道您可能已经编写的任何其他程序或您可能已经安装的 Python 库。这一点很重要，因为已经安装在计算机上的 Python 库可能会给应用程序带来问题。例如，一个新版本的包可能会通过取消某些功能来破坏您的代码。虚拟环境通过允许您控制库的版本并将它们与已经安装在您机器上的库分开来缓解这个问题，有助于防止任何依赖关系被覆盖或彼此冲突。此外，它有助于保持整洁有序。您的应用程序可能不会使用您机器上安装的所有 Python 库。如果你要在虚拟环境之外构建一个应用程序，那么每个库都将被包含在所谓的需求文件中(稍后会详细介绍)，这是部署你的应用程序的必要元素。这只会导致额外的膨胀。

要实际设置虚拟环境，请打开终端或命令提示符。如果你在 Mac 上，通过键入 cd desktop 导航到你的桌面位置——在 windows 上是 CD C:\ Users \ YOUR PC USERNAME \ Desktop。接下来键入 mkdir streamlit _ project & & CD streamlit _ project。在您的项目目录中，通过键入 python3 -m venv 创建一个虚拟环境，后跟您决定的虚拟环境名称。惯例通常称之为虚拟环境。但是我要把我的名字叫做 virt_env。他就是那个样子:

```
python3 -m venv virt_env # Mac
python -m venv virt_env # Windows
```

接下来，我们将激活虚拟环境，并开始安装项目所需的软件包。要激活虚拟环境，请根据您的操作系统键入以下内容之一:

```
source virt_env/bin/activate # Mac
virt_env\Scripts\activate # WindowsOnce activated you should see
(virt_env) ...
```

## 创建我们的应用程序

现在让我们安装 streamlit 和 Python 应用程序中需要的其他包:

```
python3 -m pip install streamlit # Mac
python -m pip install streamlit # Windows python3 -m pip install pandas # Mac
python -m pip install pandas # Windowspython3 -m pip install matplotlib # Mac
python -m pip install matplotlib # Windows
```

在我们安装完这些包之后，请从我的 [Github repo](https://github.com/cbecks1212/streamlit_covid/blob/main/covid_dashboard.py) 中复制并粘贴代码，并将其作为 covid_dashboard.py 保存在您的虚拟环境的目录中。这是我们的 python 文件，包含我们的 streamlit 应用程序，它主要做两件事:1)计算四个指标，2)按状态绘制新案例的时间序列。我将更详细地解释每一部分。

```
@st.cache(allow_output_mutation=True)def load_dataset(): try: df = pd.read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv", sep=",") states = list(df['state'].unique()) except: Exception return df, states
```

load_dataset()函数从 NYT Github 帐户中提取所有的 Covid 数据，并创建一个美国各州的列表进行过滤。在这个函数上面使用 st.cache 很重要，因为它允许将整个数据集存储在内存中，这样程序就不必在每次应用不同的过滤器时重新加载整个数据集。

```
def pct_change(data, state, today):
   ....
```

pct_change 函数有三个参数:covid 数据集、我们在下拉列表中选择的州(我们稍后会讲到)和今天的日期。该函数返回一个包含四个指标的字典(新病例、14 天病例变化、新死亡和 14 天死亡变化)。

```
def chart_data(data, state):
   ...
```

def chart_data 从下拉列表中获取我们的 covid 数据集和所选州，并返回我们所选州的新病例的时间序列数据帧。

移动到代码的最后一部分(这是我们的函数被调用和数据被返回并显示在我们的应用程序中的地方)。调用 load_dataset()并生成数据框和美国各州列表。然后，我们用 st.selectbox 实例化一个下拉列表，并向其中添加一个标签和我们的状态列表。然后，当下拉菜单被选中时，我们添加逻辑。最后，streamlit 提供了一种方便的方式来添加列和图形到您的网页，创建一个更干净的整体演示。所有这一切的美妙之处在于能够用最少的代码用纯 Python 创建流畅且响应迅速的 HTML 组件。

```
df, states = load_dataset() state_dropdown = st.selectbox('Select a State', states)if state_dropdown:data_dict = pct_change(df, state_dropdown, today)col1, col2, col3, col4 = st.columns(4)col1.markdown(f"New Cases: {data_dict['Cases']}")col2.markdown(f"""<p>14-Day Change: <span style="color:red">{data_dict['Case Change']}%</span></p>""", unsafe_allow_html=True)col3.markdown(f"New Deaths: {data_dict['Deaths']}")col4.markdown(f"""<p>14-Day Change: <span style="color:red">{data_dict['Death Change']}%</span></p>""", unsafe_allow_html=True)chart_data = chart_data(df, state_dropdown)st.line_chart(chart_data)
```

## 部署应用程序:初始步骤

我们已经创建了一个虚拟环境和我们的 streamlit 应用程序，但它只存在于我们的本地机器上。为了与全世界分享它，我们需要确保安装了 Git 和 Heroku。要安装 Git，打开一个新的命令行窗口，在 Mac 上键入

```
git --version
```

应该会出现一个弹出窗口，显示进一步的说明，允许您继续安装。对于 Windows 用户，使用以下链接下载 Git:[https://git-scm.com/downloads](https://git-scm.com/downloads)

要在 Mac 上安装 Heroku，首先安装[家酿](https://brew.sh/)，如果你还没有的话，然后在新的终端窗口中键入:

```
brew tap heroku/brew && brew install heroku
```

如果你有一个 M1 芯片，你可能会得到一个错误，但这将有助于解决问题:【https://support.apple.com/en-us/HT4

对于 Windows 用户，使用以下链接下载 Heroku CLI:[https://dev center . Heroku . com/articles/Heroku-CLI #下载并安装](https://devcenter.heroku.com/articles/heroku-cli#download-and-install)

一旦安装了这些依赖项，就可以将代码推送到 Github 了。首先创建一个. gitignore 文件，将虚拟环境文件从 Github repo 中排除，然后通过键入 virt_env/(在做出更改后保存文件)来修改它:

```
(virt_env) > touch .gitignore # Mac
(virt_env) > echo.>.gitignore # Windows
```

现在，在命令行窗口中初始化 git，通过 Git 激活您的虚拟环境

```
git init
git add -A
git commit -m "initial commit"
```

登录您的 Github 帐户，创建一个新的存储库，然后选择“从命令行推送现有的存储库”选项将该选项中的代码片段复制并粘贴到运行虚拟环境的命令行中。

为了结束这个阶段，通过 pip freeze 创建一个需求文件。这告诉 web 服务器成功运行应用程序需要哪些依赖项

```
python -m pip freeze > requirements.txt
```

在名为 setup.sh 的虚拟环境目录中创建新文件，并将以下内容保存到其中:

```
mkdir -p ~/.streamlit/echo "\[server]\n\headless = true\n\port = $PORT\n\enableCORS = false\n\\n\" > ~/.streamlit/config.toml
```

然后在虚拟环境目录中创建一个 Procfile(就叫它 Procfile)并将以下内容保存到其中:

```
web: sh setup.sh && streamlit run covid_dashboard.py
```

现在是时候将 requirements.txt、Procfile 和 setup.sh 文件添加到 Github repo:

```
git add -A
git commit -m "files for Heroku deployment"
git push -u origin main
```

## 将应用程序部署到 Heroku:

要将应用程序部署到生产环境，请在运行虚拟环境的命令行中键入以下内容:

```
(virt_env) > heroku login
(virt_env) > heroku create
```

这将创建一个随机的 url 名称并部署您的应用程序。这将需要一两分钟的时间来完成，但是您应该会在命令行中看到一个类似于此处的 url。你可以把它复制到你的网络浏览器里。

## 总结:

读完这篇文章后，我希望代码困在您的计算机上的日子已经成为过去，您的组织中的人们开始从您的创新中受益。