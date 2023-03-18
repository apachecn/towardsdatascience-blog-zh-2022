# 开始数据科学和数据分析职业生涯的 5 个基本工具

> 原文：<https://towardsdatascience.com/5-essential-tools-to-start-a-career-in-data-science-and-data-analytics-dfcf25b5e51b>

## 学习这 5 个工具，获得作为数据科学家或数据分析师的第一份工作

![](img/49f28fba5a3279f9478c18f928cf6722.png)

在 [Unsplash](https://unsplash.com/photos/hpjSkU2UYSU) 上由 [Carlos Muza](https://unsplash.com/@kmuza) 拍摄的照片

# 动机

数据科学家的工作是利用大型结构化或非结构化数据集，以便提取有意义的信息，从而做出更好的决策。它结合了领域专业知识、数学和统计知识、数据建模和结果交流技能。然而，他们也需要工具来赋予这些概念以生命。

本文将在强调这些工具的好处之前，帮助您理解它们。

# 什么是数据科学和数据分析工具？

市场上有太多的工具，无论是开源还是付费许可，相关工具的升级可能会帮助您优化您的投资组合，并为您的下一个数据职业生涯做好准备。

本文范围内的工具是业内最常用的，分为三大类，如*数据分析可视化*、*脚本/机器学习*和*数据库管理*。

## 数据分析和可视化工具

数据可视化是数据的图形化表示。它与数据科学项目的任何其他方面一样重要。根据 [ILS 测试统计数据](https://www.atlassian.com/blog/teamwork/how-to-work-4-different-learning-types)显示，超过 65%的人是视觉学习者，因此清晰简洁的可视化有助于传达关于数据的关键信息，从而更好、更快地做出决策。

**1 →表格**

[Tableau](https://www.tableau.com/products/desktop) 是 2019 年 Salesforce 收购的无代码商业智能软件。它为分析和可视化提供了直观的拖放界面。非技术方面使其在行业中脱颖而出。

此外，它速度很快，并提供了互连来自多个来源(如电子表格、SQL 数据库等)的数据的能力。无论是从云还是内部创建单一可视化。Tableau 是地理空间和复杂数据可视化的通用工具。此外，它还兼容流行的编程语言，如 Python 和 r。

**2 →微软 PowerBI**

与 Tableau 类似， [PowerBI](https://powerbi.microsoft.com/fr-fr/) 也是一个商业智能和数据可视化工具，允许将来自多个来源的数据转换成交互式商业智能报告，并且还支持 Python 和 r。

> 但是，它们真正的区别是什么呢？

它与 Tableau 的主要区别在于它不能处理和 Tableau 一样多的数据。除此之外，它可以连接到有限数量的数据源。例如，Power BI 不能与 MongoDB 这样的 NoSQL 数据库一起正常工作。但是，它价格实惠，不仅适用于大中型公司，也适用于小型公司。

## 机器学习和脚本工具

每个数据科学家无一例外都需要具备编程技能，要么为数据处理和分析创建脚本，要么构建机器学习模型。Python 和 R 是所有数据科学家最流行的编程语言之一。

**3 → Python**

Python 提供的简单性和灵活性迅速增加了数据科学家对它的采用。例如，以下代码为 Python 和 Java 生成相同的结果`Data Science and Analytics Tools`。

*   对于 Python，我们可以在命令行解释器中键入`python`，后跟`print`语句，如下所示。

```
# Step 1: open interpreter
python# Step 2: write the following expression to show the message
>>> print("Data Science and Analytics Tools")
```

*   但是，对于 java，我们需要创建一个完整的程序，并对其进行编译，以获得相同的结果。这是因为它没有命令行解释器。

```
# Step 1: write this code in a file ShowMessage.javaclass ShowMessage {
    public static void main(String[] args) {
        System.out.println("Data Science and Analytics Tools"); 
    }
}# Step 2: compile the file
javac ShowMessage.java# Step 3: execute the program to show the message
java ShowMessage
```

除了是开源的，并且有一个大型社区，Python 提供了以下框架和库(不是详尽的),它们是数据分析和机器学习的顶级框架和库。数据科学家可以:

*   使用`[Numpy](https://numpy.org/)`执行高级数值计算，使用多维数组提供紧凑和快速的计算。
*   利用`[Pandas](https://pandas.pydata.org/)`进行数据处理、清理和分析。它被广泛使用，是数据科学家使用的最流行的工具。
*   使用`[Matplotlib](https://matplotlib.org/)`和`[Seaborn](https://seaborn.pydata.org/)`创建从简单到更高级的数据可视化，可以进一步集成到应用程序中以生成仪表板。
*   用`[Scikit-learn](https://scikit-learn.org/stable/)`、`[Pytorch](https://pytorch.org/)`、`[Keras](https://keras.io/)`实现几乎所有的机器学习和深度学习算法。
*   使用`[Beautiful](https://beautiful-soup-4.readthedocs.io/en/latest/)`从互联网上抓取数据，将其转换成合适的格式并存储，以创建数据存储。

**4 → R(工作室)**

这种编程语言是由统计学家创造的，这使得它在统计分析和数据可视化方面非常流行。它被数据科学家和业务分析师以及学术界广泛用于研究。

**R** 整合了`[tidyverse](https://www.tidyverse.org/),`一套用于数据科学任务的强大工具(并非详尽无遗)，例如:

*   使用`[ggplot2](https://ggplot2.tidyverse.org/).`创建强大的数据可视化
*   使用`[modelr](https://modelr.tidyverse.org/).`实现优雅的数据建模管道
*   使用`[dplyr](https://modelr.tidyverse.org/)`执行数据操作，这是一个包含多个便捷函数的库，可以解决最常见的任务，如数据过滤、选择、聚合等。
*   使用`[readr](https://cran.r-project.org/web/packages/readr/readme/README.html)`加载 CSV 和 TSV 数据文件，使用`[readxl](https://readxl.tidyverse.org/)`加载 Microsoft Excel 数据。

r 不仅提供了统计和可视化功能，还通过`[caret](https://topepo.github.io/caret/)`提供了机器学习能力，这是一个包含数百种算法的包。

## 数据库管理

作为数据科学家，您必须能够从本地或远程数据库中检索结构化或非结构化数据。

**5 → SQL**

结构化查询语言(SQL)是一种功能强大的语言，大、中、小型数据驱动型企业都使用这种语言来探索和操作他们的数据，以便提取相关的见解。这是因为大多数公司使用关系数据库系统，如 PostgreSQL、MySQL、SQLite 等，我们可以从 Stackoverflow 提供的 2022 年调查结果之后的[中观察到这一点。](https://survey.stackoverflow.co/2022/#most-popular-technologies-database-prof)

这一结果无疑使得 SQL 知识需求量很大。它甚至是数据科学家/机器学习专家、数据分析师、业务分析师和专业开发人员中最受欢迎的语言之一。

进一步挖掘一下调查，[这张图](https://survey.stackoverflow.co/2022/#most-popular-technologies-language-prof)显示了 SQL 的使用有多广泛，相比之下 Python 和 R 分别为 54.64%、43.51%和 3.56%。

考虑到专业开发人员使用关系数据库的百分比，这一发现显然并不令人惊讶。此外，从这些分析中得出的一个重要结论是，企业不会很快放弃 SQL。

好消息是，SQL 的人类可读方面使其成为最简单的学习语言之一，我在 DataCamp 上偶然看到了[这门课程，我相信它可能会帮助你获得构建 SQL 组合的相关技能。](https://www.datacamp.com/learn/sql)

# 结论

作为数据科学家或数据分析师，你的第一份工作可能相当令人生畏。然而，学习符合就业市场要求的技能肯定可以帮助你建立一个强大的投资组合来面对这些挑战。现在是探索的时候了，获得你一直期待的第一份工作吧！

如果你喜欢阅读我的故事，并希望支持我的写作，考虑[成为一个媒体成员](https://zoumanakeita.medium.com/membership)解锁无限制访问媒体上的故事。

欢迎在[媒体](https://zoumanakeita.medium.com/)、[推特](https://twitter.com/zoumana_keita_)和 [YouTube](https://www.youtube.com/channel/UC9xKdy8cz6ZuJU5FTNtM_pQ) 上关注我，或者在 [LinkedIn](https://www.linkedin.com/in/zoumana-keita/) 上打招呼。讨论人工智能、人工智能、数据科学、自然语言处理和人工智能是一种乐趣！