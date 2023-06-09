# 熊猫还不够。学习这 25 个熊猫到 SQL 的翻译来升级你的数据分析游戏

> 原文：<https://towardsdatascience.com/pandas-isnt-enough-learn-these-25-pandas-to-sql-translations-to-upgrade-your-data-analysis-game-af8d0c26948d>

## 熊猫中常见的 25 种 SQL 查询及其对应方法。

![](img/694a7fa94cf1f1068a4e45c6b2b7a67f.png)

照片由[詹姆斯·亚雷马](https://unsplash.com/@jamesyarema?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

> 这是我第 50 篇关于媒介的文章。非常感谢你阅读和欣赏我的作品*😊*！这绝对是一次收获颇丰的旅程。
> 
> 如果你喜欢在 Medium 上阅读我的文章，我相信你也会喜欢这个:<https://avichawla.substack.com/>**【数据科学的每日剂量】。**
> 
> ****这是什么？**这是一份面向数据科学的出版物，我在 substack 上运营。**
> 
> ****你会从中得到什么？**在这里，我介绍了关于数据科学/Python/机器学习等的优雅而方便的技巧和诀窍。，一天一个提示(见出版档案[此处](https://avichawla.substack.com/archive))。如果你感兴趣，你可以在你的收件箱里订阅每日剂量。而且完全免费。很想看到另一面！**

# **动机**

**SQL 和 Pandas 都是数据科学家处理数据的强大工具。**

**众所周知，SQL 是一种用于管理和操作数据库中数据的语言。另一方面，Pandas 是 Python 中的一个数据操作和分析库。**

**此外，SQL 通常用于从数据库中提取数据，并为 Python 中的分析做准备，主要使用 Pandas，它提供了广泛的工具和函数来处理表格数据，包括数据操作、分析和可视化。**

**SQL 和 Pandas 一起可用于清理、转换和分析大型数据集，并创建复杂的数据管道和模型。因此，精通这两种框架对于数据科学家来说非常有价值。**

**因此，在这篇博客中，我将提供一个快速指南，将最常见的 Pandas 操作翻译成它们的等效 SQL 查询。**

**我们开始吧🚀！**

# **资料组**

**出于演示目的，我使用 Faker 创建了一个虚拟数据集:**

**![](img/394d838d0756cebe048ec1b5706fe295.png)**

**随机员工数据集(图片由作者提供)**

# **#1 读取 CSV 文件**

## **熊猫**

**CSV 通常是读取熊猫数据帧最流行的文件格式。这是在熊猫身上用`pd.read_csv()`方法完成的。**

## **结构化查询语言**

**要在数据库中创建表，第一步是创建一个空表并定义其模式。**

**下一步是将 CSV 文件的内容(如果第一行是标题，从第二行开始)转储到上面创建的表中。**

## **输出**

**创建数据帧/表后，我们得到以下输出:**

**![](img/394d838d0756cebe048ec1b5706fe295.png)**

**读取 CSV 后的输出(图片由作者提供)**

# **#2 显示前 5(或 k)行**

## **熊猫**

**我们可以在熊猫身上使用`df.head()`方法。**

## **结构化查询语言**

**在 MySQL 语法中，我们可以在`select` 后使用`limit`，指定我们想要显示的记录数。**

# **#3 打印尺寸**

## **熊猫**

**DataFrame 对象的`shape`属性打印行数和列数。**

## **结构化查询语言**

**我们可以使用 count 关键字来打印行数。**

# **#4 打印数据类型**

## **熊猫**

**您可以使用`dtypes`参数打印所有列的数据类型:**

## **结构化查询语言**

**在这里，您可以打印如下数据类型:**

# **#5 修改列的数据类型**

## **熊猫**

**这里，我们可以使用如下的`astype()`方法:**

## **结构化查询语言**

**使用`ALTER COLUMN`改变列的数据类型。**

**上述操作将永久修改表中列的数据类型。但是，如果您只是想在过滤时这样做，请使用`cast`。**

# **# 6–11 过滤数据**

**有各种方法来过滤熊猫的数据帧。**

****#6:** 您可以按如下方式过滤一列:**

**上述内容可以转换为 SQL，如下所示:**

****#7:** 此外，您还可以过滤多个列:**

**上述过滤的 SQL 等价物是:**

****#8:** 您还可以使用`isin()`从值列表中进行筛选:**

**为了模拟上述情况，我们在 SQL 中使用了`in`关键字:**

****#9:** 在 Pandas 中，您还可以使用点运算符选择特定的列。**

**在 SQL 中，我们可以在`select`之后指定所需的列。**

****#10:** 如果您想在 Pandas 中选择多个列，您可以执行以下操作:**

**通过在 SQL 中的`select`之后指定多个列也可以做到这一点。**

**你也可以根据熊猫的 NaN 值进行过滤。**

**我们没有 NaN 值，所以我们看不到行。**

**这同样可以扩展到 SQL，如下所示:**

**我们还可以执行一些复杂的基于模式的字符串过滤。**

**在 SQL 中，我们可以使用`LIKE`子句。**

**您也可以在字符串中搜索子字符串。例如，假设我们想要查找所有`last_name`包含子字符串“an”的记录。**

**在熊猫身上，我们可以做到以下几点:**

**在 SQL 中，我们可以再次使用`LIKE`子句。**

# **# 14–16 排序数据**

**排序是数据科学家用来对数据进行排序的另一种典型操作。**

## **熊猫**

**使用`df.sort_values()`方法对数据帧进行排序。**

**您也可以按多列排序:**

**最后，我们可以使用`ascending`参数为不同的列指定不同的标准(升序/降序)。**

**这里，`ascending`对应的列表表示`last_name`降序排列，`level`升序排列。**

## **结构化查询语言**

**在 SQL 中，我们可以使用`order by`子句来实现。**

**此外，通过在`order by`子句中指定更多的列，我们可以包含更多的列作为排序标准:**

**我们可以为不同的列指定不同的排序顺序，如下所示:**

# **#17 填充 NaN 值**

> **对于这一个，我有意删除了 salary 列中的几个值。这是更新后的数据框架:**

## **熊猫**

**在 Pandas 中，我们可以使用`fillna()`方法来填充 NaN 值:**

## **结构化查询语言**

**然而，在 SQL 中，我们可以使用 case 语句来实现。**

# **# 18–19 连接数据**

## **熊猫**

**如果你想用一个连接键合并两个数据帧，使用`pd.merge()`方法:**

## **结构化查询语言**

**连接数据集的另一种方法是将它们连接起来。**

## **熊猫**

**考虑下面的数据框架:**

**在 Pandas 中，您可以使用`concat()`方法并传递 DataFrame 对象以连接成一个列表/元组。**

## **结构化查询语言**

**使用 SQL 中的`UNION`(只保留唯一的行)和`UNION ALL`(保留所有行)也可以达到同样的效果。**

# **#20 分组数据**

## **熊猫**

**要对数据帧进行分组并执行聚合，请在 Pandas 中使用`groupby()`方法，如下所示**

## **结构化查询语言**

**在 SQL 中，可以使用 group by 子句并在 select 子句中指定聚合。**

**我们确实看到了相同的输出！**

# **# 21–22 寻找独特的价值**

## **熊猫**

**要打印一列中的不同值，我们可以使用`unique()`方法。**

**要打印不同值的数量，使用`nunique()`方法。**

## **结构化查询语言**

**在 SQL 中，我们可以如下使用`select`中的`DISTINCT`关键字:**

**为了计算 SQL 中不同值的数量，我们可以将`COUNT`聚合器包装在`distinct`周围。**

# **#23 重命名列**

## **熊猫**

**这里，使用`df.rename()`方法，如下所示:**

## **结构化查询语言**

**我们可以使用`ALTER TABLE`来重命名一个列:**

# **#24 删除列**

## **熊猫**

**使用`df.drop()`方法:**

## **结构化查询语言**

**类似于重命名，我们可以使用`ALTER TABLE`，将`RENAME`改为`DROP`。**

# **#25 创建新列**

**假设我们想要创建一个新列`full_name`，它是列`first_name`和`last_name`的串联，中间有一个空格。**

## **熊猫**

**我们可以在 Pandas 中使用一个简单的赋值操作符。**

## **结构化查询语言**

**在 SQL 中，第一步是添加新列:**

**接下来，我们使用 SQL 中的`SET`设置该值。**

**||在 Sqlite 中用作串联运算符。[延伸阅读](https://www.sqlitetutorial.net/sqlite-string-functions/sqlite-concat/)。**

# **结论**

**恭喜你！您现在知道了 Pandas 中最常见方法的 SQL 翻译。**

**我试着翻译了科学家们经常在熊猫身上使用的大部分数据。然而，我知道我可能错过了一些。**

**请在回复中让我知道。**

**一如既往的感谢阅读！**

**![](img/df8a7f30559e5cee8f5f08d62b54103e.png)**

**由作者使用稳定扩散生成和编辑的图像。**

**[🚀**订阅数据科学每日剂量。在这里，我分享关于数据科学的优雅技巧和诀窍，一天一个技巧。每天在你的收件箱里收到这些提示。**](https://avichawla.substack.com/)**

**[🧑‍💻**成为数据科学专家！获取包含 450 多个熊猫、NumPy 和 SQL 问题的免费数据科学掌握工具包。**](https://subscribepage.io/450q)**

**[**获取机器学习领域排名前 1%的研究论文、新闻、开源回购、推文的每周汇总。**](https://alphasignal.ai/?referrer=Chawla)**