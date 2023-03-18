# 创建电影分级模型第 2 部分:五个特征工程策略！

> 原文：<https://towardsdatascience.com/creating-a-movie-rating-model-part-2-five-feature-engineering-tactics-dc9d363bebcd>

![](img/eaece29bee0b74d9b492d30979dca877.png)

## 电影分级模型

## 研究我在原始数据集上执行特征工程时应用的策略

你好，朋友们！我们将回到我们的电影分级模型项目，对我们收集的数据进行一些功能工程。如果你错过了第一个帖子，你可以通过这个链接赶上[。如果你在](https://medium.com/tag/dkhundley-movie-model) [YouTube](https://youtube.com/playlist?list=PLNBQNFhVrlVRCyhEM0c9dTkWswzk_FeaR) 上关注我，你会知道我实际上在这个项目上以直播的形式取得了很多进展，我现在以博客的形式来分享我们迄今为止所做的事情！

简单回顾一下上一篇文章，我们刚刚从许多不同的来源收集了所有的数据，并将原始数据集保存为 CSV 格式。如果您想查看我们收集的数据，请查看“支持数据”部分，这是我的 GitHub 库的 `[README](https://github.com/dkhundley/movie-ratings-model)` [的一部分，点击这里](https://github.com/dkhundley/movie-ratings-model)。到目前为止，数据没有任何改变。

在我们可以开始用数据测试许多不同的机器学习算法之前，我们需要在我们称为**特征工程**的实践中适当地清理和设计我们的数据。现在，这篇博文的目标不是要讨论我在原始数据上执行的所有特征工程操作的本质细节。欢迎您在这个 Jupyter 笔记本[中看到所有这些细节，但是这篇文章的目的是为您提供一些关于如何为您自己的项目执行特性工程的考虑，而不管主题是什么。因此，虽然我们显然是在管理我们的数据集来预测电影评级，但下面的策略适用于任何项目。](https://github.com/dkhundley/movie-ratings-model/blob/main/notebooks/feature-engineering.ipynb)

事不宜迟，让我们跳进我用过的各种战术！

# 策略 0:知道何时忽略无用的数据

好吧，我知道这个帖子说的是五种战术，但如果你包括这一种，我想你可以说有六种战术。我称之为“策略#0 ”,因为如果你有一个包含所有有用特征的原始数据集，那么你就不需要这么做。此外，我认为你可以争辩说，这种策略不完全是一种特性工程策略，因为我们没有做任何工程。它只是从最终数据集中删除要素。

当我们收集原始数据时，我们从电影数据库(TMDb)中获得了一个名为“TMDb Popularity”的字段。我当时不确定这个特性有什么作用，但不管怎样，它是和原始数据集一起收集的。经过进一步分析，我发现这是一个高度可变的特性，用于计算某一天某部电影的受欢迎程度。例如，1999 年的原创电影*《黑客帝国》*在 2019 年并不那么受欢迎，但在 2021 年末随着该系列的最新作品*《黑客帝国复活》*在影院首映，变得非常受欢迎。因为“TMDb 流行度”是非常主观可变的，所以我认为它对我们的最终数据集来说不是一个好特性，并完全放弃了这个特性。

# 策略 1:简单的数字编码

数据集中的要素包含基于二进制字符串的值是很常见的。在我的特定数据集中，我的用于确定二进制肯定/否定支持率的预测要素在值中编码为“是”或“否”。因为机器学习算法需要处理数值数据，所以我们需要将这些基于字符串的值转换为数值。

因为值是二进制的(是/否)，我们可以执行一个简单的数字编码，将“是”值转换为数字 1，将“否”值转换为数字 0。下面是为我们的项目实现这一目的的代码:

```
*# Performing the encoding of the "biehn_yes_or_no" feature*
**for** index, row **in** df**.**iterrows():
    movie_name **=** row['movie_name']
    **if** row['biehn_yes_or_no'] **==** 'Yes':
        df**.**loc[index, 'biehn_yes_or_no'] **=** 1
    **elif** row['biehn_yes_or_no'] **==** 'No':
        df**.**loc[index, 'biehn_yes_or_no'] **=** 0*# Changing the datatype of the 'biehn_yes_or_no' to int*
df['biehn_yes_or_no'] **=** df['biehn_yes_or_no']**.**astype(int)
```

# 策略 2:一个热门编码

虽然数字编码对我们上面的二进制特征很好，但我们有更多的特征，比二进制特征有更多的类别，如“是/否”。我们不能像上面那样进行简单的数字编码，因为这将固有地产生偏差，因为我们基本上是将特征转换为序数特征。在我的特定数据集中，这些分类特征之一包括主要和次要电影类型。自然，电影类型并没有一个固有的顺序。比如“喜剧类”电影，客观上并不比“剧情类”电影好。进行数字编码会产生顺序偏差，这是我们不想做的。

代替数字编码，我们想要执行一个热编码。这将把原始分类特征扩展成许多新特征，每个新特征代表在分类特征中找到的单个类别。就我们的电影类型而言，这意味着每部电影都将有一个新的个人特色。当然，对于这些电影中的一些，我们会遇到空值的问题，所以当一个热编码恰当地捕获这些空值时，我们还需要创建一个“全部捕获”特性。在我的具体例子中，我没有太多电影类型的空值，所以我很喜欢这种“全部包含”的空特性方法。如果您有大量的空值，您可能需要重新考虑您的方法。

下面是我执行的代码，用于对电影类型执行 one hot 编码。

```
*# Defining the OneHotEncoders for the genre columns*
primary_genre_encoder **=** OneHotEncoder(use_cat_names **=** **True**, handle_unknown **=** 'ignore')
secondary_genre_encoder **=** OneHotEncoder(use_cat_names **=** **True**, handle_unknown **=** 'ignore')*# Getting the one-hot encoded dummies for each of the genre columns*
primary_genre_dummies **=** primary_genre_encoder**.**fit_transform(df['primary_genre'])
secondary_genre_dummies **=** secondary_genre_encoder**.**fit_transform(df['secondary_genre'])*# Concatenating the genre dummies to the original dataframe*
df **=** pd**.**concat([df, primary_genre_dummies, secondary_genre_dummies], axis **=** 1)*# Dropping the original genre columns*
df**.**drop(columns **=** ['primary_genre', 'secondary_genre'], inplace **=** **True**)
```

# 策略 3:从无用的绝对值中产生相对值

乍一看，这种策略可能听起来令人困惑，但从我们的电影数据集中的一个例子来看，它是有意义的。我一路上收集到的一个特征是电影最初的上映年份。这个特性是一个绝对的整数值，因此，例如，电影*《黑客帝国》*在这里的值是 1999。正如前面的策略中提到的，按原样使用该数据会产生序数偏差，因为它会根据其固有的无意义的年份值对数据进行不公平的加权。

好消息是，如果我们把它转换成更相关的东西，这仍然是一个有价值的特性。具体来说，我认为这将是一个很有帮助的功能，可以知道电影评论家卡兰·比恩(Caelan Biehn)更喜欢老电影还是新电影。也就是说，我们可以将这个原始的`year`特性转换成一个更相关的`movie_age`特性，它指示从原始版本到现在的 2022 年已经过去了多少年。(如果我想在未来保持这种模式的更新，我必须在每个日历年重新进行培训。)

下面是我运行来执行这个相关工程的具体代码。

```
*# Extracting current year*
currentYear **=** datetime**.**now()**.**year*# Engineering the "year" column to be a relative "movie_age" column based on number of years since original release*
**for** index, row **in** df**.**iterrows():
    movie_name **=** row['movie_name']
    year_released **=** row['year']
    movie_age **=** currentYear **-** year_released
    df**.**loc[index, 'movie_age'] **=** movie_age
```

# 策略 4:删除不必要的字符

根据原始数据的存储方式，保持原样可能是好的，或者可能需要在传递到机器学习算法之前进行非常轻微的修改。在我的特定数据集中，烂番茄评论家分数存储为基于字符串的原始值，因为它的末尾有一个百分号(%)。我们想要利用与这个百分比相关的数值，但是我们不需要那个百分比符号。也就是说，我们可以执行下面的代码来去掉这个百分号，并将这个特性从基于字符串的特性转换为基于整数的特性。

```
*# Removing percentage sign from RT critic score*
**for** index, row **in** df**.**iterrows():
    **if** pd**.**notnull(row['rt_critic_score']):
        df**.**loc[index, 'rt_critic_score'] **=** int(row['rt_critic_score'][:2])
```

# 策略 5:处理所有的空值

在将数据传递给机器学习算法之前，我们需要对原始数据集中存在的所有空值做一些处理。这将是一个包含许多不同策略的大杂烩，在我自己的原始数据集中，我做了许多不同的事情。如何处理特定数据集的空值取决于您，但这里有几个我如何处理数据集的空值的例子:

*   **烂番茄评论家得分**:看着[这篇新闻文章](https://morningconsult.com/2019/10/29/rotten-tomatoes-scores-continue-to-freshen-what-does-this-mean-for-movies/)，看起来平均评论家得分徘徊在 59%左右，所以我在这里用 59 分估算所有的空值。
*   **Metacritic 的 Metascore** :这个很棘手。虽然我能够找到上面的来源指出烂番茄评论家分数为 59%,但我找不到 Metascore 的对等物。不幸的是，我们将不得不以 50 英里的速度行驶在路中间。
*   烂番茄观众评分:这也是一个很难处理的问题，因为我找不到一个可以给出明确答案的来源。从我分析数据的时间来看，我发现虽然评论家和观众可以在他们的论点上有所不同，但他们似乎都有一个电影评分在 59%左右的钟形曲线。因此，为了匹配烂番茄评论家的评分，我们也要用值 59 来填充这些空值。

下面是我运行的代码来支持上面的空插补。

```
*# Filling rt_critic_score nulls with critic average of 59%*
df['rt_critic_score']**.**fillna(59, inplace **=** **True**)*# Transforming RT critic score into an integer datatype*
df['rt_critic_score'] **=** df['rt_critic_score']**.**astype(int)*# Filling metascore nulls with 50.0*
df['metascore']**.**fillna(50.0, inplace **=** **True**)*# Filling rt_audience_score with audience average of 59%*
df['rt_audience_score']**.**fillna(59.0, inplace **=** **True**)
```

这篇文章到此结束！在执行了所有的特征工程之后，我们现在准备尝试将这些清理后的数据拟合到许多不同的机器学习算法中。在下一篇博文中，我们将测试许多不同的二进制分类和回归算法，以支持对每个评论分数的推断。我们将看看这些算法在验证指标方面的表现，然后从中选择一个作为我们的最终候选。令人兴奋的东西！感谢你阅读这篇文章，我们下期再见！