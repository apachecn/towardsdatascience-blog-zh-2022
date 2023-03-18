# 机器学习算法的分类变量

> 原文：<https://towardsdatascience.com/categorical-variables-for-machine-learning-algorithms-d2768d587ab6>

## Python 和数据仓库中的一键编码

![](img/5071e368593c196053b97a046ee51b48.png)

照片由[上的](https://unsplash.com/@burst?utm_source=medium&utm_medium=referral)爆裂[未爆裂](https://unsplash.com?utm_source=medium&utm_medium=referral)

虽然大多数机器学习算法只处理数值，但许多重要的现实世界特征不是数值而是分类的。作为分类特征，它们具有层次或价值。这些可以表示为各种类别，例如年龄、州或客户类型。或者，这些可以通过宁滨潜在的数字特征来创建，例如通过年龄范围(例如，0-10、11-18、19-30、30-50 等)来识别个体。).最后，这些可以是数值标识符，其中值之间的关系没有意义。邮政编码就是一个常见的例子。数字上接近的两个邮政编码可能比数字上远离的另一个邮政编码相距更远。

由于这些分类特征不能直接用于大多数机器学习算法，分类特征需要转换成数字特征。虽然有许多技术可以转换这些特性，但最常用的技术是一键编码。

在一次性编码中，分类变量被转换成一组二进制指示符(整个数据集中的每个类别一个指示符)。因此，在包含晴朗、部分多云、雨、风、雪、多云、雾的级别的类别中，将创建包含 *1* 或 *0* 的七个新变量。然后，对于每个观察，与类别匹配的变量将被设置为 *1* ，所有其他变量被设置为 *0。*

## 使用 scikit 编码-学习

使用 scikit-learn 预处理库可以很容易地执行一键编码。在这段代码中，我们将从 dataframe *df 对存储在变量 *column* 中的列进行一次性编码。*

```
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
onehotarray = encoder.fit_transform(df[[column]]).toarray()
items = [f'{column}_{item}' for item in encoder.categories_[0]]
df[items] = onehotarray
```

这将由一位热码编码器创建的列添加回原始数据帧中，并通过原始列名以及类别级别来命名每个添加的列。

## 数据库中的一次性编码

假设数据已经存在于数据仓库中。开源 Python 包 RasgoQL 可以直接在仓库上执行这种编码，而完全不需要移动数据，而不是将数据提取到系统上的 pandas 并转换数据以将数据推回仓库。这种方法节省了移动数据的时间和费用，可以处理太大而不适合单台机器内存的数据集，并在新数据到达仓库时自动生成编码变量。这意味着编码不仅适用于现有的数据和建模管道，还自动适用于生产管道。

要使用 RasgoQL 执行一键编码，可以使用下面的代码。

```
onehotset = dataset.one_hot_encode(column=column)
```

同样，要编码的列存储在 Python 变量*列中。*

可以通过运行`preview`将这些数据下载到 python 环境中，作为数据帧中十行的样本:

```
onehot_df = onehotset.preview()
```

或者可以下载完整数据:

```
onehot_df = onehotset.to_df()
```

为了让每个人都可以在数据仓库上看到这些数据，可以使用`save`发布这些数据:

```
onehotset.save(table_name='<One hot Tablename>',
               table_type='view')
```

或者，将此保存为表格，将 **table_type** 从**‘view’**更改为**‘table’**。如果您想检查 RasgoQL 用来创建这个表或视图的 SQL，运行`sql`:

```
print(onehotset.sql())
```

最后，如果您在生产中使用 dbt，并且希望将这项工作导出为 dbt 模型供您的数据工程团队使用，请调用`to_dbt`:

```
onehotset.to_dbt(project_directory='<path to dbt project>'
```

一键编码是许多机器学习管道中应用的最常见的特征工程技术之一。使用 scikit-learn 很容易应用，但是，除非生产系统运行 Python，否则将需要重新编写以将该步骤迁移到生产系统。在现代数据堆栈中，这种迁移通常涉及将代码重写为 SQL。

为了在 SQL 中执行一次性编码，首先要识别分类变量中的所有级别，然后生成 case 语句来填充每个级别的二进制指示符变量。这可能非常乏味。使用 RasgoQL，SQL 代码会自动生成并保存到数据仓库中。这意味着尽管仍在使用 Python 和 pandas，数据科学家已经生成了可以直接用于生产的代码。

将建模过程中开发的代码转换为生产就绪代码的过程可能是整个数据科学生命周期中最耗时的步骤之一。使用一种工具，可以很容易地将建模代码转移到产品中，这可以节省大量的时间。此外，通过将数据准备任务移动到数据仓库，数据科学家可以处理存储在数据仓库中的全部数据，并且不再受到将数据传输到他们的计算环境所花费的时间的限制，也不再受到环境存储器的大小限制。

如果你想查看 RasgoQL，可以在这里找到文档[，在这里](https://docs.rasgoql.com/)找到存储库[。](https://github.com/rasgointelligence/RasgoQL)