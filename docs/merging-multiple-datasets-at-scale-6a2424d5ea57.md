# 大规模合并多个数据集

> 原文：<https://towardsdatascience.com/merging-multiple-datasets-at-scale-6a2424d5ea57>

## 在现代数据堆栈上连接数据

![](img/cbd3ee17a3314f9d5936077ab8dab63e.png)

照片由[蒂姆·福斯特](https://unsplash.com/@timberfoster?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

从事数据科学工作时，数据科学家很少能够在一个文件中找到项目所需的所有数据。即使数据总是在单个文件中可用，特征工程技术也经常创建额外的数据集，这意味着数据将需要结合在一起。在现代数据堆栈上，这可以使用 SQL 来完成，如这里的和这里的所述。对于使用 Python 的数据科学家来说，这可以通过 pandas 中的`merge`命令来完成。

使用 [AdventureWorks](https://github.com/microsoft/sql-server-samples/tree/master/samples/databases/adventure-works) 数据(在麻省理工学院许可下发布，更多文档见[此处](https://docs.microsoft.com/en-us/sql/samples/adventureworks-install-configure?view=sql-server-ver15&tabs=ssms))，考虑理解促销对互联网销售的影响的问题。一个 Jupyter 笔记本展示了如何将所有的冒险工作数据读入熊猫数据帧，可以在这里找到。检查 internet_sales 数据框架(在笔记本中称为 *import_sales* ),它看起来很有希望，因为它包含购买信息、显示促销的促销关键字、购买的折扣金额和总销售额。可惜数据好像不好。无论是否有促销活动，折扣金额总是为零，尽管有各种促销活动，销售额也是相同的。

在真实数据中遇到这类数据错误并不罕见。虽然这可能是最初使用 AdventureWorks 数据时产生的错误，但在现实世界中，这通常是由于软件中的问题以及在保存发票之前未能正确应用促销详细信息而导致的。正如数据科学中经常出现的情况，需要找到一种变通方法。幸运的是，存在一个促销数据框架(在上面的笔记本中称为 *import_promotion* ),其中包含关于每次促销和折扣百分比的信息。这个数据将需要加入到销售数据和实际销售价格计算。

## 熊猫合并

将 AdventureWorks 销售和促销数据分别提取为互联网销售和促销数据的`import_sales`和`import_promotion`。首先，我们可以创建一个简化的促销数据集，只包含*PROMOTIONKEY*和*DISCOUNTPCT*。

```
promotiondf = import_promotion[['PROMOTIONKEY', 
                                'DISCOUNTPCT']].copy()
```

此促销数据框架可与原始销售数据框架结合，如下所示:

```
sales = pd.merge(import_sales, 
                 lean_promo,
                 on='PROMOTIONKEY',
                 how='left')
```

或者，这可以计算为:

```
sales = import_sales.merge(import_sales, 
                           lean_promo,
                           on='PROMOTIONKEY',
                           how='left')
```

dataframe `sales`现在包含了折扣前的价格和应该应用的折扣百分比。实际销售价格可以计算为:

```
sales['ACTUALSALESAMT'] = (1 - sales['DISCOUNTPCT']) *
                               sales['SALESAMOUNT']
```

现在，可以用实际销售额来完成对折扣影响的进一步分析。

虽然以这种方式使用熊猫既普遍又有效，但它也不是没有缺点。特别是，使用 pandas 需要将数据从数据仓库中提取出来，并移动到可以进行计算和合并的机器上，然后再发送回数据仓库进行存储。在海量数据集的情况下，这可能是不可行的。即使是中等大小的数据适合服务器的内存，将数据提取到 pandas 并将其推回仓库的时间也可能超过进行分析本身的时间。

## 使用 RasgoQL 在数据仓库中执行此操作

不用移动数据来使用 pandas，同样的连接和计算可以在数据仓库中完成。并不是所有的数据科学家都喜欢使用 SQL 或者愿意继续使用 Python。这是因为有更好的 ide 可用，或者在这项工作中能够快速分析数据。开源包 RasgoQL 有助于弥合这一差距，它允许数据科学家在 Python 中工作，但在数据仓库中执行代码。

首先，RasgoQL 需要云中的表的句柄。这是通过`dataset`功能完成的:

```
salesds = rql.dataset('ADVENTUREWORKS.PUBLIC.FACTINTERNETSALES')
promotionds = rql.dataset('ADVENTUREWORKS.PUBLIC.DIMPROMOTION')
```

接下来，将促销数据与销售数据连接起来:

```
sales = sales_ds.join(join_table=promotion_ds.fqtn, 
                      join_type='RIGHT',
                      join_columns={'PROMOTIONKEY':'PROMOTIONKEY'})
```

可以应用`math`函数来创建实际销售额:

```
sales = sales.math(math_ops=['(1 - DISCOUNTPCT) * SALESAMOUNT'],
                             names=['ACTUALSALESAMT'])
```

可以通过运行`preview`将这些数据下载到 python 环境中，作为数据帧中十行的样本:

```
sales_sample_df = sales.preview()
```

或者可以下载完整数据:

```
sales_df = sales.to_df()
```

为了让每个人都可以在数据仓库上查看这些数据，可以使用`save`发布这些数据:

```
sales.save(table_name='<Actual Sales Data Tablename>',
           table_type='view')
```

或者，将此保存为表格，将 **table_type** 从**“视图”**更改为**“表格”**。如果您想检查 RasgoQL 用来创建这个表或视图的 SQL，运行`sql`:

```
print(sales.sql())
```

最后，如果您在生产中使用 dbt，并且希望将这项工作导出为 dbt 模型，供您的数据工程团队使用，请致电`to_dbt`:

```
sales.to_dbt(project_directory='<path to dbt project>'
```

这种方法导致在数据仓库上生成并执行 SQL 连接，从而允许连接所有数据，而无需将数据移动到工作站。此外，由于这会创建在仓库上运行的 SQL，因此这些特性会自动准备好用于生产，而无需将 Python 代码转换为 SQL。

当处理少量数据时，pandas 会是一个有用的工具。随着数据集变得越来越大，如果不是不可能的话，继续使用 pandas 会很困难。大量的 Python 包会有所帮助，但是利用数据仓库中可用的处理能力，最好的解决方案可能是在那里执行尽可能多的计算。

将数据准备过程迁移到数据仓库是一项挑战，通常需要数据工程的帮助。使用开源包 RasgoQL，数据科学家可以在数据仓库中利用 SQL 的强大功能，而不需要额外的资源。

如果你想查看 RasgoQL，文档可以在这里找到和库[这里](https://github.com/rasgointelligence/RasgoQL)。