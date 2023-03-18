# 现代数据堆栈上的插补

> 原文：<https://towardsdatascience.com/imputation-on-the-modern-data-stack-6c0e37e20e7a>

## 规模估算

![](img/e2e1acbb214a01dccb0b0ecb9cc8c3b2.png)

[西格蒙德](https://unsplash.com/@sigmund?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

由于大多数机器学习算法不处理空值，缺失值插补是特征工程管道的标准部分。这种插补有多种技术，从非常简单到创建复杂的模型来推断值应该是多少。

最简单的方法是为缺失值估算一个值，最常见的是要素的平均值或中值。对于基于树的方法，输入极值(大于要素的最大值或小于最小值)也是可行的。通常，数据科学家认为这些简单的技术不足以替代丢失的值。在这些情况下，可以创建一个模型，根据该观察的其他特征来估算值。这可以从相对简单的(kNN 模型)到与原始机器学习工作范围相似的复杂模型。虽然这种基于模型的方法可能很有吸引力，但现代机器学习算法的强大功能意味着，输入单个值并创建一个缺失指标也可以轻松完成。在这个博客中，我们将估算平均值，但是对代码的简单修改将允许我们估算其他值。

## Python 中的插补

在 Python 中，有两种常用的方法来执行这种插补。首先，这个插补可以用熊猫来做。

```
df['varname'] = df['varname'].fillna(df['varname'].mean())
```

为了创建缺失的指标，需要首先创建它

```
df['varname_mi'] = df['varname'].isna()
df['varname'] = df['varname'].fillna(df['varname'].mean())
```

通常，这在 Scikit-learn 中作为特性工程管道的一部分来完成。使用`train_test_split`创建训练和测试分区，可以创建一个`SimpleImputer`来替换丢失的值。

```
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_splitX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1066)imp = SimpleImputer(missing_values=np.NaN, strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)
```

与 pandas 不同，`SimpleImputer`可以通过改变调用来同时创建一个缺失值指示器，包括将参数 *add_indicator* 设置为 *True。*

```
imp = SimpleImputer(missing_values=np.NaN, 
                    strategy='mean',
                    add_indicator=True)
```

## 挑战

在这两种情况下，这些方法都适用于可在单个工作站上管理的数据集。在数据集包含数百千兆字节或万亿字节数据的问题中，这种方法不再有效。虽然 Dask 之类的包可以让您处理这些大规模数据集，但您仍然需要将数据移动到您的机器或集群中。对于这些大小的数据集，移动数据的时间可能会很长。即使数据大小易于管理，当最终模型投入生产时，这些基于 Python 的方法仍会造成问题，因为这些插补需要在生产代码中重复。当生产环境不是 Python 或者不支持加载 Python 对象时，这可能是一个挑战。

## 利用云数据仓库的力量进行插补

使用 RasgoQL 可以缓解这两个问题。RasgoQL 在数据仓库中执行插补，并允许您保存所有特征工程步骤，以便在预测时可以在仓库中重新运行。

要使用 RasgoQL 的转换，首先用`RasgoQL`连接到系统。

```
import rasgoql
creds = rasgoql.SnowflakeCredentials.from_env()
rql = rasgoql.connect(creds)
```

并用`rql.dataset(fqtn=<Fully Qualified Table Name>)`得到需要对其进行缺失值插补的数据集(数据仓库中的一个表或视图)。所有支持的转换都可以在这里[找到](https://docs.rasgoql.com/primitives/transform)或者以编程方式列出:

```
transforms = rql.list_transforms()
**print(transforms)**
```

若要转换数据，请对数据集调用转换名称。

```
newset = dataset.impute(imputations={'VARIABLE_1': 'mean',
                                     'VARIABLE_2': 'mean'},
                                     flag_missing_vals=False)
```

请注意，插补转换采用需要插补的字段字典，并允许不同的插补方法:*平均值、中值、众数*或单个值。此外，将 *flag_missing_vals* 设置为 *True* 将导致为插补字典中的每个字段创建缺失指标特征。

## 转换链接

通过将多个调用附加到转换函数上，或者通过在前一个转换的输出上运行一个转换函数，可以将多个转换相互链接起来。这允许您从相对简单的转换构建更复杂的转换，并通过一个 SQL 调用创建最终的数据集。转换链的结果可以被视为一个数据帧，它包含:

```
df_transform = newset.preview()
print(df_transform)
```

默认情况下，`preview`返回十行。要将整个数据作为数据帧下载，调用`to_df`。

由于这些转换是在云数据仓库上运行的，所以它运行的是 SQL 代码。转换链的 SQL 语句可以通过运行

```
print(newset.sql())
```

虽然这段代码创建了转换链，但是它还没有执行，结果保存在仓库中。为了保存它，需要通过调用链上的`save`来创建一个新表。

```
newset.save(table_name='<NEW TABLE NAME>',
            table_type='view')
```

该数据被保存为视图。为了将其保存为表格，请将 **table_type** 改为**‘表格’**。

此时，Rasgo 已经针对数据仓库中存储的全部数据运行了缺失值插补，并将结果作为任何 Python 环境中的数据帧返回，并且可以使用标准工具用于建模或可视化。此外，如果这被保存为视图，当新数据被添加到数据仓库中的基础表时，不需要重新运行数据准备阶段，因为估算数据会自动运行并可供下载。

这种准备是自动运行的，这影响了将数据准备管道投入生产的能力。无需打包 python 代码或将其重写为生产系统中使用的语言，数据会自动在仓库中保持最新，相关数据行可通过已创建的插补提取。

现在，不用浪费几周或几个月的时间重写(或等待软件工程重写)和测试代码的输出，数据立即可用。

最后，由于这种插补在数据仓库中立即可用，这意味着其他数据科学家和分析师不再需要提取原始数据并准备供自己使用。相反，最终转换后的数据可以直接从仓库中提取，并用于建模、可视化或报告。

为了了解这些工具的强大功能，我们创建了一个数据集，其中包括几个关键字段中缺失的数据，并上传到我们的数据仓库。在创建和不创建缺失指标的情况下，对缺失字段的平均值进行估算。最后，为两种情况训练一个 [CatBoost](https://catboost.ai/) 分类器，并计算对数损失和 AUC。

```
y = transformed_df[target]
X = transformed_df.drop(columns=[target])
X_train, X_test, y_train, y_test = train_test_split(X, y,     
                                                    test_size=.2,                  
                                                  random_state=1066)categorical_features = X_train.select_dtypes(exclude=[np.number])train_pool = Pool(X_train, y_train, categorical_features)
test_pool = Pool(X_test, y_test, categorical_features)model = CatBoostClassifier(iterations=1000, 
                           max_depth=5,
                           learning_rate=0.05,
                           random_seed=1066,
                           logging_level='Silent', 
                           loss_function='Logloss')model.fit(X_train, 
          y_train, 
          eval_set=test_pool, 
          cat_features=categorical_features, 
          use_best_model=True, 
          early_stopping_rounds=50)y_predict = model.predict_proba(X_test)logloss = log_loss(y_test, y_predict[:, 1])
auc = roc_auc_score(y_test, y_predict[:, 1])print(logloss, auc)
```

这给出了最终结果

*   无缺失指标:AUC: 0.6881，对数损失:0.2574
*   缺失指标:AUC: 0.7045，对数损失 0.2563

在这种情况下，使用 CatBoost，缺少指标的模型比没有指标的模型略胜一筹

如果您想查看 RasgoQL，可以在这里找到文档[，在这里](https://docs.rasgoql.com/)找到资源库[。](https://github.com/rasgointelligence/RasgoQL)