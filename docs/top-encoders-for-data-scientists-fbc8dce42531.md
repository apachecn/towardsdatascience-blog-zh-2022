# 数据科学家的顶级编码器

> 原文：<https://towardsdatascience.com/top-encoders-for-data-scientists-fbc8dce42531>

## 意见

## 2022 年，为您的机器学习算法提供更好的分类编码方法(而不是一次性编码)

![](img/05f9741cad0d330b3528fa0c53b6d30d.png)

桑迪·米勒在[Unsplash](https://unsplash.com/s/photos/transform?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【1】上拍摄的照片。

# 目录

1.  介绍
2.  目标编码器
3.  CatBoost 编码器
4.  CatBoost 库自动编码器
5.  摘要
6.  参考

# 介绍

曾经的编码统治者，*一键编码*，已经不复存在。更新、更好、更简单的编码方法已经出现，并开始取代传统的编码类型。在一个算法实现更快、特性可解释性更好的世界里，等等。，数据科学过程中仍有一部分相当陈旧和缺乏——编码——它花费的时间太长、太乏味、太混乱。并非所有的模型，但在一些模型中，是不能被模型吸收的分类特征。您会看到错误，并想，*为什么我不能使用一个看起来如此明显的特性，为什么我必须将我的数据框架扩展到数百列*？这些新的编码方法使您能够将更多的精力放在您想要使用的功能上，有些功能非常明显，而不是您愿意花费时间来消化它们。话虽如此，让我们更深入地探讨一下，如何通过对分类特征进行更好的编码来获得更好的结果。

# 目标编码器

![](img/f44c9d43eb3b79c64c1cf8e5b40e0497.png)

[布雷迪·贝里尼](https://unsplash.com/@brady_bellini?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在[Unsplash](https://unsplash.com/s/photos/target?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【2】上的照片。

以下编码器就是基于这种目标编码方法构建的。根据[sci kit-learn](https://contrib.scikit-learn.org/category_encoders/targetencoder.html)【3】，“*特征被替换为目标的后验概率，以及目标在你的训练数据上的先验概率*”。更简单地说，当您有一个分类特性时，您可以将列值从一个文本值/字符串替换为该分类值的数字表示。

> 以下是您将用于此编码的通用代码:

```
pip install category_encoders**import** **category_encoders** **as** **ce**target_encoder = ce.TargetEncoder(cols=['cat_col_1', 'cat_col_2'])target_encoder.fit(X, y)X_transformed = target_encoder.transform(X_pre_encoded)
```

所以，假设你有:

*   cat_col_1 =动物
*   cat_col_2 =天气类型

您可以用一个实际的 float 代替 one-hot-encoding，one-hot-encoding 会将可能的值转置到新的列中，如果实际上是该类别值，则表示为 0/1。

此外，不是`dog`、`fish`、`cat`，你会有→ 0.33、. 20、. 10 等。根据你的数据，和天气类型类似的东西，比如`raining`、`sunny`、`snowing` → 0.05、. 60、. 05 等。

*这种目标编码方法也是接下来两种方法的基础方法。*

# CatBoost 编码器

![](img/88cdc8fa137f9c00310b64147bb54272.png)

詹姆斯·萨顿在[Unsplash](https://unsplash.com/s/photos/cat?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【4】上的照片。

与上面的编码器类似，字符串/文本、对象或类别值类型将被替换为浮点型。代码也是一样的，除了用`CatBoostEncoder`替换`TargetEncoder`。

使用这种目标编码方法将编码器应用于分类要素的另一个好处是，您不仅可以减少数据帧的维数，还可以通过另一个库(如 SHAP)更容易地了解分类要素的要素重要性。

[CatBoost 编码](https://contrib.scikit-learn.org/category_encoders/catboost.html) [5]更好使用，因为它通过过去数据的有序目标编码以及更多随机排列来防止目标泄漏。

> 从 CatBoost 文档中，我们可以看到该方法是下面的[方法](https://catboost.ai/en/docs/concepts/algorithm-main-stages_cat-to-numberic)【6】，主要有以下好处，对此我表示赞同:

*   改变输入的随机顺序和数据类型的转换
*   通过利用不仅仅是数字特征来改进培训，因此您在预处理数据上花费的时间更少

# CatBoost 库自动编码器

![](img/0661baab9b64df329b626ac13dfe09f7.png)

图片由 [Geran de Klerk](https://unsplash.com/@gerandeklerk?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在[Unsplash](https://unsplash.com/s/photos/two-cats?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)【7】上拍摄。

有了一种全新的名为 CatBoost 的算法，可以自动对数据的类别数据类型进行编码，而不必手动对数据应用编码器。该算法结合了目标编码的优点，特别是 CatBoost 编码的优点，此外，还在本机代码中实现了功能解释。

虽然它与上面的编码器没有什么不同，但它甚至更容易使用，对我来说，这是对分类特征进行编码的最简单的方法，特别是当您还处理有时会混淆的数字特征时(*两种数据类型*)。

这种编码器不仅更容易使用，算法库和编码器经常击败大多数其他算法，包括 Random Forest 和 XGBoost，这在 CatBoosts 的主页上有文档记录。

> 以下是用 Python 实现 CatBoost 代码的简单方法:

```
# import your library
from catboost import CatBoostRegressorCAT_FEATURES = [] #list of your categorical features# set up the model
catboost_model = CatBoostRegressor(n_estimators=100,
                                   loss_function = 'RMSE',
                                   eval_metric = 'RMSE',
                                   cat_features = CAT_FEATURES)
# fit model
catboost_model.fit(X_train, y_trian, 
                   eval_set = (X_test, y_test),
                   use_best_model = True,
                   plot = True)# easily print out your SHAP feature importances, including your CAT_FEATURESshap_values = catboost_model.get_feature_importance(Pool(
                       X_train,
                       label = y_train,
                       cat_features = CAT_FEATURES
),type = "ShapValues")shap_values = shap_values[:,:-1]shap.summary_plot(shap_values, X_train, max_display=50)
```

# 摘要

正如你所看到的，数据科学总是在不断发展，编码方法也不例外。在本文中，我们看到了一些与分类变量的 one-hot-enoding 相反的替代方法。好处包括能够在总体上利用分类特征，以及通过更简单的解释轻松地接受它们，最终允许更快、更便宜(*有时*)和更准确的建模。

> 总而言之，以下是较好的编码类型:

```
* Target Encoder* CatBoost Encoder* CatBoost Library Automatic Encoder
```

我希望你觉得我的文章既有趣又有用。如果您同意或不同意这些编码器，请随时在下面发表评论。为什么或为什么不？关于数据科学预处理，您认为还有哪些编码器值得指出？这些当然可以进一步澄清，但我希望我能够为您的数据和数据科学模型提供一些更有益的编码器。

***我不隶属于这些公司。***

*请随时查看我的个人资料、* [Matt Przybyla](https://medium.com/u/abe5272eafd9?source=post_page-----fbc8dce42531--------------------------------) 、*和其他文章，并通过以下链接订阅接收我的博客的电子邮件通知，或通过点击屏幕顶部的订阅图标* *点击订阅图标* ***，如果您有任何问题或意见，请在 LinkedIn 上联系我。***

**订阅链接:【https://datascience2.medium.com/subscribe】**

**引荐链接:**[https://datascience2.medium.com/membership](https://datascience2.medium.com/membership)

(*如果你在 Medium* 上注册会员，我会收到一笔佣金)

# 参考

[1]桑迪·米勒在 [Unsplash](https://unsplash.com/s/photos/transform?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片，(2021)

[2]照片由[布雷迪·贝里尼](https://unsplash.com/@brady_bellini?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在[Unsplash](https://unsplash.com/s/photos/target?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)(2021)拍摄

[3] Will McGinnis，[sci kit-学习目标编码器](https://contrib.scikit-learn.org/category_encoders/targetencoder.html)，(2016)

[4]詹姆斯·萨顿(James Sutton)在 [Unsplash](https://unsplash.com/s/photos/cat?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的照片，(2018)

[5] Will McGinnis， [scikit-learn CatBoost 编码器](https://contrib.scikit-learn.org/category_encoders/catboost.html)，(2016)

[6] Yandex， [CatBoost —分类—数字](https://catboost.ai/en/docs/concepts/algorithm-main-stages_cat-to-numberic)，(2022)

[7]图片由 [Geran de Klerk](https://unsplash.com/@gerandeklerk?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 在[Unsplas](https://unsplash.com/s/photos/two-cats?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)(2017)上拍摄