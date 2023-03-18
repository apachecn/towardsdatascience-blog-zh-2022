# 如何利用零方数据获得更深刻的见解

> 原文：<https://towardsdatascience.com/how-to-leverage-zero-party-data-for-greater-insights-f4de0e2801d6>

## 用一个简单的 Python 演示。

![](img/306e60027cd1db6b2c700c569ad63898.png)

在 [Unsplash](https://unsplash.com/s/photos/data?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄的 [ThisisEngineering RAEng](https://unsplash.com/@thisisengineering?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

第三方数据是推动现代营销和分析的最重要的燃料。它是来自公司外部的数据，无论是来自客户、供应商还是其他来源。

这对于清楚地了解您的业务表现以及如何改进非常重要。无论你何时访问一个应用程序，从[跟踪 cookies](/hundreds-of-companies-are-having-a-party-with-my-cookie-data-and-i-wasnt-invited-e7f70837b27) 到社交媒体登录，这些数据通常都会在后台自动收集。

但在当今数据驱动的世界中，另一种类型的数据变得越来越重要:零方数据。零方数据由 Forrester Research 创造，是客户“有意和主动与品牌分享”的数据

在本文中，我们将讨论为什么零方数据很重要，并通过 Python 中的一个示例演示如何使用它来提高机器学习模型的准确性。

# 为什么零方数据很重要

“数据是新的石油”这句老生常谈的话今天仍然适用。你拥有的数据越多，你就能更好地洞察你的业务和客户。零方数据是一个有价值的信息来源。

零方数据的优势之一是它总是最新的。它总是最新的，因为客户自愿与您共享它。这与第三方数据形成对比，如果不定期更新，第三方数据可能会过时(甚至被欺诈性地收集)。

零派对数据的另一个优势是它更加个人化和相关。因为客户是自愿分享的，所以更有可能与他们的兴趣和需求相关。这与第三方数据形成对比，第三方数据可能与您的客户不太相关，甚至无关。

# 数量与机器学习的准确性相关

不仅如此，[你有越多的数据](/6-ways-to-improve-your-ml-model-accuracy-ec5c9599c436)来推动机器学习算法，它们就能更好地工作。虽然这种[并不总是](/why-more-data-is-not-always-better-de96723d1499)的情况，但今天最先进的模型，从 OpenAI 的 GPT-3 到悟道 2.0，都依赖于惊人的数据量。

机器学习对企业越来越重要，因为它可以帮助你自动化客户细分和线索评分等任务。

但重要的不仅仅是数量。数据的质量也很重要。这是零方数据的亮点，因为它比第三方数据更有可能是高质量的。

# 如何获得零方数据

现在你知道了为什么零方数据如此重要，那么你该如何着手获取它呢？这里有一些建议。

## **向你的顾客索要。**

获得零方数据的最好方法之一就是直接向你的客户索要。您可以通过多种方式做到这一点，例如通过调查或要求他们与您分享联系信息。

传统的方式是通过弹窗“询问”，甚至是隐性的批准进行用户追踪。虽然第三方数据收集工具运行从[脸书像素](https://www.facebook.com/business/learn/facebook-ads-pixel)到 [Hotjar](https://www.hotjar.com/) 到 [Mixpanel](https://mixpanel.com/) 的色域，但零方数据收集是较新的。有了 [involve.me](https://www.involve.me/) 这样的工具，就可以通过小测验、调查、计算器等形式获得零方数据。

## **为分享提供奖励。**

另一种获得零方数据的方法是提供激励让客户与你分享。这可以是折扣、奖励或其他特别优惠的形式。

# 实用指南:用零方数据构建预测模型

既然您已经理解了零方数据的重要性，那么您实际上如何将它合并到您的预测建模中呢？让我们看一个 Python 中的简单例子，比较和对比使用零方数据和传统数据的结果。

我们将使用[银行客户调查—定期存款营销数据集](https://www.kaggle.com/sharanmk/bank-marketing-term-deposit)，这是一个 UCI ML 数据集，包含银行电话营销数据，如年龄、工作、婚姻状况、教育、余额、住房和个人贷款状况。目标是预测客户是否会认购定期存款。

第一步是导入必要的库。

```
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
```

现在，让我们导入数据集。

```
df = pd.read_csv('bank_customer_survey.csv')
```

由于我们的数据集充满了非数值，让我们对其进行编码。

```
df['y'] = df['y'].astype(int)
df = df.apply(LabelEncoder().fit_transform)
```

首先，我们将使用通用的、预先包含的第三方数据构建一个模型。

```
X = df # our input data
y = pd.DataFrame(X.pop(‘y’))X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
logreg = LogisticRegression()logreg.fit(X_train, y_train)print(‘Accuracy of Logistic regression classifier on training set: {:.2f}’.format(logreg.score(X_train, y_train)))
print(‘Accuracy of Logistic regression classifier on test set: {:.2f}’.format(logreg.score(X_test, y_test)))
```

我们得到以下输出:

```
Accuracy of Logistic regression classifier on training set: 0.89
Accuracy of Logistic regression classifier on test set: 0.89
```

换句话说，我们可以使用逻辑回归分类器以 89%的准确率预测客户是否会订阅定期存款。

现在，在添加了一些零方数据之后，是时候构建相同的模型了。出于演示目的，我制作了一个综合专栏，向客户提出一个调查问题，并给出一个自愿回答(零方数据的示例):

> “未来 5 年要不要买房？”

我们可以伪随机地将我们的合成列添加到数据框架中，其中“是”的答案与购买定期存款相关联，这在现实世界的数据中可能会发生。

```
dfNo = df[df['y']==0]
dfYes = df[df['y']==1]lstYes = [0,1,1,1,1]
dfYes[“Do you want to buy a house in the next 5 years?”] = np.random.choice(lstYes, size=len(dfYes))lstNo = [0,0,0,0,1]
dfNo[“Do you want to buy a house in the next 5 years?”] = np.random.choice(lstNo, size=len(dfNo))dfZeroPartyData = dfNo.append(dfYes)
```

现在，让我们重建我们的模型。

```
X = dfZeroPartyData # our input data
y = pd.DataFrame(X.pop(‘y’))X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)logreg = LogisticRegression()
logreg.fit(X_train, y_train)print(‘Accuracy of Logistic regression classifier on training set: {:.2f}’.format(logreg.score(X_train, y_train)))
print(‘Accuracy of Logistic regression classifier on test set: {:.2f}’.format(logreg.score(X_test, y_test)))
```

我们得到以下输出:

```
Accuracy of Logistic regression classifier on training set: 0.92
Accuracy of Logistic regression classifier on test set: 0.92
```

换句话说，我们使用逻辑回归分类器以 92%的准确率预测了客户是否会订阅定期存款。通过添加零方数据，我们将准确率提高了 3%。鉴于数据生成的随机性，您可能会得到不同的结果。

当然，这只是一个使用合成数据的演示示例，但这些好处每天都在现实世界中实现，企业使用零方数据或客户有意和主动与品牌分享的数据来建立更好的模型。

# 代码

请参见[这个 Google Colab 文件](https://colab.research.google.com/drive/1z4xU6HnOu5LB6GI9JY75UiHRMIiu1Xmm?usp=sharing)获取完整代码，并在评论中随意提出任何问题！