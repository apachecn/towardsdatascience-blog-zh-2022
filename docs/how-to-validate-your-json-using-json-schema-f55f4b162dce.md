# 如何使用 JSON 模式验证您的 JSON

> 原文：<https://towardsdatascience.com/how-to-validate-your-json-using-json-schema-f55f4b162dce>

## Python 中 JSON 模式的简明介绍

![](img/03ab3f89e08bbecfb76dcc0335bbd7f0.png)

费伦茨·阿尔马西在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

想象以下场景:你和你的队友正在开发一个新功能。你的任务是创建一个包含一些结果的 JSON 并发送给你的队友。她的任务是获取这个 JSON，解析它并保存在数据库中。你们口头上同意了键和类型应该是什么，并且你们每个人都实现了他们的部分。听起来很合理，如果 JSON 结构简单，它确实可以工作。但是有一天你出了个 bug，把钥匙送错了。您吸取了教训，决定创建一个 API，并将其记录在您团队最喜欢的文档平台中。现在，你们都可以看看这个 API，以确保正确地实现了它。

但这就足够了吗？假设您确实正确地实现了它。假设另一个队友做了一个改变，现在它返回一个数字数组而不是单个数字。你的队友不知道你的 API，所有东西都坏了。

如果在发送和解析 JSON 之前，可以直接在代码中验证 JSON，会怎么样？这就是我们拥有 JSON 模式的目的！

在这篇文章中，我将介绍 JSON Schema，为什么它如此强大，以及我们如何在不同的场景中使用它。

# 什么是 JSON 模式？

JSON 模式是一种基于 JSON 的格式，用于定义 JSON 数据的结构。它为给定的应用程序需要什么样的 JSON 数据以及如何与之交互提供了一个契约。它可以用于 JSON 数据的验证、文档、超链接导航和交互控制。

该模式可以在 JSON 文件中定义，并加载到您的代码中，也可以直接在代码中创建。

## 如何验证我们的 JSON？

轻松点。

```
validate(instance=your_json, schema=schema)
```

[例如](https://python-jsonschema.readthedocs.io/en/stable/):

```
**from** **jsonschema** **import** validate

**>>>** *# A sample schema, like what we'd get from json.load()*
**>>>** schema = {
**... **    "type" : "object",
**... **    "properties" : {
**... **        "price" : {"type" : "number"},
**... **        "name" : {"type" : "string"},
**... **    },
**...** }

**>>>** *# If no exception is raised by validate(), the instance is valid.*
**>>>** validate(instance={"name" : "Eggs", "price" : 34.99}, schema=schema)

**>>>** validate(
**... **    instance={"name" : "Eggs", "price" : "Invalid"}, schema=schema,
**...** )                                   
Traceback (most recent call last):
    ...
ValidationError: 'Invalid' is not of type 'number'
```

# 我为什么要使用 JSON 模式？

每个 JSON 对象都有一个基本的键值结构。键是一个字符串，值可以是任何[类型](http://json-schema.org/understanding-json-schema/reference/type.html) — [数字](https://json-schema.org/understanding-json-schema/reference/numeric.html)、字符串、数组、JSON 等。

在某些情况下，该值只能是特定的类型，而在其他情况下，该值更加灵活。我们的 JSON 中有些键是必需的，有些是可选的。还有更复杂的场景。例如，如果我们得到了某个密钥，那么第二个密钥必须出现。一个键值可以依赖于第二个键值。

所有这些场景以及更多场景都可以使用 JSON Schema 进行本地测试和验证。通过使用它，您可以验证自己的 JSON，并确保它在与其他服务集成之前满足 API 要求。

# 简单 JSON 模式

在这个例子中，我们的 JSON 包含了关于狗的信息。

```
{
 "breed": "golden retriever",
 "age": 5,
 "weight": 13.5,
 "name": "Luke"
}
```

让我们更仔细地看看这个 JSON 的属性，以及我们希望对每个属性实施的要求:

*   品种——我们只想代表三个品种:金毛寻回犬、比利时马利诺犬和边境牧羊犬。我们想证实这一点。
*   年龄—我们希望将年龄四舍五入到年，因此我们的值将表示为整数。在本例中，我们还希望将最大年龄限制为 15 岁。
*   权重—可以是任何正数、整数或浮点数。
*   名称—始终是字符串。可以是任何字符串。

我们的计划是-

```
{
 "type": "object",
 "properties":
     {
        "breed": {"type":"string", "enum":[
                                            "golden retrievers", 
                                            "Belgian Malinois", 
                                            "Border Collie"
                                         ]
                 },
        "age": {"type": "int", "maximum":15, "minimum":0},
        "weight": {"type":"number", "minimum":0},
        "name": {"type":"string"}
     }
}
```

这样，只能添加 0 到 15 之间的年龄值，没有负体重，并且只有三个特定的品种。

# 简单数组模式

我们还可以验证数组值。

例如，我们想要一个具有以下属性的数组:2 到 5 项，唯一值，仅字符串。

```
['a','b','c']{
 "type": "array",
 "items": {"type": "string"},
 "minItems": 2,
 "maxItems": 5,
 "uniqueItems": true
}
```

# 更复杂的功能

## 必需的属性

有些属性是必需的，如果它们缺失，我们会提出一个错误。

可以添加`**required**`关键字。

```
{
 "type": "object",
 "properties":
     {
        "breed": {"type":"string"},
        "age": {"type": "int", "maximum":15, "minimum":0}
     }
 "required":["breed"]
}
```

在这种情况下，如果缺少“品种”属性，将会引发错误。其他属性如“年龄”仍然是可选的。

## 需要家属

如果给定的属性存在于对象中，关键字`**dependentRequired**`有条件地要求某些属性存在。

```
{
  "type": "object",

  "properties": {
    "name": { "type": "string" },
    "credit_card": { "type": "number" },
    "billing_address": { "type": "string" }
  },

  "required": ["name"],

  "dependentRequired": {
    "credit_card": ["billing_address"]
  }
}
```

在这种情况下，如果出现“信用卡”属性，则需要“帐单地址”。

## 一个/任何一个

到目前为止，每个属性只有一种类型。如果我们的财产可以有几种不同的类型呢？

示例 1 — `**anyOf**` —要根据`anyOf`进行验证，给定的数据必须对任何(一个或多个)给定的子模式有效。

```
{
  "anyOf": [
    { "type": "string"},
    { "type": "number", "minimum": 0 }
  ]
}
```

在这种情况下，我们的数据可以是字符串，也可以是大于或等于 0 的数字。

示例 2 — `**oneOf**` —为了对`oneOf`进行验证，给定的数据必须对其中一个给定的子模式有效。

```
{
  "oneOf": [
    { "type": "number", "multipleOf": 5 },
    { "type": "number", "multipleOf": 3 }
  ]
}
```

在这种情况下，数据只能是数字，可以是 5 的倍数，也可以是 3 的倍数，但不能两者都是！

# 摘要

JSON 模式是一个强大的工具。它使您能够验证您的 JSON 结构，并确保它满足所需的 API。您可以根据需要创建任意复杂和嵌套的模式，您所需要的只是需求。您可以将它作为附加测试或在运行时添加到您的代码中。

在这篇文章中，我介绍了基本结构，并提到了一些更复杂的选项。有很多可以探索和利用的东西，你可以阅读。

我认为任何将 JSONs 作为工作一部分的人都应该熟悉这个包及其选项。仅仅通过简单地验证您的 JSON 结构，它就有可能节省您大量的时间并简化您的集成过程。我知道自从我开始使用它以来，它节省了我很多时间。

 