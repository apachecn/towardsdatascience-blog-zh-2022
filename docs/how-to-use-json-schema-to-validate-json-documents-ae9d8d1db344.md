# 如何使用 JSON 模式在 Python 中验证 JSON 文档

> 原文：<https://towardsdatascience.com/how-to-use-json-schema-to-validate-json-documents-ae9d8d1db344>

## 了解确保数据质量的标准方法

![](img/7b756843abf369457baf066378441934.png)

[图片由 kreatikar 拍摄于 Pixabay](https://pixabay.com/illustrations/developer-programmer-technology-3461405/)

JSON 文档可以包含任意数量的键/值对。键必须是字符串，但值可以是任何支持的类型，如字符串、数字、布尔值等。该值甚至可以是复杂类型，如数组或嵌套对象。这使得 JSON 文档既非常灵活又非常松散。然而，这使得数据处理更加困难，因为数据团队经常通过 API 获取数据，而这些 API 的响应通常是 JSON 格式的。拥有一致的数据格式可以使数据管道更加健壮。有了统一的数据输入，就不需要担心意外的数据类型，也不需要在数据清理上花太多时间。因此，您可以更专注于数据分析，提高工作效率。

在这篇文章中，我们将介绍如何使用 JSON 模式来验证 JSON 文档。基本概念以及基本和高级用例将通过简单的代码片段进行介绍，这些代码片段易于理解。

## 什么是 JSON 模式？

JSON 模式是定义一些 JSON 数据模式的 JSON 文档。好吧，老实说，这种解释非常奇怪和难以理解，但一旦我们看到后面的代码，就会变得更加清晰。目前，我们需要理解两点:

*   JSON 模式本身是一个具有键/值对的有效 JSON 文档。每个键都有特殊的含义，用于定义一些 JSON 数据的模式。
*   模式类似于 SQL 数据库中的表定义，定义 JSON 中字段的数据类型。它还定义了哪些字段是必需的，哪些是可选的。

让我们从一个简单的 JSON 模式开始:

这个 JSON 模式指定目标 JSON 是一个具有两个 ***属性*** (通常也称为 ***键***/**/*字段*** 的对象，并且`name`属性是必需的。让我们更深入地研究一下每个验证关键字:

*   `type`关键字指定目标 JSON 是一个对象。它也可以是一个数组，通常是 API 响应的对象数组。我们将在后面看到如何定义数组字段的模式。然而，在大多数情况下，顶级的`type`几乎总是`object`。
*   `properties`关键字为 JSON 对象的每个字段指定模式。目标 JSON 的每个字段都被指定为一个键/值对，键是实际的字段名，值是目标 JSON 中的字段类型。每个字段的关键字`type`与顶级关键字的含义相同。可以想象，这里的`type`也可以是`object`。在这种情况下，相应的字段将是一个嵌套对象，这将在后面演示。
*   `required`关键字是一个数组，包含需要存在的属性。如果此处指定的任何属性缺失，将引发`ValidationError`。

除了基本的验证关键字，即上面指定的`type`、`properties`和`required`，还有其他模式关键字可以在在线文档和一些工具自动生成的 JSON 模式中看到。

有两个模式关键字，即`[$schema](https://json-schema.org/draft/2020-12/json-schema-core.html#name-the-schema-keyword)`和`[$id](https://json-schema.org/draft/2020-12/json-schema-core.html#name-the-id-keyword)`。`$schema`定义用于模式的[草图](https://json-schema.org/specification-links.html)。如果未指定`$schema`，将使用最新的[草图](https://json-schema.org/draft/2020-12/release-notes.html)，这通常是所希望的。作为一个初学者，如果你过多地潜入水中，你可能会很容易迷路。我们通常不需要接触`$schema`字段，我们会在这篇文章的最后介绍一点。另一方面，`$id`为模式定义了一个 URI，使得当前模式可以从外部被其他模式访问。如果没有指定`$id`，那么当前的模式只能在本地使用，这也是正常情况下需要的，至少对于小项目是这样。然而，对于更大的项目，您的机构可能有一个关于如何存储模式以及如何引用它们的内部系统。在这种情况下，您可以相应地设置`$id` 关键字。

有两个注释关键字，即`title`和`description`，它们分别指定 JSON 模式的标题和描述。它们可用于文档，并使您的模式更易于阅读和理解。它们也会被一些图形工具很好地显示出来。为了简单起见，在本文中不会指定它们，但是通常应该将它们添加到您的项目中以获得最佳实践。

## 如何用 Python 验证 JSON 文档和模式？

在 Python 中，我们可以使用`[jsonschema](https://pypi.org/project/jsonschema/)`库来验证一个 ***JSON 实例*** (也可以称为 ***JSON 文档*** ，只要它是明确的)。可以安装 [pip](https://pip.pypa.io/en/stable/) :

让我们根据上面定义的 JSON 模式来验证一些 JSON 实例。注意，从技术上讲 [JSON 是一个字符串](/how-to-work-with-json-in-python-bdfaa3074ee4)，但是我们需要指定要验证的 JSON 的底层数据，这样更方便。

这表明所定义的模式可以按预期用于验证 JSON 实例。不正确的数据类型或缺少某些必填字段将触发`ValidationError`。但是，应该注意的是，默认情况下允许附加字段，这可能是也可能不是您想要的。如果您想要一个严格的模式，并且只允许由`properties`关键字定义的字段，那么您可以将`additionalProperties`指定为`False`:

## 如何定义数组字段的模式？

尽管将数组作为顶级字段并不常见，但将其作为属性却非常常见。让我们给上面定义的模式添加一个数组属性。我们需要将`type`设置为`array`，并用`items`关键字为每一项指定`type`:

正如我们看到的，可以正确检查数组元素的类型。但是，默认情况下允许空数组。为了改变这种行为，我们可以将`minItems`设置为 1，或者您期望的对您的情况有意义的数字。

## 如何定义嵌套对象字段的模式？

如上所述，属性的`type`关键字与顶级关键字具有相同的含义和语法。因此，如果一个属性的`type`是`object`，那么这个属性就是一个嵌套对象。让我们给 JSON 数据添加一个`address`属性，它将是一个嵌套对象:

正如我们所看到的，嵌套对象字段具有与顶级字段完全相同的模式定义语法。因此，为嵌套对象定义模式相当简单。

## 使用`$defs`来避免代码重复。

如果需要在同一个模式中的多个地方使用`address`字段，该怎么办？如果我们在任何需要的地方复制字段定义，就会有代码重复，这是程序员所讨厌的，因为它不是干的。在 JSON 模式定义中，我们可以使用`[$defs](https://json-schema.org/draft/2020-12/json-schema-core.html#name-schema-re-use-with-defs)`关键字来定义可以在其他地方引用的小子模式，以避免代码重复。让我们用`$defs`重构上面的模式，以潜在地避免代码重复:

正如我们所看到的，使用`$defs`定义子模式的新模式的工作方式和以前一样。但是，如果需要在同一模式的不同地方使用`address`字段，它的优点是可以避免代码重复。

## 如何设置元组字段的模式？

最后，如果我们希望`scores`字段是一个元素数量固定的元组呢？可惜 JSON schema 中没有 tuple 字段，我们需要通过数组来实现一个 tuple 的定义。[一般逻辑](https://json-schema.org/draft/2020-12/release-notes.html)是一个数组有项(`items`)并且可选地有一些位置定义的项在普通项(`prefixItems`)之前。对于一个元组，只有`prefixItems`没有`items`，达到了一个元组有固定数量的元素的效果。重要的是，每个元组元素的类型必须显式定义。

如果您想为一个 tuple 字段定义模式，您需要对 JSON schema 的草稿有所了解，这要高级一些。草案是 JSON 模式的标准或规范，定义了验证器应该如何解析模式。有几个草案可用，最新的是[2020–12](https://json-schema.org/draft/2020-12/release-notes.html)。你可以在这里找到草稿列表。

正常情况下，我们不需要担心`$schema`字段和要使用的草稿。然而，当我们需要定义一个元组字段时，这是我们应该注意的事情。

如果安装的`[jsonschema](https://python-jsonschema.readthedocs.io/en/stable/)`库是最新版本(撰写本文时为 v4.9.0)，那么将使用最新草案([2020–12](https://json-schema.org/draft/2020-12/release-notes.html))。如果这是您想要的版本，您不需要通过`$schema`关键字指定草稿。然而，为了清晰起见，最好总是在 JSON 模式中指定草稿的版本。为了简单起见，在这篇文章的开头省略了它，这样你就不会不知所措，但是建议在实践中使用它。

另一方面，如果您想使用一个不同的草案版本而不是最新的版本，那么您需要在草案版本中明确指定关键字`$schema`。否则无法正常工作。

让我们分别使用草稿 2020–12 和 2019–09 定义`scores`字段的模式，并演示如何使用`$schema`关键字以及如何相应地定义元组字段:

正如我们所看到的，使用关键字`prefixItems`和`items`的 draft 2020–12 元组字段的模式定义更直观，因此推荐使用。有关元组字段定义从 2019–09 到 2020–12 的变更的更详细说明，请查看[本发行说明](https://json-schema.org/draft/2020-12/release-notes.html)。

此外，需要注意的是，即使我们希望`scores`字段是一个元组，它也必须被指定为一个数组(Python 中的 list ),而不是验证器的元组。不然就不行了。

## 使用验证器有效地验证多个 JSON 文档

如果你有一个有效的 JSON 模式，并想用它来验证许多 JSON 文档，那么[推荐](https://python-jsonschema.readthedocs.io/en/stable/validate/#jsonschema.validate)使用`Validator.validate`方法，这比`jsonchema.validate` API 更有效。一个[验证器](https://python-jsonschema.readthedocs.io/en/stable/validate/#versioned-validators)是一个实现特定草稿的特殊类。比如有`Draft202012Validator`、`Draft201909Validator`、`Draft7Validator`等。如果类名中没有指定草案版本，`Validator`本身意味着所有验证器类都应该遵守的协议(类似于接口)。

除了工作方式类似于`jsonchema.validate` API 的`Validator.validate`方法之外，您还可以使用`Validator.check_schema`来检查一个模式对于一个特定的草稿是否有效。你也可以使用`Validator.is_valid`来悄悄地检查一个 JSON 是否有效，如果无效就不引发`ValidationError`。让我们用一些简单的例子来演示这些方法的用法，这样可以使它们更容易理解:

在这篇文章中，我们介绍了什么是 JSON 模式，以及如何使用它来验证 JSON 文档中的不同数据类型。我们已经介绍了字符串和数字等基本数据类型以及数组和嵌套对象等复杂数据类型的基础知识。此外，我们还学习了如何使用`$defs`关键字来避免代码重复，该关键字用于定义子模式，对于复杂的模式来说非常方便。最后但并非最不重要的是，草案的基础知识介绍。我们现在知道了如何用不同的草案定义一个 tuple 字段的模式，以及如何用一个使用特定草案的验证器更有效地验证多个 JSON 文档。

相关文章:

*   [如何在 Python 中执行 JSON 转换、序列化和比较](/how-to-work-with-json-in-python-bdfaa3074ee4)
*   [使用 mypy 和 pydantic 进行 Python 类型化和验证](https://lynn-kwong.medium.com/python-typing-and-validation-with-mypy-and-pydantic-a2563d67e6d)