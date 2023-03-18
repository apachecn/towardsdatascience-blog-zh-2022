# 如何使用 Jinja2 编写 dbt 宏

> 原文：<https://towardsdatascience.com/how-to-write-dbt-macros-using-jinja2-ff021d535bf3>

## 编写第一个 dbt 宏的教程和备忘单

![](img/43e0ef24bc205b171f374eb016084396.png)

照片由[沙哈达特·拉赫曼](https://unsplash.com/@hishahadat?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/s/photos/function?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

我最近遇到了一个难倒我的问题。那些不是最好的吗？他们立刻让我们变得卑微。

这个问题导致了我写的代码如何工作的许多挫折。这是我第一次和 [dbt 宏](https://docs.getdbt.com/docs/building-a-dbt-project/jinja-macros)一起深入挖掘。我以前使用过它们，但从未添加定制逻辑，我只是复制了我在 dbt 文档中找到的内容。

我没有纠结，而是将这个架构问题作为一个机会来学习更多关于 dbt 宏和 Jinja 函数的知识。并且，通过写这篇文章，我希望填补关于宏如何工作的知识空白。

# 什么是 dbt 宏？

如果你不知道， **dbt 宏**基本上就是嵌入在你的 SQL 代码中的函数。逻辑非常类似于一个简单的 Python 函数。他们利用一种叫做 Jinja 的语言来编写这些函数。虽然对于那些用 Python 编写函数的人来说，Jinja 非常简单，但是你仍然需要通读文档来理解它是如何工作的。

# 让我们复习一些基本的 Jinja 语法:

## 变量

如果你想**设置一个变量**，你可以这样写:

`{%- set default = development -%}`

`default`是变量的名称，`development`是被设置为等于变量的对象。记得用正确的语法将它包装起来:`{%- -%}`。

如果你想**引用一个变量**，你需要用两个弯曲的大括号把它括起来。像这样:

```
{{ default }}
```

## If 语句

编写 **if 块** is 几乎与 Python 相同，只是用适当的 Jinja 语法包装它们:

```
-- if 
{%- if default = development -%}-- else if
{%- elif default = production -%}-- else 
{%- else -%}-- closing the function
{%- endif -%}
```

确保用`{%- endif -%}`块关闭中频功能。

## 记录

记录变量是调试你编写的任何函数的重要部分。为了在 Jinja 中做到这一点，只需使用以下语法插入您希望打印到控制台的变量:

`{% do log(node, info=true) %}`

在这里，您将把节点的信息记录到您的控制台。请记住，您需要设置`info=true`来将信息记录到您的控制台上，而不仅仅是记录在日志文件中。

# 编写宏

现在，如何将所有这些放在一起编写 dbt 宏呢？您可以像定义 Python 函数一样定义它。

首先，必须用 word macro 打开它，并用适当的 Jinja 语法包装它。像这样:

```
{% macro -%}
```

然后，在单词`macro`之后，必须指定宏的名称，类似于指定 Python 函数的名称。

```
{% macro generate_schema_name() -%}
```

如果函数接受任何变量作为输入，则将这些变量添加到括号中。

```
{% macro generate_schema_name(custom_schema_name, node) -%}
```

同样，类似于 Python(我听起来像一个坏掉的记录)，您可以为这些变量设置默认值，以防在引用函数时没有变量传入。

```
{% macro generate_schema_name(custom_schema_name=none, node=none) -%}
```

最后但同样重要的是，当您在宏中编写完逻辑后，必须用下面的代码块关闭它:

```
{%- endmacro %}
```

不算太坏，对吧？

# 要引用的 dbt 变量

文档太多总比太少好。但是，有时候，当有这么多好的信息时，你不可能找到你所需要的。纠结了几天，终于找到 dbt 的 [Jinja 函数文档](https://docs.getdbt.com/reference/dbt-jinja-functions)。

这在尝试创建自定义宏时非常有用。它会给你一个好主意，告诉你如何利用现有资源实现你的愿景。

不要逐一查看每个人，你可以自己完成，让我们回顾一下我发现最有帮助的。

## 结节

节点引用特定 dbt 模型的所有配置设置。这在提取模型的名称或路径时经常被引用。我强烈建议像我们上面所做的那样记录节点，以查看您可以从中提取用于宏的所有不同信息。

```
{% do log(node, info=true) %}
```

例如，我经常在宏代码中使用`node.path`,以便获得我的模型的文件路径。如果一个模型在某个文件夹中，我写代码做一件事。如果它在另一个文件夹中，我写代码做另一件事。

## 目标

这是宏中另一个常用的变量。如果您熟悉 dbt，那么您应该知道目标是用来设置您的开发环境的。你的大多数项目可能都有一个`dev`目标和一个`prod`目标。每种模式下的数据库和模式信息都有所不同。

目标变量是您希望如何引用这些不同的信息。您可以通过调用`target.name`使用当前目标名称，通过调用`target.database`使用当前目标的数据库。

这在根据您的工作环境设置不同条件时特别有用。您可以通过在 if 语句中使用`target.name`来查看您是在 dev 还是 prod 中。

# 我们写个宏吧！

现在，是时候用我们刚刚学过的东西来写你的第一个宏了。虽然 dbt 已经在每个项目中内置了一些[宏](https://docs.getdbt.com/docs/building-a-dbt-project/building-models/using-custom-schemas)，比如`create_schema_name`和`drop_old_relations`，但是知道如何定制这些宏来满足您的需求也很重要。就我个人而言，我重新配置了`create_schema_name`来匹配我想要如何命名我的 dbt 模型。

## 我来分享一下我的做法。

首先，我从宏的基本外壳开始。这包括起始行和结束行，起始行包含函数的名称。

```
{% macro generate_schema_name() -%}{%- endmacro %}
```

接下来，我添加了两个 if 语句和一个 else 语句。我知道我想要包含两个条件和一个 else 块来捕捉任何不满足任一条件的情况。不要忘记你的结尾部分！

```
{% macro generate_schema_name() -%} {%- if -%} {%- elif -%} {%- else -%} {%- endif -%}{%- endmacro %}
```

现在，我要设置什么条件呢？让我们根据目标中设置的环境来执行每项操作。一个用于开发场景，另一个用于生产场景。

```
{% macro generate_schema_name() -%} {%- if target.name == 'dev'-%} {%- elif target.name == 'prod' -%} {%- else -%} {%- endif -%}{%- endmacro %}
```

注意，我们的函数名是`generate_schema_name()`。这意味着我们希望根据环境设置不同的模式。命名 dev 和 prod 中的模式有什么意义？

这都是个人喜好。这并不是我的数据库和模式实际上是如何建立的，只是在这个例子中我将如何使用它们。

我想编写一个宏，将开发中的所有模型的模式设置为`data_mart_dev`。然而，我希望所有的生产模型都是我的`dbt_project.yml`中定义的。

注意，这个宏`generate_schema_name()`是一个已经由 dbt 编写的宏，它接受一个变量`custom_schema_name`作为输入。`Custom_schema_name`指项目文件中指定的内容。此外，我们还想传入模型的节点。

```
{% macro generate_schema_name(custom_schema_name, node) -%} {%- if target.name == 'dev'-%} {%- elif target.name == 'prod' -%} {%- else -%} {%- endif -%}{%- endmacro %}
```

现在，我们希望将`data_mart_dev`设置为宏中第一个 if 块内的模式，因为我们希望所有的开发模型都在这里构建。字符串被简单地写成不带任何引号的普通文本。

```
{% macro generate_schema_name(custom_schema_name, node) -%} {%- if target.name == 'dev'-%} data_mart_dev {%- elif target.name == 'prod' -%} {%- else -%} {%- endif -%}{%- endmacro %}
```

接下来，我们需要在第二个 if 块中将模式设置为`custom_schema_name`。用`{{ }}`引用变量。在管道后面添加`trim`将在调用变量时消除任何空白。

```
{% macro generate_schema_name(custom_schema_name, node) -%} {%- if target.name == 'dev'-%} data_mart_dev {%- elif target.name == 'prod' -%}

      {{ custom_schema_name | trim }}   {%- else -%} {%- endif -%}{%- endmacro %}
```

最后，我们需要定义当这些条件都不满足时，模式应该是什么。让我们创建一个默认变量，并将其设置为等于目标模式。

```
{% macro generate_schema_name(custom_schema_name, node) -%} default_schema = target.schema   {%- if target.name == 'dev'-%} data_mart_dev {%- elif target.name == 'prod' -%}

      {{ custom_schema_name | trim }} {%- else -%} {{ default_schema | trim }} {%- endif -%}{%- endmacro %}
```

现在我们的宏完成了！您选择运行的每个 dbt 模型都会自动调用这个函数。您的`dev`模型将在模式`data_mart_dev`中创建，您的`prod`模型将在您的`dbt_project.yml`文件中定义的模式中创建。

# 结论

恭喜你。您已经编写了第一个 dbt 宏，它为不同的开发环境设置了不同的模式名。dbt 宏可能很难理解，但是一旦你理解了，你可以用它做很多很酷的事情。定制您的项目，使它们更有效，以及测试您的模型的可能性是无穷无尽的。

但是，要确保不要过度。使用 dbt 的好处是它使您的代码模块化。您的代码应该被存储起来，以便可以在多个地方重复使用。宏也是如此。你不想为一个非常小的用例编写宏。您希望宏应用于项目中的多个位置。当您决定是否应该编写新宏时，请记住这一点。

如果你有兴趣学习更多关于 dbt 宏的知识，一定要订阅我在 Substack 上的[新闻简报](https://madisonmae.substack.com/p/hey-there-heres-what-you-can-expect)。很快我将分享一篇关于我最近为一个非常具体的用例编写的宏的独家文章。它将包含大量有价值的信息，您可能希望将这些信息应用到您自己的 dbt 模型中！快乐编码。