# JSON 模式:NoSQL 数据的完整性检查

> 原文：<https://towardsdatascience.com/json-schema-integrity-checking-for-nosql-data-b1255f5ea17d>

# JSON 模式:NoSQL 数据的完整性检查

## 用 JSON 模式验证 JSON 和 CSV 数据的格式。Python 和 pandas 中的示例代码。

![](img/f2114755e7895cbea0db8a08a0d695de.png)

在 [Unsplash](https://unsplash.com/s/photos/programming?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上由 [Vipul Jha](https://unsplash.com/@lordarcadius?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 拍摄的照片

# 问题是

关系数据库有模式。这使您始终知道您拥有的数据是您期望的格式。例如,“名字”和“姓氏”字段必须具有非空字符串值，并且“薪金”必须包含大于 0 的整数。与模式不匹配的传入数据将被拒绝。

但是许多现代数据格式，比如 JSON，不是关系型的，也不强制使用模式。JSON 允许不同的数据类型与不同行中的相同标记相关联。例如，姓氏可以同时是“Jones”和 56.07，也可以完全不存在。也许更糟的是，一个数组可以有混合类型，例如["John "，" Betty "，13.89]

不合规的数据是下游运营的一个大问题。如果找到的值中有两个是“？？?"和“我希望我赚更多的钱”？如果 PRIMARY_CARE_DOC 字段为空，医院如何联系患者的医生？

# 拯救 JSON 模式

JSON 模式是一个可读的 JSON 对象，它描述了一组允许的 JSON 对象。

下面是一个简单的例子，说明 JSON 对象必须有名为 LNAME、FNAME 和 HOURLY_WAGE 的字段，LNAME 是必需的，没有其他字段。

```
{
 “type”: “object”,
 “properties”: {
   “HOURLY_WAGE”: {“type”: “number”},
   “FNAME”: {“type”: “string”},
   “LNAME”: {“type”: “string”}
   },
 “additionalProperties”: false,
 “required”: [“LNAME”]
}
```

所以这个 JSON 符合上面的模式:

```
{
  "LNAME": "Jones",
  "FNAME": "Betty",
  "HOURLY_WAGE": 15
}
```

但事实并非如此，因为缺少 LNAME 并且不允许使用 LAST_NAME:

```
{
  "LAST_NAME": "Jones",
  "FNAME": "Betty",
  "HOURLY_WAGE": 15
}
```

下面是一个更复杂的模式，它规定 LNAME 中必须有一些内容(不能是空字符串)，HOURLY_WAGE 必须在 10 到 30 之间，DEPT 必须包含一个列出的值。

```
{
 “type”: “object”,
 “properties”: {
   “HOURLY_WAGE”: {“type”: “number”, “minimum”: 10, “maximum”: 30},
   “FNAME”: {“type”: “string”},
   “LNAME”: {“type”: “string”, “minLength”: 1},
   “DEPT”: {“enum”: [“HR”, “Acct”, “Dev”]} 
   },
 “additionalProperties”: false,
 “required”: [“LNAME”, “DEPT”, “HOURLY_WAGE”]
}
```

所以这符合模式:

```
{
  "LNAME": "Jones",
  "FNAME": "Betty",
  "HOURLY_WAGE": 15,
  "DEPT": "Acct"
}
```

但这不是。(对象有四个错误— LNAME 为空、HOURLY_WAGE 太低、不允许 DEPT 值、不允许名称/字段管理器。)

```
{
  "LNAME": "",
  "FNAME": "Betty",
  "HOURLY_WAGE": 5,
  "DEPT": "Accounting",
  "MANAGER": "Bob"
}
```

到目前为止，我们已经将模式可视化地应用于一个 JSON 对象，这在您阅读文章时是很好的。在 Python 中，如何以编程方式做到这一点？轻松点。

```
from jsonschema import validatemy_schema = { "my schema here" }
my_json = { "my json object here" }try:
    validate(instance=my_json, schema=my_schema)
except:
    print ("Handle bad JSON here...")
```

# 能力

JSON Schema 已经从早期发展成为一种健壮的验证方法，并且拥有比上面显示的更多的功能。检查 JSON 对象的其他方法包括:

*   一个数是另一个数的偶数倍，如 5、10、15、20 等。
*   字符串匹配给定的正则表达式。
*   数组具有最小(或最大)数量的元素，或者具有唯一的元素。
*   字符串是有效的日期时间、日期或时间。
*   字符串是有效的电子邮件地址。
*   验证其他模式的元模式。
*   还有更多…

虽然它们看起来可能不像，但下面是完整的、有效的 JSON 对象，并且可以有一个 JSON 模式。然而，在我看来，这些“裸 JSON”对象是糟糕的样式，因为这些值没有名称标签，因此没有语义。但是请注意，如果遇到这种 JSON，您可以*为其定义模式。*

```
-77“hello”[4, 5, 6]
```

JSON 模式不具备现代关系系统(如 Oracle 数据库)的全部验证功能。有一些类型的完整性检查是 JSON Schema 做不到的。例如，您不能确保特定字段中的值也作为主键出现在另一个文件中(外键约束)。一般来说，JSON 模式可以查看一个对象(或相似对象的文件),但不能跨多个数据集查看。

# 提示和技巧

设计 JSON 模式是一门艺术。如果它们太具体，它们将只接受您想要的确切的数据格式，但却令人头疼，因为它们不允许合理的变化。例如，要求每个员工都有一个经理，这意味着您不能输入其经理刚刚辞职的员工。另一方面，如果模式太松散，它们会接受明显不好的数据，比如根本没有名字的客户。

虽然本文主要关注 JSON，但是您可以使用 JSON 模式来验证任何 NoSQL(非关系)数据格式。对于 CSV/TSV/YAML/etc，只需首先将字符串转换为 JSON，然后应用模式。显然，这样做是有时间/性能成本的，但是根据您的应用程序和数据大小，转换的时间可能是值得的。像这样的一些代码将把一个标准的 CSV 文件转换成有效的 JSON…

```
import pandas as pddf = pd.read_csv ("my_file.csv", header='infer', sep=',')df.to_json ("my_file.json", orient='records', lines=True)
```

# 了解更多信息

https://www.json.org/json-en.html

https://json-schema.org[(JSON 模式)](https://json-schema.org)

[https://jsonschema.net](https://jsonschema.net)(生成模式的 GUI 工具，以 JSON 对象为例)

[https://www.jsonschemavalidator.net](https://www.jsonschemavalidator.net)(测试模式的 GUI 工具)

[https://json-schema . org/implementations . html # validator-python](https://json-schema.org/implementations.html#validator-python)(应用 JSON 模式的 Python 库)