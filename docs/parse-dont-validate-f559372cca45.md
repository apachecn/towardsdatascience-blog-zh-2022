# 解析，不验证| Python 模式

> 原文：<https://towardsdatascience.com/parse-dont-validate-f559372cca45>

## 验证数据就像看门人一样，将数据解析成有意义的数据类型，为原始数据增加了有价值的信息

![](img/69f64df8ef7583032df90d24382f51fd.png)

照片由[威尔·波拉达](https://unsplash.com/@will0629?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

当使用处理外部数据的应用程序时，我们通常会设置几层验证和数据转换来保护我们的业务逻辑不会崩溃……或者更糟，成为攻击的受害者。

第一层保护包括验证输入数据的一致性及其实际有效性。我们不想让我们的系统受到 SQL 注入攻击，甚至是无法正确处理的不完整字段。保护信息系统的一种广泛而经典的方法是使用数据验证:我们执行有效性检查，如果它们没有全部通过，那么我们就丢弃传入的数据。

为了简单起见，在本文中，我们将假设已经执行了安全检查，并且我们对数据模型的数据一致性感兴趣。所以就从它开始吧。

# 简单的例子

假设一家商店要求我们构建一个组件来联系他们的客户。客户可以将她或他的数据插入到表单中，然后收到回复。这些数据包括姓名、电子邮件和电话号码。此外，还有根据所提供的联系类型联系用户的功能。

用 Python 建模数据最直接的方法是使用字典:

```
customer1 = {
    "name": "John", 
    "surname": "Smith",
    "email": "john.smith@mymail.com",
}
customer2 = {
    "name": "Jessica",
    "surname": "Allyson",
    "telephone": "0123456789",
}
```

这种方法遵循了 Rich Hickey 在他著名的演讲“也许不是”中强调的观点。在那次演讲中，Clojure 的发明者捍卫了只使用实际存在的字段的字典，而不是使用可选字段(在他提到的语言中称为 Maybe)的想法。

里奇·希基(Rich Hickey)是一位魅力非凡、技术高超的工程师和社区领袖，人们很容易被他的谈话所说服。然而，这种数据建模有一个主要的缺点:在验证了字典的结构之后，验证之后的代码将没有关于数据内部结构的更多信息。

让我们用一个例子来说明:

```
def validate_customer_data(customer: Dict[str, str]) -> None:
    if "name" not in customer or "surname" not in customer:
        raise ValueError("Customer data must contain keys name and surname")
    if "email" not in customer and "telephone" not in custormer):
        raise ValueError("At least one field among email and telephone must be present in customer data")def contact_customer(customer: Dict[str, str]) -> None:    
    if "telephone" in customer:
        open_call(customer["telephone"])
    elif "email" in customer:
        open_email_client(customer["email"]) customer = receive_customer_data()
validate_customer_data()
contact_customer(customer)
```

在上面这段代码中，我们首先以某种方式接收客户数据*，然后验证该数据，最后用它联系客户。然而，经过验证，我们只知道客户向我们提供了至少一个来自电子邮件或电话号码，而不是这两个中的哪一个。我们从验证函数中得到的服务是，如果两个字段都不存在，就抛出一个错误。contact_customer 函数需要控制这两个字段中的哪一个出现，并相应地采取行动。*

*请注意，基于(数据)类的建模方法只会迫使我们处理无对象，在这里没有多大帮助:*

```
*from dataclasses import dataclass@dataclass
class Customer:
    name: str
    surname: str
    email: Optional[str]
    telephone: Optional[str] def contact(self) -> None:
       if self.telephone is not None:
           open_call(self.telephone)
       elif self.email is not None:
           open_email_client(self.email)*
```

*在这里，在做出决策之前控制数据结构的代码看起来微不足道，不值得进行更复杂的设计选择。然而，当数据结构变得比一个玩具示例更大时，控制代码将相应地增加(在某些情况下以超线性的方式)。这可能导致性能成本，并且容易出错。*

*这一节要带回家的要点是，在验证函数时会学到一些额外的知识，而这些知识在函数返回的那一刻就会丢失。*

*我想在本文中讨论的解决方案是将数据解析成一种类型，这种类型包含了我们作为验证的一部分而生成的信息。这种方法受益于 Python 类型的静态部分，我在上一篇文章中已经提到过。*

*</strong-static-typing-to-prevent-illegal-code-states-7a13e122cbab>  

# 解析，不验证

解析，不验证是 Alexis King 在她的同名文章中引入的一个概念，该文章关注 Haskell 语言。然而，Haskell 和 Python 之间的巨大差异不应该让你认为相同的思想不能移植到 Python 中。事实上，解析是 [Pydantic library](https://pydantic-docs.helpmanual.io/) 的基本思想，我建议您在阅读完本文后查阅一下。一旦想法得到澄清，即使没有 Haskell 知识，King 的文章也应该非常容易理解。

主要思想是验证防止坏数据进入我们的应用程序。然而，一些信息是在验证过程中创建的，然后立即被丢弃。另一方面，解析允许我们拒绝格式错误的数据，但也允许我们将有效数据映射到保留这些信息的数据类型。应用程序的其余部分将使用这些类型来阻止已经执行的控件。

例如，在上面的例子中，字典可以被映射到新的类型:

```
@dataclass
class CustomerWithPhone:
  name: str
  surname: str
  phone: str@dataclass
class CustomerWithEmail:
  class CustomerWithPhone:
  name: str
  surname: str
  email: str@dataclass
class CustomerWithPhoneAndEmail:
  class CustomerWithPhone:
  name: str
  surname: str
  phone: str
  email: str
```

上面的类也可以有自己的方法来联系客户，现在不再需要检查有效字段。此外，Python 为通用函数提供了[单一分派](https://docs.python.org/3/library/functools.html#functools.singledispatch):

```
from functools import singledispatch @singledispatch
def contact_customer(customer):
  raise NotImplementedError("Cannot contact a customer of unknown type")@contact_customer.register(CustomerWithPhone)
def _contact_customer_by_phone(customer):  # this name is irrelevant 
  .
  .
  call_number(customer.phone)
  .
  .@contact_customer.register(CustomerWithEmail)
def _contact_customer_by_email(customer):  # this name is irrelevant 
  .
  .
  send_email(customer.email)
  .
  .
```

上面的代码声明了一个名为 contact_customer 的通用函数，并为它定义了两个不同的实现，分别用于两个输入类型 CustomerWithPhone 和 CustomerWithEmail。Python 解释器在运行时根据对象的运行时类型决定调用哪个函数。您不需要编写任何代码来决定要调用的正确函数。

为了从我们定义的类型中获得最大利益，使用静态类型检查变得很重要，例如使用 [mypy](https://pypi.org/project/mypy/) ，这样您就可以自动控制静态和动态的代码正确性。这里重要的是，它们不是手动控制，也不是由您(或您的团队)必须维护的代码组成的。

使用静态类型处理解析的数据是确保代码正确性的一种方式。你永远不会调用错误的函数或读取错误的属性。Mypy 将为您检查并突出显示此类错误。在自动完成和可变建议方面，我们还免费获得 IDE 支持。

# 结论

静态类型不是 Python 语言最初设计的一部分，但随着时间的推移，它正成为一个越来越强大的功能。大型项目可以从可以静态检查的深思熟虑的数据类型中受益匪浅。

使用数据类型的第一步是将应用程序的输入数据转换成这种类型。解析就是这个过程，它检查数据的正确性，然后将它们分配到应用程序中正确的位置。静态检查和动态多态可以帮助开发人员通过为数据本身的控件编写尽可能少的代码来减少错误的数量。

</python-polymorphism-with-class-discovery-28908ac6456f>  </machine-translation-evaluation-with-cometinho-c89880731409>  </tips-for-reading-and-writing-an-ml-research-paper-a505863055cf>  

# 中等会员

你喜欢我的文章吗？你是否正在考虑申请一个中级会员来无限制地阅读我的文章？

如果您通过此链接订阅，您将通过您的订阅支持我，无需为您支付额外费用【https://medium.com/@mattiadigangi/membership*