# 如何从 Azure DevOps 管道使用 Azure SQL 访问令牌身份验证

> 原文：<https://towardsdatascience.com/how-to-use-azure-sql-access-token-authentication-from-azure-devops-pipelines-344fa7dafa49>

## 使用 Python 和 Azure PowerShell

![](img/afc768540178e1cca6321308fabe8fcf.png)

戴维·霍伊泽在 [Unsplash](https://unsplash.com/s/photos/rotterdam?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上的照片

如果您需要从 DevOps 部署管道访问 Azure SQL 数据库，以便在数据库上执行一些自定义脚本。如果你需要使用用户名和密码认证之外的东西，并希望利用 Azure Active Directory，使用访问令牌可能是你的解决方案。

在这篇文章中，我将提供一个如何使用 Azure Active Directory 访问令牌向 Azure SQL 进行身份验证的示例。Azure DevOps 服务连接用于获取访问令牌。要做到这一点，一个先决条件是拥有一个作为用户添加到数据库的服务连接。建立服务连接的推荐方式是使用 Azure Active Directory 服务主体，也称为应用程序注册。

**！！从安全角度来看，不建议使用带有用户名和密码**的 SQL 认证。在 Azure SQL 的当前版本中，甚至可以完全关闭 SQL 身份验证，而只使用 Active Directory 身份验证。

**先决条件**

1.  带有服务连接的 Azure DevOps 项目
2.  一个 SQL 数据库，其中一个数据库用户代表服务连接

**使用 Azure PowerShell 获取访问令牌**

在下面的示例中，我们使用 Azure 管道的 Azure PowerShell 任务来利用服务连接凭据获取访问令牌。使用 SQLServer PowerShell 模块，我们可以使用“Invoke-Sqlcmd”来执行查询。

**通过 Python 使用访问令牌**

在下一个示例中，我们安装 pyodbc 模块，并针对我们的数据库执行一个定制的 python 脚本。确保编写一些逻辑来将参数传递和捕捉到 python 脚本中。我添加了一个示例 python 函数，并设置了连接字符串。我花了很长时间才让它工作起来。问题出在连接字符串上，在[文档](https://docs.microsoft.com/en-us/sql/connect/odbc/using-azure-active-directory?view=azuresqldb-current#authenticating-with-an-access-token)中有提到，但是我却漏掉了这句话:“*连接字符串不能包含* `*UID*` *、* `*PWD*` *、* `*Authentication*` *，或者* `*Trusted_Connection*` *关键字*”

为了能够使用访问令牌，需要一个函数来“扩展”访问令牌。更多信息参见[文档](https://docs.microsoft.com/en-us/python/api/adal/adal.authentication_context.AuthenticationContext?view=azure-python#methods)。

现在，您可以根据用户被添加到的数据库角色对数据库进行各种操作了！删除表时一定要小心:)

**参考文献**:

[微软文档——Azure SQL 使用访问令牌进行认证](https://docs.microsoft.com/en-us/sql/connect/odbc/using-azure-active-directory?view=azuresqldb-current#authenticating-with-an-access-token)

[微软文档—使用 Azure 活动目录的 Azure SQL](https://docs.microsoft.com/en-us/sql/connect/odbc/using-azure-active-directory?view=azuresqldb-current)

[微软文档—使用 Azure SQL 配置和管理 Azure AD 身份验证](https://docs.microsoft.com/en-us/azure/azure-sql/database/authentication-aad-configure?tabs=azure-powershell)

[微软文档—使用 Python 进行认证](http://us/python/api/adal/adal.authentication_context.AuthenticationContext?view=azure-python#methods)