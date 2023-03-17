# Python 自动化:使用 Cx Oracle 连接到 Oracle 数据库

> 原文：<https://towardsdatascience.com/automation-python-scripts-connecting-to-oracle-database-using-cx-oracle-1fe1801821b5>

## 我如何使用 Cx Oracle 和其他一些 Python 库来自动化小型 ETL 任务

![](img/5c7b4b285455e276a1669eda6fde7c55.png)

来自[像素](https://www.pexels.com/photo/close-up-photo-of-person-typing-on-laptop-1181675/)的图像

我一直使用 cx_Oracle Python 扩展模块来访问我的 Oracle 数据库，因此我能够自动化我的大多数小任务。

Oracle 提供的在线文档可以帮助您直接访问 Oracle 数据库，但在本文中，我将演示如何使用 Cx Oracle 和其他一些 python 库来自动执行从本地计算机到 Oracle 数据库的小型 ETL 任务。

目标是编写一个 **python 脚本**，它读取本地文件夹中的文件，执行数据清理和转换，然后插入到 Oracle 数据库中。然后，可以通过任务调度程序对这个 python 脚本文件进行调度，以便它可以按照设定的时间表自动运行。将添加额外的日志记录代码来跟踪脚本中发生的事情。

**先决条件:**

在您可以在 python 脚本中编写 Cx Oracle 之前，您需要安装 [Oracle 客户端库](https://cx-oracle.readthedocs.io/en/latest/user_guide/installation.html)，当您成功安装这些库后，您就可以开始编写 python 脚本了。为了开始这个脚本，我导入了这些库:

```
import pandas as pd
import cx_Oracle
import math
import logging
```

# 记录器信息块:

logger info 代码块生成一个日志文件，记录您的脚本是否成功运行。如果脚本是自动执行的，这些日志会很有用，这样您就可以得到命令状态的通知。

# 数据加载和清理:

第一个 python 函数从服务器或本地文件夹读取文件，在必要时清理和转换数据，然后返回所需的数据框输出，该输出将用于插入 Oracle 数据库表。代码非常简单，可以添加任何额外的命令来转换数据集。

# Cx Oracle 连接

第二个 python 函数是数据库连接和 Oracle 命令发生的地方。块的第一部分需要连接细节，如用户名、密码、主机名、端口和 SID，以便 python 可以与 Oracle 通信。第二部分是通过简短的 SQL 查询定义命令或任务。我在函数中添加了 3 个查询，Truncate *(清除表但保留表模式)*、Insert *(在表中注入数据)*和 Grant Access *(授予公共访问权限以便用户可以查询您的表)* —如果需要，可以添加更多的查询。

# 脚本运行程序

调用这两个函数的 python 脚本的最后一部分。如果您已经注意到 logger.info 在整个脚本的大多数行中都被调用，这使得跟踪每个命令的状态变得更加容易。

这个 python 脚本文件现在可以用作批处理文件的一部分，可以通过任务调度器上传或通过命令提示符*运行(logger info 也通过 cmd 打印通知)*。

如果你对导入速度感兴趣，我已经在小型到大型数据集上进行了测试。例如，对于 100 万行代码，python 脚本花了大约 12 分钟才成功导入到我的 Oracle 表上。当然，这取决于您本地机器上的其他几个因素。

总的来说，这个简单的 python 脚本和自动化过程将帮助您节省时间来完成占用您一些时间的小任务。

> 【https://www.kathleenlara.com/】网站:
> 
> **推特:**[https://twitter.com/itskathleenlara](https://twitter.com/itskathleenlara)