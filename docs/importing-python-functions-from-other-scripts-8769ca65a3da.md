# 从其他脚本导入 Python 函数

> 原文：<https://towardsdatascience.com/importing-python-functions-from-other-scripts-8769ca65a3da>

## 停止将函数复制和粘贴到新脚本中

![](img/9fe40fa11ce2e6b478e8192d373847d6.png)

照片由[马腾·范登·霍维尔](https://unsplash.com/@mvdheuvel?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

有时，一个新项目需要现有的代码。更聪明地工作(而不是更努力地工作)意味着利用现有的工作。目前，从一个脚本复制粘贴到另一个脚本似乎是一个快速、无害的解决方案。然而，在一个文件中定义函数并导入它简化了过程并消除了潜在的错误。

并非所有函数都需要单独的脚本文件。但是，如果您希望在多个项目中重用该函数，请单独保存它。

# 在外部脚本中存储函数的好处

1.  组织
    用描述性名称将相关功能存储在脚本中更容易找到。例如，在`conversions.py`中寻找转换函数比在`lab_report_1.py`、`lab_report_2.py`等中搜索更直观。
2.  版本控制
    在多个地方重新创建一个函数意味着一个脚本中的编辑和更新不会转移到其他脚本。在一个地方定义函数并导入它，允许所有调用该函数的脚本都有最新的定义。
3.  可读性
    将函数存储在外部可以更容易地看到脚本的总体目标。同行评审者不需要看到每个函数的定义，尤其是那些简单的函数。一个描述性的函数名通常就足够了。
4.  简单性
    与复制粘贴方法相比，导入方法使得重用函数更加容易。代码行不会被意外删除。不需要打开额外的脚本。简化的流程消除了可能的错误。

# 条款

为清晰起见，以下是描述`import`语句时常用的一些术语:

*   **库**:NumPy、Pandas 等相关模块的集合。
*   **模块**:带有“.py”扩展名；与脚本互换使用
*   **模块名**:不带“，”的文件名。py "扩展
*   **子模块**:一个带有“的文件。子目录中的 py "扩展名

```
directory
|-- module.py
|-- subdirectory
        |-- sub_module.py
```

# 从目录中的脚本导入特定函数

要从当前工作目录中的脚本导入函数，请添加以下内容:

```
from script_in_cwd.py import specific_function
```

为了给`specific_function`一个不同的名字，在`import`语句中添加`as`。

```
from script_in_cwd.py import specific_function as new_name
```

脚本不能处理同名的两个函数。使用`as`避免导入错误。

# 从脚本导入所有函数和模块

要导入脚本中的所有函数，请使用`*`。

```
from script_in_cwd.py import *
```

这将导入在`script_in_cwd.py`中定义的所有函数。如果`script_in_cwd.py`有`import`语句，那么`*`也会导入那些库、模块和函数。例如，如果`script_in_cwd.py`有`import numpy`，那么上面的语句也会导入`numpy`。**导入对象的名称将被绑定在本地名称空间**中，这意味着脚本将独立识别这些名称。换句话说，导入的对象可以在不引用父模块名称的情况下被调用(`script_in_cwd`)。

导入所有函数的另一种方法是:

```
import script_in_cwd.py
```

像以前一样，这个方法导入所有定义的函数和任何用`import`语句调用的东西。模块名称(`script_in_cwd`)将被本地绑定，但其他导入对象的名称不会被本地绑定。那些对象必须在父模块的名字之后被调用。例如:

```
# To call function after importscript_in_cwd.specific_function()
```

如果两个导入的对象使用相同的名称，使用`import {module}`方法代替`from {module} import *`，因为 **Python 不能导入两个同名的对象**。然而，使用`import {module}`，这些对象的名称被绑定到它们唯一的父模块名称上。

# 从子目录中的脚本导入

要从子目录导入:

```
from subdirectory.submodule import *
```

像以前一样，导入对象的名称被绑定到本地名称空间。

另一种选择是:

```
import subdirectory.submodule
```

同样，导入的对象名称将**而不是**被本地绑定，但是`subdirectory.submodule`将被本地绑定。要调用导入的对象，首先引用父模块的名称(`subdirectory.submodule`)。

# 从目录外的脚本导入

从当前工作目录之外导入需要`sys.path`，这是 Python 搜索的所有目录的列表。要添加新的搜索路径:

```
import sys
sys.path.append('/User/NewDirectory')
```

这会将新路径附加到`sys.path`的末尾。Python 按顺序搜索这些路径。使用`sys.path.insert`强制 Python 更快地搜索路径。例如:

```
import sys
sys.path.insert(1, '/User/NewDirectory')
```

这些附加内容不仅适用于当前脚本。Python 还会在这些路径中搜索未来的项目，除非它们被删除。要删除不需要的搜索路径:

```
import sys
sys.path.remove('/User/NewDirectory')
```

# 结论

感谢您阅读我的文章。如果您喜欢我的内容，*请考虑关注我*。此外，欢迎所有反馈。我总是渴望学习新的或更好的做事方法。请随时留下您的评论或联系我 katyhagerty19@gmail.com。

[](https://medium.com/@katyhagerty19/membership) [## 加入我的介绍链接媒体-凯蒂哈格蒂

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

medium.com](https://medium.com/@katyhagerty19/membership)