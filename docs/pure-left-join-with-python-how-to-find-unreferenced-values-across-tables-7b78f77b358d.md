# Python 中 SQL Left Join 的等价及其在数据清理中的应用

> 原文：<https://towardsdatascience.com/pure-left-join-with-python-how-to-find-unreferenced-values-across-tables-7b78f77b358d>

## 使用 Python 识别表中未引用值的技巧

![](img/be921397c2820d6d17dfd81c1187fda6.png)

照片由[菲利普·布特](https://unsplash.com/@flipboo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

如果你曾经使用过关系数据库，你一定知道 [SQL 连接](https://en.wikipedia.org/wiki/Join_(SQL))——它们有很多用例，但是在这篇文章中，我将把重点放在数据清理上。

当执行左、右或全外连接时，您创建的表要么包含所有记录，要么只包含某些表中的记录。对于没有匹配项的行，将放置一个空值。因此，连接对于识别缺失或未引用的值非常有用。

假设您的数据库中有一个包含所有用户的`users`表。此外，您还有其他几个表引用了`users`表的 id，如`posts`、`logins`、`subscriptions`等。您有兴趣找出哪些用户可以从数据库中删除，因为他们没有与您的网站进行有意义的交互。这可以通过检查 id 是否在其他地方被引用来完成。

```
SELECT 
  u.userID AS 'User ID', 
  p.userID AS 'Post table' 
FROM users AS u
  LEFT JOIN posts AS p ON u.userID = p.userID
-- as users can have multiple posts
GROUP BY p.userID;
```

```
User ID      Post table   
---------    ----------  
1            Null   
2            Null   
3            3
4            4
5            Null
```

上表显示用户 1、2、5 没有创建任何帖子。您可能想要进一步研究并添加`logins`和`subscriptions`——这很好，但是如果您有许多额外的表想要以这种方式连接，您可能会遇到一些性能问题(**提示** : *如果您在使用 SQL 脚本，千万不要在您的生产数据库上这样做，首先创建一个本地副本*)。

# 用 Python 分析你的表

如果您遇到性能问题或者需要更好的工具来分析您的数据库，一个想法是求助于 python，因为它有一个非常好的数据生态系统。例如，您可以使用 [SQLAlchemy](https://www.sqlalchemy.org/) 或 Jupyter Notebook 的 SQL magic 函数来获取记录并将它们存储在列表(或字典)中。

为了演示如何用 python 进行左外连接，在这种情况下，我们不打算连接到数据库，而是创建一些随机数据并将其存储在字典中。我们将有一个包含所有可能的用户 id 的`users`表和五个随机引用这些 id 的其他表:

```
import random
import pandas as pd

# defining range for userIDs, one to ten
r = (1, 11)
s, e  = r

# creating dict to hold 'tables' and adding all possible user IDs 
tables = {}
tables['users'] = list(range(*r))

# generating ten tables with random IDs from the initial defined range of userIDs
for i in range(1, 6):
   table = random.sample(range(*r), random.randint(s-1, e-1))
   tables[f'table{i}'] = table
```

# 熊猫

使用 pandas 似乎是显而易见的，因为它是 python 中数据的首选包。它有两个连接表的函数，`pd.merge()`和`pd.join()`(还有`pd.concat()`——注意它的工作方式有点不同)，但是如果你至少有两列，其中一列是你要连接的，另一列包含你的值，这些函数工作得最好。这不是我们的情况，因为我们只有 id 列表。

让我们看看如果我们加入这两个列表会发生什么，`tables['users']`和`tables['table1']`:

```
df_users = pd.DataFrame(tables['users'])
df_table1 = pd.DataFrame(tables['table1'])

pd.merge(df_users, df_table1, how='left')
```

```
OUTPUT:

     0
   ---
 0   1
 1   2
 2   3
 3   4
 4   5
 5   6
 6   7
 7   8
 8   9
 9  10
```

嗯，结果是令人失望的，它似乎没有做任何事情。默认情况下，该函数连接唯一一列上的两个数据帧，因此我们获得所有的用户 id，仅此而已(除了索引之外)。在底层，它确实执行了正确的连接，但是因为我们没有额外的列，所以没有什么可显示的。我们需要添加`indicator=True`参数来查看结果:

```
pd.merge(df_users, df_table1, how='left', indicator=True)
```

```
OUTPUT:

            0      _merge
    ---------   ---------
0           1   left_only
1           2   left_only
2           3        both
3           4        both
4           5   left_only
5           6   left_only
6           7        both
7           8   left_only
8           9   left_only
9          10        both
```

`_merge`列显示记录是存在于两个列表中还是只存在于第一个列表中。通过将初始数据帧的索引设置为唯一的现有列，并连接这些列，我们可以使结果更好:

```
pd.merge(df_users.set_index(0), df_table1.set_index(0), how='left',
  left_index=True, right_index=True, indicator=True)
```

```
OUTPUT:

       _merge
    ---------
0 
1   left_only
2   left_only
3        both
4        both
5   left_only
6   left_only
7        both
8   left_only
9   left_only
10       both
```

虽然这种方法可行，但是如果您想要连接多个列表(表)，它就很笨拙。

# 设置

虽然这不是一个连接，但是使用 Python *集合*(注意，*集合*不能包含重复的值)可以实现预期的结果——识别未引用的值。

```
set_users = set(tables['users'])
set_table1 = set(tables['table1'])

unreferenced_ids = set_users - set_table1
```

通过从另一个集合中减去一个*集合*，可以找到两个*集合*之间的差异——元素出现在`users`中，而不在`table1`集合*集合*中。其余的表也可以重复这个过程。

# 使用循环

对我来说最有效的解决方案是遍历列表(表格)并为未引用的 id 添加`None`值。这是可能的，因为列表是有序的，我们可以遍历所有用户 id 并检查它们是否存在于其他表中。

```
# creating a new dict
final_tables = {}

# transfering user IDs
final_tables['users'] = tables.pop('users')

# looping through the tables
for key, value in tables.items():

    # creating a new column
    column = []

    # checking values against all user IDs
    for user in final_tables['users']:
        # adding True if ID is referenced
        if user in value:
            column.append(True)
        # adding None if ID is not referenced
        else:
            column.append(None)

    final_tables[key] = column

# converting the new dict holding the processed tables to a dataframe
df = pd.DataFrame.from_dict(final_tables).set_index('users')
```

```
OUTPUT:

         table1   table2   table3   table4   table5
        ------   ------   ------   ------   ------
users     
    1     True     True     True     True     True
    2     True     None     Nooe     True     None
    3     None     True     True     None     True
    4     None     None     True     True     True
    5     True     None     None     True     None
    6     True     True     True     None     True
    7     None     None     True     True     True
    8     True     True     None     True     None
    9     True     None     True     None     None
   10     None     None     True     True     None
```

…就这样。该表显示了在 pandas 数据框架中的其他表中如何引用用户 id。

总而言之，如果您习惯于在关系数据库中执行左表连接，并希望在 Python 中实现类似的功能，您有几个选择。有*熊猫、*，但是令人惊讶的是，在两个单独的列上执行连接来查找未引用的值并不简单。或者，您可以使用*集合*来获得两列的唯一值的差异。但是最好的选择可能是使用简单的循环，尤其是当您想要识别几个表中不匹配的记录时。