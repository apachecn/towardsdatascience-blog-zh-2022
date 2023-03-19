# 如何用 Python 找到最佳的 Wordle 首组合词

> 原文：<https://towardsdatascience.com/how-to-find-the-best-wordle-first-combination-words-with-python-ded4b0679a5>

## 搜索最佳的第一和第二单词组合

![](img/ac651220c95203507bd21c123073c35c.png)

尼尔斯·胡内尔弗斯特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

玩 Wordle 时，找到正确的字母通常是由第一个单词决定的。第一个单词越有效，我们就能得到越多的线索来把字母拼对。通常，每个人都有自己的偏好。

对于那些不知道的人来说，wordle 是一个由 Josh Wardle 创建的日常文字游戏，每天都会有一个新的文字难题需要解决。想了解更多，你可以去 https://www.nytimes.com/games/wordle/index.html 的。

在这篇文章中，我将寻找一个解决方案来使用 python 获得最好的第一个单词。这里我只使用基本的统计方法，以便于大家理解。

我将这篇文章分为 4 个部分，即:

1.  搜索第一个单词
2.  寻找第二个词
3.  寻找第一个和第二个单词的组合
4.  寻找最佳组合

# 搜索第一个单词

我使用了一个来自 Donald E. Knuth 的《计算机编程艺术》(TAOCP)<https://www-cs-faculty.stanford.edu/~knuth/taocp.html>**的 数据集。数据集包含由 5 个字母(根据 wordle 上的字母限制)组成的英语单词，总共 5757 个单词。**

> ****引用**:以上数据集来自唐纳德·e·克努特教授。斯坦福大学计算机编程艺术荣誉退休教授。**

**我做的第一件事是导入依赖项并加载数据集文件。**

```
**import pandas as pd
import numpy as np
import math**
```

```
**words = []
with open('sgb-words.txt') as f:
    words = [line.rstrip() for line in f]**
```

**数据集包含以下数据。**

```
**which
there
their
about
would
...
pupal**
```

**第一个处理是通过删除每个单词中的相同字母来完成的。这是必要的，这样我们可以得到一个有 5 个不同字母的单词。方法如下。**

```
**distinct_words = []
for word in words:
    distinct_words.append(list(set(word)))py**
```

**结果会是这样的。**

```
**[['w', 'h', 'i', 'c'],
['t', 'h', 'e', 'r'],
['t', 'h', 'e', 'i', 'r'],
['a', 'b', 'o', 'u', 't'],
['w', 'o', 'u', 'l', 'd'],
...
['p', 'u', 'a', 'l']]**
```

**之后，我们可以得到每个字母的重量。方法很简单，就是把每个字母加起来，结果以字典的形式呈现出来。权重将决定字母出现的频率，字母出现的越频繁，字母的权重越大。**

```
**letter_counter = {}
for word in distinct_words:
    for letter in word:
        if letter in letter_counter:
            letter_counter[letter] += 1
        else:
            letter_counter[letter] = 0py**
```

**结果会是这样的。**

```
**{'h': 790,
 'w': 500,
 'i': 1538,
 'c': 919,
 'e': 2657,
 't': 1461,
 'r': 1798,
 'u': 1067,
 'a': 2180,
 'o': 1682,
 'b': 668,
 'l': 1433,
 'd': 1099,
 's': 2673,
 'f': 501,
 'g': 650,
 'k': 573,
 'n': 1218,
 'y': 867,
 'p': 894,
 'v': 308,
 'm': 793,
 'q': 52,
 'j': 87,
 'x': 137,
 'z': 120}**
```

**如果排序，结果会是这样的。**

```
**>>> {key: val for key, val in sorted(letter_counter.items(), key = lambda x: x[1], reverse = True)}{'s': 2673,
 'e': 2657,
 'a': 2180,
 'r': 1798,
 'o': 1682,
 'i': 1538,
 't': 1461,
 'l': 1433,
 'n': 1218,
 'd': 1099,
 'u': 1067,
 'c': 919,
 'p': 894,
 'y': 867,
 'm': 793,
 'h': 790,
 'b': 668,
 'g': 650,
 'k': 573,
 'f': 501,
 'w': 500,
 'v': 308,
 'x': 137,
 'z': 120,
 'j': 87,
 'q': 52}**
```

**从这些结果可以看出，出现频率最高的 5 个字母是`s`、`e`、`a`、`r`和`o`。**

**此外，这是以百分比表示的结果。**

```
**>>> values = letter_counter.values()
>>> total = sum(values)
>>> percent = [value * 100\. / total for value in values]
>>> for i, letter in enumerate(letter_counter.keys()):
...    print("{}: {}".format(letter, percent[i]))h: 2.962685167822989
w: 1.8751171948246765
i: 5.767860491280705
c: 3.4464654040877556
e: 9.964372773298331
t: 5.479092443277705
r: 6.742921432589537
u: 4.00150009375586
a: 8.17551096943559
o: 6.307894243390212
b: 2.505156572285768
l: 5.374085880367523
d: 4.121507594224639
s: 10.024376523532721
f: 1.878867429214326
g: 2.4376523532720795
k: 2.148884305269079
n: 4.567785486592912
y: 3.2514532158259892
p: 3.3527095443465216
v: 1.1550721920120008
m: 2.973935870991937
q: 0.19501218826176636
j: 0.3262703918994937
x: 0.5137821113819614**
```

**接下来，我们只需要找到字母权重最高的单词。方法如下。**

```
**word_values = []
for word in distinct_words:
    temp_value = 0
    for letter in word:
        temp_value += letter_counter[letter]
    word_values.append(temp_value)
words[np.argmax(word_values)]**
```

**而结果是`arose`。如果从上面的数据来看，可以看到单词`arose` 的字母具有很高的权重。因此，根据统计结果，我们可以得出结论，单词`arose` 是用在单词的第一个单词中的最佳单词。**

> **但是仅仅第一个词就够了吗？**

**有时候我们需要多一个字才能得到足够的线索。因此，我们将搜索另一个词。**

# **寻找第二个词**

**在我们得到第一个单词后，下一步是得到第一个单词中不包含字母的单词列表。比如我们得到的第一个词是`arose`。所以数据集中的单词列表不能包含字母`a`、`r`、`o`、`s`和`e`。如果有一个单词包含这些字母，那么这个单词将从列表中删除。方法如下。**

```
**result_word = []
first_word_list = list(set(best_word))for word in words:
    in_word = False
    i = 0
    while i < len(first_word_list) and not in_word:
        if first_word_list[i] in word:
            in_word = True
        i += 1
    if not in_word:
        result_word.append(word)**
```

**结果如下。**

```
**['which',
'think',
'might',
'until',
...
'biffy']**
```

**字数由之前的 5757 字缩减到 310 字。只剩下 5%左右的单词了。**

**下一步是我们将重复这个过程，就像我们搜索第一个单词一样。完整的代码如下。**

```
**import pandas as pd
import numpy as np
import mathdef best_words(words):
    distinct_words = []
    for word in words:
        distinct_words.append(list(set(word)))
    letter_counter = {}
    for word in distinct_words:
        for letter in word:
            if letter in letter_counter:
                letter_counter[letter] += 1
            else:
                letter_counter[letter] = 0
    word_values = []
    for word in distinct_words:
        temp_value = 0
        for letter in word:
            temp_value += letter_counter[letter]
        word_values.append(temp_value)
    return word_valuesdef get_best_word(words, word_values):
    return words[np.argmax(word_values)]def remove_word_contain_letters(words, first_word):
    result_word = []
    first_word_list = list(set(first_word))

    for word in words:
        in_word = False
        i = 0
        while i < len(first_word_list) and not in_word:
            if first_word_list[i] in word:
                in_word = True
            i += 1
        if not in_word:
            result_word.append(word)
    return result_wordwords = []
with open('sgb-words.txt') as f:
    words = [line.rstrip() for line in f]word_values = best_words(words)
first_word = get_best_word(words, word_values)
second_words = remove_word_contain_letters(words, first_word)
second_values = best_words(second_words)
second_word = get_best_word(second_words, second_values)print(first_word)  # first word
print(second_word)  # second word**
```

**第一个和第二个单词的结果是`arose`和`unity`。**

**从上面的方法可以得出结论，`arose`和`unity`是开始 Wordle 游戏的最佳单词。但如果我们看一下上一篇帖子的字母数量统计，可以看出`u`和`y`这两个字母并不在使用最多的前 10 个字母中。这表明，`arose`和`unity`这两个词可能不是最合适的词。**

# **寻找第一个和第二个单词的组合**

**在这一节中，我们将讨论，以便我们可以得到两个单词，它们的字母都是出现频率最高的字母。**

**我们只需要重复我们以前做过的过程。如果在之前的过程中我们只使用了具有最佳值的第一个单词，那么现在我们也使用第二好的单词作为第一个单词，这样我们可以得到更多的结果变化。**

**步骤如下。**

**第一种方法是计算所有单词的值，然后根据值对单词进行排序。**

```
**values = best_words(words)
values_index = np.argsort(values)[::-1]**
```

**之后，我们将像以前一样搜索第一个和第二个单词。不同的是，这里我们将继续循环查找第一个和第二个单词的组合，以便产生具有最佳值的单词。**

```
**best_val = 0
best_word_list = []
top_words = sorted(values, reverse=True)for i, idx in enumerate(values_index):
    best_word = words[idx]
    second_words = remove_word_contain_letters(words, best_word)
    second_values = best_words(second_words)
    second_best_word = get_best_word(second_words, second_values)
    temp_value = 0
    for letter in second_best_word:
        temp_value += letter_counter[letter]
    if temp_value + top_words[i] >= best_val:
        best_val = temp_value + top_words[i]
        print(best_word, second_best_word, top_words[i] + temp_value)**
```

**结果是这样的。**

```
**arose unity 17141
tears doily 17388
stare doily 17388
tares doily 17388
rates doily 17388
aster doily 17388
tales irony 17507
taels irony 17507
stale irony 17507
least irony 17507
tesla irony 17507
steal irony 17507
slate irony 17507
teals irony 17507
stela irony 17507
store inlay 17507
lores antic 17559
...
laird stone 17739
adorn tiles 17739
radon tiles 17739
tonal rides 17739
talon rides 17739
lined roast 17739
intro leads 17739
nitro leads 17739
nodal tries 17739**
```

**从这些结果来看，第一列是第一个字，第二列是第二个字，第三列是第一个字和第二个字的值之和。**

**如果看上面的结果，`arose`和`unity`这两个词并不是价值最大的词组合。此外，有许多单词组合会获得值 17739，如果您注意单词组合中获得该值的所有字母，则是在数据集中出现最多的十个字母。所以可以得出结论，得到值 17739 的单词组合是我们能得到的最高单词组合。**

> **但是哪个是最佳的单词组合呢？**

**为了得到这个问题的答案，我们需要根据字母的位置知道它们的重量。**

# **寻找最佳组合**

**现在，我们将寻找最佳单词组合，作为 Wordle 游戏中的第一个和第二个单词。接下来我们需要做的是计算每个位置的字母权重。方法如下。**

```
**letter_list =['r', 'o', 'a', 's', 't', 'l', 'i', 'n', 'e', 's']
letter_value = {}for letter in letter_list:
    letter_counter = {}
    for i in range(len(letter_list)//2):
        loc_counter = 0
        for j in range(len(words)):
            if words[j][i] == letter:
                loc_counter += 1
        letter_counter[str(i)] = loc_counter
    letter_value[letter] = letter_counter**
```

**变量`letter_list`由出现次数最多的字母组成。之后，我们将从数据集中的所有单词中统计这些字母在单词的开头出现了多少次，依此类推。**

**`letter_value`的内容如下。**

```
**{'r': {'0': 268, '1': 456, '2': 475, '3': 310, '4': 401},
 'o': {'0': 108, '1': 911, '2': 484, '3': 262, '4': 150},
 'a': {'0': 296, '1': 930, '2': 605, '3': 339, '4': 178},
 's': {'0': 724, '1': 40, '2': 248, '3': 257, '4': 1764},
 't': {'0': 376, '1': 122, '2': 280, '3': 447, '4': 360},
 'l': {'0': 271, '1': 360, '2': 388, '3': 365, '4': 202},
 'i': {'0': 74, '1': 673, '2': 516, '3': 284, '4': 45},
 'n': {'0': 118, '1': 168, '2': 410, '3': 386, '4': 203},
 'e': {'0': 129, '1': 660, '2': 397, '3': 1228, '4': 595}}**
```

**这些结果解释了，例如，字母`r`作为第一个字母出现了 268 次，第二个字母出现了 456 次，等等。所以我们可以得到每个位置的值。**

**接下来，我们将使用`letter_value`来计算我们之前得到的单词组合的权重。方法如下。**

```
**result_list = []
for i in range(len(best_word_list)):
    word_value = 0
    for word in best_word_list[i]:
        for j, letter in enumerate(word):
            if letter in letter_value:
                word_value += letter_value[letter][str(j)]
    result_list.append(word_value)**
```

**这是结果。**

```
**for i in range(len(result_list)):
    print(best_word_list[i], result_list[i])=== result ===['arose', 'unity'] 3219
['tears', 'doily'] 5507
['stare', 'doily'] 4148
['tares', 'doily'] 6565
...
['lined', 'roast'] 4983
['intro', 'leads'] 4282
['nitro', 'leads'] 4831
['nodal', 'tries'] 5910**
```

**为了获得最佳的值组合，我们可以输入下面的语法。**

```
**result_index = np.argsort(result_list)[::-1]
best_word_list[result_index[0]]**
```

**而最好的单词组合是`toned`和`rails`。**

**最后，这是 Wordle 首次使用 Python 系列进行单词搜索的完整代码。**

```
**import pandas as pd
import numpy as np
import math**
```

```
**def best_words(words):
    distinct_words = []
    for word in words:
        distinct_words.append(list(set(word)))
    letter_counter = {}
    for word in distinct_words:
        for letter in word:
            if letter in letter_counter:
                letter_counter[letter] += 1
            else:
                letter_counter[letter] = 0
    word_values = []
    for word in distinct_words:
        temp_value = 0
        for letter in word:
            temp_value += letter_counter[letter]
        word_values.append(temp_value)
    return word_values**
```

```
**def get_best_word(words, word_values):
    return words[np.argmax(word_values)]**
```

```
**def remove_word_contain_letters(words, first_word):
    result_word = []
    first_word_list = list(set(first_word))

    for word in words:
        in_word = False
        i = 0
        while i < len(first_word_list) and not in_word:
            if first_word_list[i] in word:
                in_word = True
            i += 1
        if not in_word:
            result_word.append(word)
    return result_word**
```

```
**words = []
with open('sgb-words.txt') as f:
    words = [line.rstrip() for line in f]

distinct_words = []
for word in words:
    distinct_words.append(list(set(word)))
letter_counter = {}
for word in distinct_words:
    for letter in word:
        if letter in letter_counter:
            letter_counter[letter] += 1
        else:
            letter_counter[letter] = 0**
```

```
**word_values = best_words(words)
first_word = get_best_word(words, word_values)
second_words = remove_word_contain_letters(words, first_word)
second_values = best_words(second_words)
second_word = get_best_word(second_words, second_values)**
```

```
**values = best_words(words)
values_index = np.argsort(values)[::-1]**
```

```
**best_val = 0
best_word_list = []
top_words = sorted(values, reverse=True)**
```

```
**for i, idx in enumerate(values_index):
    best_word = words[idx]
    second_words = remove_word_contain_letters(words, best_word)
    second_values = best_words(second_words)
    second_best_word = get_best_word(second_words, second_values)
    temp_value = 0
    for letter in second_best_word:
        temp_value += letter_counter[letter]
    if temp_value + top_words[i] >= best_val:
        best_val = temp_value + top_words[i]
        best_word_list.append([best_word, second_best_word])

letter_list =['r', 'o', 'a', 's', 't', 'l', 'i', 'n', 'e', 's']
letter_value = {}**
```

```
**for letter in letter_list:
    letter_counter = {}
    for i in range(len(letter_list)//2):
        loc_counter = 0
        for j in range(len(words)):
            if words[j][i] == letter:
                loc_counter += 1
        letter_counter[str(i)] = loc_counter
    letter_value[letter] = letter_counter

result_list = []
for i in range(len(best_word_list)):
    word_value = 0
    for word in best_word_list[i]:
        for j, letter in enumerate(word):
            if letter in letter_value:
                word_value += letter_value[letter][str(j)]
    result_list.append(word_value)**
```

```
**result_index = np.argsort(result_list)[::-1]
print(best_word_list[result_index[0]])**
```

**可以得出结论，单词`toned`和`rails`是开始一个 Wordle 游戏的最佳单词组合。除了单词组合中的字母是在数据集中出现最多的字母之外，这些字母还被放置在具有最高值的位置。**

**答案可能不是完全最优的，因为它只依赖于统计数据，而没有考虑其他因素。如果你有其他方法来获得 Wordle 游戏中的最优单词，请在评论中写下。**