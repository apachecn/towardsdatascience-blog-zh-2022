# 清理文本数据入门

> 原文：<https://towardsdatascience.com/primer-to-cleaning-text-data-7e856d6e5791>

## 清洗文本是自然语言处理预处理的一个重要部分

![](img/79002ef8101df35bc463f491fe2531c6.png)

来自[像素](https://www.pexels.com/ko-kr/photo/macbook-pro-1181373/)的免费使用照片

# 介绍

在自然语言处理(NLP)领域，预处理是一个重要的阶段，在这里进行文本清理、词干提取、词汇化和词性标注等工作。在 NLP 预处理的这些不同方面中，我将涵盖我们可以应用的文本清理方法的综合列表。这里的文本清理指的是移除或转换文本的某些部分，以便文本变得更容易被正在学习文本的 NLP 模型理解的过程。这通常通过减少文本数据中的噪声使 NLP 模型表现得更好。

# 将所有字符转换成小写

string 包(Python 中的默认包)包含各种有用的字符串函数。lower 函数就是其中之一，把所有字符都变成小写。

```
**def** make_lowercase(token_list):
    # Assuming word [tokenization](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html) already happened # Using list comprehension --> loop through every word/token, make it into lower case and add it to a new list
    words = [word.lower() for word in token_list]    # join lowercase tokens into one string
    cleaned_string = " ".join(words) 
    return cleaned_string
```

# 删除标点符号

Python 中的 string.punctuation(就是前面提到的包)包含以下几项标点符号。

```
#$%&\’()*+,-./:;?@[\\]^_{|}~`**import** stringtext = "It was a great night! Shout out to @Amy Lee for organizing wonderful event (a.k.a. on fire)."PUNCT_TO_REMOVE = string.punctuationans = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))ans
>> "It was a great night Shout out to Amy Lee for organizing wonderful event aka on fire"
```

字符串包中的另一个方法 translate 函数使用输入字典来执行映射。maketrans 函数是 translate 函数的兄弟方法，它创建用作 translate 方法输入的字典。请注意，maketrans 函数接受 3 个参数，如果总共传递了 3 个参数，则第三个参数中的每个字符都被映射为 None。这个特性可以用来删除字符串中的字符。

根据上面的代码片段，我们将 maketrans 函数的第一个和第二个参数指定为空字符串(因为我们不需要这些参数)，并将第三个参数指定为上面 string.punctuation 中定义的标点项。然后，存储在变量*文本*中的字符串中的标点符号将被删除。

# 删除号码

```
text = "My cell phone number is 123456\. Please take note."text_cleaned = ''.join([i for i in text if not i.isdigit()])text_cleaned
>> "My cell phone number is. Please take note."
```

你也可以使用正则表达式做同样的事情，它是字符串操作最好的朋友之一。

```
text_cleaned = [re.sub(r’\w*\d\w*’, ‘’, w) for w in text]text_cleaned
>> "My cell phone number is. Please take note."
```

# **移除表情符号**

随着各种社交媒体平台生成的非结构化文本数据的数量不断增加，更多的文本数据包含非典型字符，如表情符号。表情符号可能很难被机器理解，并且可能会给你的 NLP 模型添加不必要的噪声。从文本数据中删除表情符号就是这种情况。然而，如果你试图进行情感分析，尝试将表情符号转换成某种文本格式而不是彻底删除它们可能是有益的，因为表情符号可以包含与手头文本相关的情感的有用信息。一种方法是创建您自己的自定义词典，将不同的表情符号映射到一些表示与表情符号相同情感的文本(例如{🔥:火})。

看看这篇[帖子](https://www.analyticsvidhya.com/blog/2022/01/text-cleaning-methods-in-nlp/)，它展示了如何从你的文本中删除表情符号。

```
**import** re**def** remove_emoji(string): emoji_pattern = re.compile(“[“ u”U0001F600-U0001F64F” # emoticons u”U0001F300-U0001F5FF” # symbols & pictographs u”U0001F680-U0001F6FF” # transport & map symbols u”U0001F1E0-U0001F1FF” # flags (iOS) u”U00002702-U000027B0" u”U000024C2-U0001F251" “]+”, flags=re.UNICODE) return emoji_pattern.sub(r’’, string)remove_emoji(“game is on 🔥🔥”)>> 'game is on '
```

# **拼出宫缩**

python 中的收缩包(需要使用！pip 安装收缩)允许我们拼出收缩。通过在执行标记化时创建更多的标记，拼出缩写可以为文本数据添加更多的信息。例如，在下面的代码片段中，当执行基于空白的单词标记化时，标记“would”不会被视为单独的标记。相反，它是象征“她愿意”的一部分。但是，一旦我们修复了缩写，我们就会看到，在执行单词标记化时，单词“would”作为一个独立的标记存在。这为 NLP 模型添加了更多的令牌以供使用。这可以帮助模型更好地理解文本的含义，从而提高各种 NLP 任务的准确性。

```
**import** contractionstext = “She**'d** like to hang out with you sometime!”contractions.fix(text)>> “She **would** like to hang out with you sometime!”
```

但是由于这个包可能不是 100%全面的(即没有覆盖存在的每一个缩写)，您也可以创建自己的自定义字典，将包没有覆盖的某些缩写映射到这些缩写的拼写版本。这篇[帖子](https://studymachinelearning.com/text-data-cleaning-preprocessing/)向你展示了如何做到这一点的例子！

# 剥离 HTML 标签

我们使用 Python 的 BeautifulSoup 包来剥离 HTML 标签。这个包是用于网页抓取的，但是它的 html 解析器工具可以用来剥离 HTML 标签，如下所示！

```
**def** strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    **return** stripped_text # Below is another variation for doing the same thing**def** clean_html(html):     
     # parse html content
     soup = BeautifulSoup(html, "html.parser") for data in soup(['style', 'script', 'code', 'a']):
     # Remove tags
         data.decompose( ) # return data by retrieving the tag content
     return ' '.join(soup.stripped_strings)
```

# 删除重音字符

```
**import** unicodedata**def** remove_accent_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    **return** text
```

# **删除 URL、提及(@)、hastags (#)和特殊字符**

我们可以利用正则表达式来删除 URL、提及、标签和特殊字符，因为它们保持一定的结构和模式。下面只是一个例子，说明我们如何匹配字符串中的 URL、提及和标签模式，并删除它们。请记住，应该有多种方法，因为有多种方法可以形成正则表达式来获得相同的输出。

```
**## Remove URLs****import** re**def** remove_url(text):
     return re.sub(r’https?:\S*’, ‘’, text)print(remove_url('The website [https://www.spotify.com/](https://www.google.com/) crashed last night due to high traffic.'))
>> 'The website crashed last night due to high traffic.'**## Remove Mentions (@) and hastags (#)****import** re**def** remove_mentions_and_tags(text):
     text = re.sub(r'@\S*', '', text)
     **return** re.sub(r'#\S*', '', text)print(remove_mentions_and_tags('Thank you @Jay for your contribution to this project! #projectover'))
>> 'Thank you Jay for your contribution to this project! projectover'**## Remove Special Characters****def** remove_spec_chars(text):
     text = re.sub('[^a-zA-z0-9\s]', '' , text)
     **return** text[https://medium.com/mlearning-ai/nlp-a-comprehensive-guide-to-text-cleaning-and-preprocessing-63f364febfc5](https://medium.com/mlearning-ai/nlp-a-comprehensive-guide-to-text-cleaning-and-preprocessing-63f364febfc5)
```

# **删除停止字**

[停用词是一些非常常见的词，在帮助选择文档或为自然语言处理](https://nlp.stanford.edu/IR-book/html/htmledition/dropping-common-terms-stop-words-1.html)建模时可能没有什么价值。通常，当我们对自然语言处理进行预处理时，这些单词可能会从文本数据中被丢弃或删除。这是因为停用词由于出现频率过高，可能不会增加提高 NLP 模型准确性的价值。就像典型的机器学习模型一样，低方差的特征价值较低，因为它们无助于模型基于这些特征区分不同的数据点。这同样适用于 NLP，其中停用词可以被认为是低方差特征。同样，停用词会导致模型过度拟合，这意味着我们开发的模型对于看不见的数据表现不佳，并且缺乏推广到新数据点的能力。

```
# Retrieve stop word list from NLTK
stopword_list = nltk.corpus.stopwords.words(‘english’)stopword_list.remove(‘no’)stopword_list.remove(‘not’)**from** **nltk.tokenize.toktok** **import** ToktokTokenizertokenizer = ToktokTokenizer( )**def** remove_stopwords(text, is_lower_case=**False**): tokens = tokenizer.tokenize(text) tokens = [token.strip( ) **for** token **in** tokens] # List comprehension: loop through every token and strip white space filtered_tokens = [token **for** token **in** tokens **if** token **not** **in** stopword_list] # Keep only the non stop word tokens in the list filtered_text = ' '.join(filtered_tokens) # join all those tokens using a space as a delimiter **return** filtered_text
```

请注意，有另一种方法可以从一个名为 SpaCy 的不同包中检索停用词，这是另一个常用于 NLP 任务的有用包。我们可以这样做:

```
**import** **spacy**en = spacy.load('en_core_web_sm') # load the english language small model of spacystopword_list = en.Defaults.stop_words
```

# 警告和一些结束语

就像任何其他数据科学任务一样，不应该盲目地进行 NLP 的预处理。考虑你的目标是什么。例如，从你搜集的社交媒体文本数据中移除标签和提及符号，你想从中得到什么？是因为这些符号没有给你正在构建的预测某个语料库的情感的 NLP 模型增加多少价值吗？除非您提出这些问题并且能够清楚地回答，否则您不应该临时清理文本。请记住，询问“为什么”在数据科学领域非常重要。

在本文中，在进入 NLP 循环的下一阶段之前，我们查看了清理文本的各种方法的综合列表，如词汇化和如何实现它们的代码片段。

如果你觉得这篇文章有帮助，请考虑通过以下链接注册 medium 来支持我: )

joshnjuny.medium.com

你不仅可以看到我，还可以看到其他作者写的这么多有用和有趣的文章和帖子！

# 关于作者

*数据科学家。加州大学欧文分校信息学专业一年级博士生。*

*密歇根大学刑事司法行政记录系统(CJARS)经济学实验室的前研究领域专家，致力于统计报告生成、自动化数据质量审查、构建数据管道和数据标准化&协调。Spotify 前数据科学实习生。Inc .(纽约市)。*

他喜欢运动、健身、烹饪美味的亚洲食物、看 kdramas 和制作/表演音乐，最重要的是崇拜我们的主耶稣基督。结账他的 [*网站*](http://seungjun-data-science.github.io) *！*