# 用 Python 翻译长 PDF 报告

> 原文：<https://towardsdatascience.com/translate-long-pdf-reports-in-python-eab3be08ceb4>

## **免费自动提取和翻译完整的德国央行报告**

![](img/d51e89e520b77d573cddb9abbadcc034.png)

照片由[米卡·鲍梅斯特](https://unsplash.com/@mbaumi?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

为了工作，我最近不得不翻译许多经合组织国家央行的旧报告。幸运的是，乱码在线翻译已经成为过去，当处理许多长文档时，常见的手动解决方案通常不可行。有许多有用的 Python 包可以帮助完成这项任务，在各种优秀的现有文章[中有介绍。然而，当面对这项任务时，我发现常用的例子往往太*风格化*，并且许多已建立的工具不再被维护*以支持社区构建的后续项目。*](/pdf-text-extraction-in-python-5b6ab9e92dd)

这就是为什么在这篇文章中，我想 **1)** 提供一个 PDF 翻译的真实世界的例子，并 **2)** 提供一个最好的软件包的更新。

# **2 + 1 任务**

因此，我们将一起翻译一份央行报告，就像代码一样，你可以在我的 [Git 库](https://github.com/pcschreiber1/PDF_Extraction-Translation)上找到。首先，我们需要清楚地知道自己想做什么。在我们的例子中，我们需要以某种方式提取 pdf 的内容，翻译它，然后(潜在地)将它转换成人类易于阅读的格式:**提取- >翻译- >编写。**我们分别处理每项任务，最后把它们结合在一起。

## 提取

正如您可能已经体验过的，从 PDF 中检索文本可能相当棘手。原因是 pdf 只存储字符的*位置*，而*不记录*构成*的单词或行*。我们选择的库是新的**pdf plumb**项目，它建立在非常好的 **pdfminer.six** 库(它本身取代了 **PDFMiner** )之上，但是拥有更好的文档和令人兴奋的新特性。我们将在这里使用的一个特性是表格过滤。为了完整性，请注意流行的 **PyPDF2** 包更适合 PDF 合并，而不是文本提取。

```
import pdfplumberpdf = pdfplumber.open(“src/examples/1978-geschaeftsbericht-data.pdf”)
```

我们导入库并打开所需的文档。**pdf plumb**的中心对象是**页面类**，它允许我们单独访问每个页面及其内容。请注意，虽然我们可以简单地一次提取所有文本，但是将 pdf 缩减为一个大字符串会导致我们丢失许多有用的信息。

下面是如何通过应用 **extract_text()** 方法，使用索引来访问单个页面并轻松访问它们的文本。

```
page11 = pdf.pages[11]
page11.extract_text()
>>> 2  schließlich diese Wende in ihrer Politik durch die Heraufsetzung des Diskont- und Lom \nbardsatzes.
```

虽然这看起来已经很好了(作为比较，查看 PDF 的第 12 页)，但我们看到句子被行尾符打断，我们可以预测这会给翻译带来问题。由于段落在句号之后自然会有换行符，我们将利用这一点只保留想要的换行符。

```
def extract(page):
  """Extract PDF text and Delete in-paragraph line breaks.""" # Get text extracted = page.extract_text() # Delete in-paragraph line breaks extracted = extracted.replace(".\n", "**/m" # keep par breaks
                      ).replace(". \n", "**/m" # keep par breaks
                      ).replace("\n", "" # delete in-par breaks     
                      ).replace("**/m", ".\n\n") # restore par break
 return extractedprint(extract(page11)[:500])
>>> 2  schließlich diese Wende in ihrer Politik durch die Heraufsetzung des Diskont- und Lom bardsatzes.
```

好多了！但是看下一页，我们发现文档中的表格有问题。

```
page12 = pdf.pages[12]print(extract(page12)[:500])
>>> 1  3 Zur Entwicklung des Wirtschaftswachstums Jährliche Veränderung in o;o Zum Vergleich: I  Bruttoin-Brutto- I ...
```

**过滤掉表格**

**pdf plumb**包的一个亮点是**滤镜**方法。该库带有查找表格的内置功能，但是将它与**过滤器**结合起来需要一些[独创性](https://github.com/jsvine/pdfplumber/issues/242#issuecomment-668448246)。本质上，**pdf plumb**将每个字符分配到所谓的“盒子”中，其坐标被**过滤器**作为输入。为了简洁起见，我将不解释 **not_within_bboxes** 函数，而是指向最初的 [Git 问题](https://github.com/jsvine/pdfplumber/blob/stable/pdfplumber/table.py#L404)。我们传递已识别的属于表格的字符，并用 **not_within_bboxes** 函数将它们组合起来。重要的是，由于 filter 方法只接受没有参数的函数，我们使用**部分**冻结了 box 参数。这是我们在上面创建的**提取**函数的前一步。

```
from functools import partialdef not_within_bboxes(obj, bboxes):
"""Check if the object is in any of the table's bbox."""
  def obj_in_bbox(_bbox):
    """Find Bboxes of objexts."""
    v_mid = (obj["top"] + obj["bottom"]) / 2
    h_mid = (obj["x0"] + obj["x1"]) / 2
    x0, top, x1, bottom = _bbox return (h_mid >= x0) and (h_mid < x1) and (v_mid >= top) and (v_mid < bottom) return not any(obj_in_bbox(__bbox) for __bbox in bboxes)def extract(page):
  """Extract PDF text, Filter tables and delete in-par breaks."""
  # Filter-out tables
  if page.find_tables() != []: # Get the bounding boxes of the tables on the page.
  bboxes = [table.bbox for table in page.find_tables()]
  bbox_not_within_bboxes = partial(not_within_bboxes, bboxes=bboxes) # Filter-out tables from page
  page = page.filter(bbox_not_within_bboxes) # Extract Text
  extracted = page.extract_text() # Delete in-paragraph line breaks
  extracted = extracted.replace(".\n", "**/m" # keep par breaks
                      ).replace(". \n", "**/m" # keep par breaks
                      ).replace("\n", "" # delete in-par breaks
                      ).replace("**/m", ".\n\n") # restore par break
 return extractedprint(extract(page12)[:500])
>>> 3 des Produktionspotentials anzusehen. Die statistischen Möglichkeiten lassen es nur an näherungsweise zu, ...
```

太棒了。表格被成功地过滤掉了，我们现在可以看到页面以一个被分页符分成两半的句子开始。我们把它留给了提取，但是我鼓励你尝试更多的特性，比如提取页码，改进段落分隔和修复经常出现的错误，比如识别“%”的“0/o”。

## 翻译

AWS 和 DeepL 为高质量的文本翻译提供了两个突出的 API，但是如果我们想要翻译几个长报告，基于字符的定价方案会变得非常昂贵。为了免费翻译，我们使用 Google Api 和一个关键的变通方法，实现了长文本的翻译。

```
from deep_translator import GoogleTranslator
```

由于 GoogleTranslate API 不是由 Google 维护的，所以社区在翻译方面已经多次遇到问题。这就是为什么我们在这里使用 **deep_translator** 包，它充当 API 的一个有用的包装器，使我们能够在翻译引擎之间无缝切换，如果我们希望的话。重要的是，GoogleTranslator 可以自动识别源语言(在我们的例子中是德语)，所以我们只需要指定我们的目标语言:英语。

```
translate = GoogleTranslator(source=’auto’, target=’en’).translate
```

有了这个包装器，翻译变得非常简单，如下例所示。

```
translate("Ich liebe Python programmieren.")
>>> 'I love Python programming.'
```

然而，关键问题是大多数翻译引擎有 5000 字节的上传限制。如果一个作业超过了这个时间，连接就会被终止——例如，这会阻止**第 11 页**的翻译。当然，我们可以单独翻译每个单词/句子，然而，这降低了翻译质量。这就是为什么我们收集低于上传限制的句子并一起翻译。

原来，我在这里找到了这个变通办法。它使用流行的自然语言处理工具 **nltk** 来识别句子。这个包的文档非常棒，我推荐任何感兴趣的人尝试一下。这里，我们将注意力限制在包的**标记器**上。重要的是，tt 一再强调，只有高质量的输入才能带来高质量的翻译输出，因此在这些准备步骤中付出更多努力将会很容易获得回报！

因为这对于第一次使用的用户来说是令人畏惧的，所以我在这里展示了安装相关的 **nltk** 功能的 shell 脚本(在 Windows OS 上)。“popular”子集包括现在将使用的 **nltk.tokenize** 包。

```
# Shell scriptpip install nltk
python -m nltk.downloader popular
```

如下所示， **sent_tokenize** 函数创建了一个句子列表。语言参数默认为英语，这对于大多数欧洲语言来说很好。请查看 nltk 文档，看看你需要的语言是否被支持。

```
from nltk.tokenize import sent_tokenizetext = "I love Python. " * 2
sent_tokenize(text, language = "english")
>>> ['I love Python.', 'I love Python.']
```

现在，我们需要的第二个要素是收集低于上传限制的句子块的算法。一旦我们发现添加另一个句子将超过 5k 字节，我们翻译集合，并从当前句子开始一个新的块。重要的是，如果一个句子本身应该超过 5k 字节(记住，这大约相当于一页)，我们只需丢弃它，并提供一个文本注释。结合 I)翻译客户端的设置，ii)句子标记化，以及 iii)组块式翻译，我们最终得到以下翻译函数。

```
def translate_extracted(Extracted):
  """Wrapper for Google Translate with upload workaround."""
  # Set-up and wrap translation client
  translate = GoogleTranslator(source='auto', target='en').translate # Split input text into a list of sentences
  sentences = sent_tokenize(Extracted) # Initialize containers
  translated_text = ''
  source_text_chunk = '' # collect chuncks of sentences, translate individually
  for sentence in sentences:
    # if chunck + current sentence < limit, add the sentence
    if ((len(sentence.encode('utf-8')) +  len(source_text_chunk.encode('utf-8')) < 5000)):
      source_text_chunk += ' ' + sentence # else translate chunck and start new one with current sentence
    else:
      translated_text += ' ' + translate(source_text_chunk) # if current sentence smaller than 5000 chars, start new chunck
     if (len(sentence.encode('utf-8')) < 5000):
       source_text_chunk = sentence # else, replace sentence with notification message
     else:
       message = "<<Omitted Word longer than 5000bytes>>"
       translated_text += ' ' + translate(message) # Re-set text container to empty
       source_text_chunk = '' # Translate the final chunk of input text, if there is any valid   text left to translate
  if translate(source_text_chunk) != None:
    translated_text += ' ' + translate(source_text_chunk) return translated_text
```

为了看看它是否有效，我们将我们的翻译功能应用于我们之前已经处理过的页面。对于不讲德语的人来说，显然每小时生产率在 1978 年提高了大约 4%。

```
extracted = extract(pdf.pages[12])
translated = translate_extracted(extracted)[:500]print(translated)
>>>3 of the production potential. The statistical possibilities allow only an approximation of the closures that still occur physically due to long-term shrinkage ...
```

## 写作

我们几乎有我们需要的一切。像我一样，你可能需要将提取的文本转换成人类容易阅读的格式。虽然很容易将字符串保存到**。在 Python 中，缺少换行符使得它们不适合长报告。相反，我们将在这里使用 **fpdf2** [库](https://pyfpdf.github.io/fpdf2/index.html)将它们写回 PDF，该库显然继承了不再维护的 **pyfpdf** 包。**

```
from fpdf import FPDF
```

在初始化一个 FPDF 对象后，我们可以为我们翻译的每一页添加一个页面对象，并将它们写在那里。这将帮助我们保持原始文档的结构。需要注意两点:首先，在 **multi_cell** 中，我们将宽度设置为零以获得全宽，并选择高度为 *5* 以获得细线间距。其次，由于预装字体与 Unicode 不兼容，我们将编码改为“ *latin-1* ”。有关下载和使用 Unicode 兼容字体的说明，请参见**FPD F2**网站上的说明。

```
fpdf = FPDF()
fpdf.set_font("Helvetica", size = 7)fpdf.add_page()
fpdf.multi_cell(w=0,h=5,
               txt= translated.encode("latin-1",errors = "replace"
                             ).decode("latin-1")
                )
fpdf.output("output/page12.pdf")
```

现在，就像在提取中一样，显然你可以用 **fpdf2** 做更多的事情，比如添加页码、标题布局等等。然而，对于本文的目的来说，这个最小的设置就足够了。

# **把所有东西绑在一起**

现在，我们将把所有内容汇集到一个管道中。请记住，为了避免丢失太多信息，我们对每个页面进行单独操作。重要的是，我们对翻译做了两处修改:由于一些页面是空的，但是空字符串对于 **GoogleTranslator** 来说不是有效的输入，我们在翻译之前放置了一个 if 条件。其次，因为 **nltk** 将我们的分段符(即“ *\n\n* ”)分配到句子后面的*的开头， **GoogleTranslate** 忽略这些。这就是为什么我们用列表理解法单独翻译每一段。耐心点，翻译 150 页可能需要 7 分钟！*

```
# Open PDF
with pdfplumber.open(“src/examples/1978-geschaeftsbericht-data.pdf”) as pdf:
  # Initialize FPDF file to write on
  fpdf = FPDF()
  fpdf.set_font(“Helvetica”, size = 7) # Treat each page individually
  for page in pdf.pages:
    # Extract Page
    extracted = extract(page) # Translate Page
    if extracted != “”:
      # Translate paragraphs individually to keep breaks
      paragraphs = extracted.split(“\n\n”)
      translated = “\n\n”.join(
        [translate_extracted(paragraph) for paragraph in paragraphs]
        )
    else:
      translated = extracted # Write Page
    fpdf.add_page()
    fpdf.multi_cell(w=0, h=5,
                   txt= translated.encode(“latin-1”,
                                          errors = “replace”
                                 ).decode(“latin-1”)) # Save all FPDF pages
 fpdf.output(“output/trans_1978-geschaeftsbericht-data.pdf.pdf”)
```

# 结论

谢谢你坚持到最后。我希望这篇文章能给你一个关于如何翻译 pdf 和什么是最先进的软件包的实例。在整篇文章中，我指出了这个基本示例的各种潜在扩展(例如，添加页码、布局等。)，所以请分享你的方法——我很想听听。当然，我也总是渴望听到关于如何改进代码的建议。

注意安全，保持联系！