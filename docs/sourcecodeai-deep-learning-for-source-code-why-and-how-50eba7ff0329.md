# SourceCodeAI——对源代码的深度学习——为什么和如何

> 原文：<https://towardsdatascience.com/sourcecodeai-deep-learning-for-source-code-why-and-how-50eba7ff0329>

![](img/cabce0ed03de306fe98dcd0499e5da60.png)

照片由来自 [Pexels](https://www.pexels.com/photo/set-of-multicolored-plastic-construction-toys-scattered-on-floor-7444982/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) 的 Enric Cruz López 拍摄

# **又一个 NLP？**

源代码 AI 最近成了热门话题；公司比以往任何时候都更多地在这个方向上花费精力，试图利用它来满足他们的需求。动机很明显；首先，随着人工智能应用从领域专业知识中受益，还有什么比让整个 R&D 组织都成为感兴趣领域的专家更好的呢！第二，鉴于最近 NLP 的许多突破，这种技术似乎可以很容易地应用于存在的每个文本领域(包括源代码)。综上所述，将深度学习 NLP 技术应用于源代码似乎是一个非常容易的事情；只需从众多预先训练好的[语言模型](https://en.wikipedia.org/wiki/Language_model) [变形金刚](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)#:~:text=A%20transformer%20is%20a%20deep,and%20computer%20vision%20(CV).)中选取一个，用于感兴趣的任务。什么会出错？但现实一如既往地更加复杂。距离成功将深度学习应用于源代码还有一英里。主要原因是该领域的许多独特特征，使得这种方法不太相关。让我们更深入地看看源代码领域的一些最重要的挑战。

# **独特的字典**

虽然 Python 的鼓吹者喜欢声称阅读一个好的 Python 程序感觉就像阅读英语 “事实是 Python(和其他任何源代码语言)在构建它的标记上是不同的；有两种主要的标记类型——用户定义的(如变量名、函数名等..)和语言内置(如单词' def '，' len '或字符' = ')。这两者都不会被常见的深度学习 NLP 模型(通常在常规英语语料库上训练)正确识别，从而导致许多“词汇表之外”的场景，已知这些场景会严重影响此类模型的性能。一个解决方案可以是使用专门针对源代码领域的语言模型(例如 [code2vec](https://code2vec.org/) ),然后针对感兴趣的问题进行调整。下一个挑战将是如何使其多语言化。

# **每种语言的独特词典**

而在一般的 NLP 世界中，使用基于英语的模型对于许多类型的应用(如情感分析、摘要等)来说已经足够了..鉴于英语的高度主导地位，在源代码上，我们通常更喜欢使用多种语言模型，来解决相同的问题(如缺陷检测)，在各种语言上，一起解决(不仅是 Python，还同时支持其他语言，如 JavaScript 和 Java)。这是学术界和商界的一个主要区别；虽然发表论文的动机可能是为了证明一个新概念(因此应用于单个源代码语言就足够了)，但在生产领域，我们希望为我们的客户提供一个限制最少的解决方案，使其尽可能支持更广泛的语言。微软的 [CodeBERT](https://arxiv.org/abs/2002.08155) 和 SalesForce 的 [CodeT5](https://arxiv.org/abs/2109.00859) 就是那个方向的例子，刻意训练多语言的语言模型(支持~6 种语言)。这种解决方案的第一个问题是，它们的特定于语言的子模型总是比通用的要好(试着用通用的 CodeT5 模型和 Python 优化的模型总结一个 Python 片段)。另一个更内在的问题是，这种有限的(~6 种语言)支持只是沧海一粟。看看 GitHub 的[语言学家的支持语言列表](https://github.com/github/linguist/blob/master/lib/linguist/languages.yml)就知道这还不够。即使我们乐观地假设这种模型可以无缝地应用于类似的语言(比如 C 和 C++，因为 CodeBert 支持 [Go 语言，而这种语言被认为是非常类似的](https://en.wikipedia.org/wiki/Go_(programming_language)#:~:text=Go%20is%20syntactically%20similar%20to,the%20proper%20name%20is%20Go.)),那么像 Yaml、XML 和 Clojure 这样语法如此不同的语言呢，假设这种转换不成立是公平的。解决方案可以是尝试采用不太通用的语言模型，而是对感兴趣的问题更优化的语言模型。下一个挑战将是如何满足所需的预测上下文范围。

# **稀疏上下文**

与应该从头到尾阅读的常规文本不同，代码更具动态性，更像是用乐高积木搭建的塔，应该根据具体情况进行编译和评估。例如，考虑一个简单的面向对象的程序，它具有一个基本抽象类(person)的[继承](https://en.wikipedia.org/wiki/Inheritance_(object-oriented_programming)#:~:text=In%20object%2Doriented%20programming%2C%20inheritance,inheritance)%2C%20retaining%20similar%20implementation.)结构，该抽象类具有一个接口(打印 person 职位)、一组实现(对于雇员和经理的不同打印)和一个将接口(打印标题)应用于基本(person)对象的输入列表的函数。让我们假设我们想要训练一个模型来总结这样的函数。我们的模型应该如何理解我们刚刚描述的程序？所需的上下文(打印标题实现)很可能在不同的位置——远在那个类上，或者甚至可能在不同的文件或项目上。即使我们会考虑一个巨大的变压器，看到了很多的背景；由于相关的上下文可能在完全不同的地方，本地上下文可能不足以解决我们试图解决的问题(此外，考虑像[封装](https://en.wikipedia.org/wiki/Encapsulation_(computer_programming))和[代码重用](https://devopedia.org/code-reuse)这样的最佳实践，这将增加本地上下文稀疏的可能性)。一个解决方案可能是在应用模型之前尝试编译和合并所有相关的代码片段。下一个挑战将是如何考虑各种代码动态状态。

# **程序状态**

不像文本总是有相同的结果，不管我们怎么读(可能不包括 [DnD](https://en.wikipedia.org/wiki/Dungeons_%26_Dragons) 任务)，在代码中，结果取决于我们提供的特定输入。例如，如果我们想要识别[空指针](https://en.wikipedia.org/wiki/Null_pointer)场景；由于糟糕的编码，它们可以是静态的(总是发生，不管输入如何)，因此应该可以通过读取代码来识别(PS，这就是为什么静态代码分析工具[非常擅长发现这些情况](https://gcn.com/cybersecurity/2009/02/a-breach-of-ethics-too/287887/))，但是它们也可以是动态的，由于糟糕的输入条件和缺乏相关验证，因此应该不太容易通过读取代码来识别。缺少的成分是数据流图；通过了解数据是如何在程序中传播的，模型可以识别代码部分和特定的数据条件何时会出现问题。使用像 Github 的 CodeQL 这样的分析程序数据流的工具可以实现这样的视图。问题是这不是一个简单的任务，而且当考虑多语言需求时(例如 CodeQL[只支持大约 7 种源代码语言](https://docs.microsoft.com/en-us/dotnet/architecture/devops-for-aspnet-developers/actions-codeql))。与此同时，它也是神奇解决方案的秘方。一个典型的权衡问题。

总而言之，源代码领域似乎很有挑战性。同时，我们应该考虑阿拉曼尼斯的“自然性假说”，该假说认为软件是人类交流的一种形式；软件语料库具有与自然语言语料库相似的统计特性；这些特性可以用来构建更好的软件工程工具。 NLP 算法应该能够处理源代码任务。我们有责任确保他们以正确的方式看待这些任务，使他们能够正确处理这些任务。如何将深度学习 NLP 技术成功应用于这些领域？

# **问题理解**

第一个常见的解决方案(通常很重要)是在试图解决问题之前正确理解问题域；我们想达到什么目的？业务的最高要求是什么？有哪些技术限制？如果多语言不是强制性的(例如当专门针对 Android 或 Javascript 应用程序时)，那么通过放松这种限制，开发将变得更加简单，使我们能够使用特定语言的预训练模型，甚至可以自己训练简单的模型。专注于特定的语言可以使用常规的 NLP 方法，有目的地过度适应目标源代码语言。通过正确理解问题领域，我们可以使用简化，这将使得能够使用通用的最佳实践来解决我们的需求。从超级复杂的跨语言问题到更一般的 NLP 任务的过渡。这本身就足以简化开发周期。

# **输入范围**

为了确保我们选择正确的输入，正确理解我们试图解决的领域也很重要。如果我们试图总结一个独立的函数，那么函数体就足够了。在函数调用具有自我解释的命名约定的外部函数的情况下也是如此。如果不是这样，我们可以考虑提供整个类或者至少相关的函数实现作为输入。如果我们的目标更加面向流程，比如识别存在 [SQL 注入](https://en.wikipedia.org/wiki/SQL_injection)风险的区域，那么不仅查看特定的 [ORM](https://en.wikipedia.org/wiki/Object%E2%80%93relational_mapping) 实现(比如与数据库交互的 [Hibernate](https://hibernate.org/orm/) 代码)而且考虑查看通向该路径的代码片段是有意义的。主要的问题是，这种方法将复杂性从我们试图训练的模型转移到支持模块，支持模块负责收集所有相关的范围部分。这意味着我们和这些支持模块一样好。给我们的生态系统增加了多余的需求。

# **数据建模**

眼尖的读者可能已经注意到，我们提出的主要问题是关于数据，而不是模型本身。算法很好。他们得到的数据是主要问题。和以前一样，通过正确理解问题，人们可以选择更适合他们需求的数据表示。[源代码测试](https://en.wikipedia.org/wiki/Abstract_syntax_tree#:~:text=In%20computer%20science%2C%20an%20abstract,construct%20occurring%20in%20the%20text.)通常用于获得更高层次的功能交互可见性。[数据流图](https://en.wikipedia.org/wiki/Data-flow_analysis)通常用于跟踪数据是如何在程序中传播的(这对检测空指针或 SQL 注入等情况很重要)。查看[系统调用](https://en.wikipedia.org/wiki/System_call)或[操作码](https://en.wikipedia.org/wiki/Opcode)而不是普通代码，可以隐式地训练多语言模型(使用可以跨不同语言共享的通用接口)。像 [code2vec](https://code2vec.org/) 这样的代码[嵌入](https://en.wikipedia.org/wiki/Word_embedding)可以实现对代码片段的高级理解。此外，这些嵌入模型中的一些已经是多语言的，同时满足了这种需求。在某些情况下，我们可以选择自己训练一个多语言模型。然后，重要的是要验证我们代表了所有相关的子群体(考虑到[采样代码数据集](/how-to-generate-code-dataset-for-machine-learning-applications-fbc1b888cc84)，特别是依赖 Github 上的[并不简单](/source-code-datasets-deep-dive-into-github-characteristics-a26c622e0794))。幼稚的方法很容易导致严重的隐性群体偏差)。将代码视为纯文本可以处理更简单的任务。它的主要构件之一是决定输入电平；它可以是单词、子单词甚至字符级别。从最具体的语言(单词)到最一般的语言(字符)。 [Spiral](https://github.com/casics/spiral) 是一个开源的例子，它试图通过标准化不同的语言编码风格(像 [Camel Case](https://en.wikipedia.org/wiki/Camel_case) naming)来使基于单词的标记化更加通用。子单词是字符(当模型需要学习识别字典单词时)和单词标记化(当单词已经是输入时)之间的折衷，用于类似的动机；尝试生成一个特定的语料库，同时确保不会有太多的超出词汇表的场景(相关的示例实现有 [BPE 和](https://huggingface.co/docs/transformers/tokenizer_summary))。对于某些情况，我们可以选择只保留与我们的情况更相关的代码输入部分(比如只保留函数名或忽略内置于标记中的源代码)。问题是，为了满足这种需求，它需要更多的理解，这又给我们的生态系统增加了一个多余的、易受影响的成分。

# **前方看什么**

软件世界将采取的最明显的方向是更加语言不可知的源代码模型。这是通过使相关模型更具体地针对问题领域，利用源代码独特的功能(如程序和数据流)，并依靠不太通用的 transformers 实现来实现的，这些实现不将源代码领域视为另一个 NLP，而是以更专业的方式，考虑到特定领域的特征(有趣的是，这类似于将图像和 NLP 深度学习最佳实践应用于音频领域的方式，不是原样，而是通过考虑音频独特的概念，调整这些架构以更好地适应音频领域)。与此同时，对于源代码领域的应用程序来说，速度可能是至关重要的(比如为了成为像 [CI-CD](https://en.wikipedia.org/wiki/CI/CD) 这样的系统的一部分，有资源和速度限制)，我们很可能会看到越来越多的轻量级实现，无论是轻量级转换器还是更通用的 NLP 架构。最后，由于源代码领域需要自己的标签(一般的 NLP 标签不太相关)，并且理解到仅仅依靠 Github 采样是不够的，我们可能会面临越来越多的努力来生成带标签的源代码数据集。激动人心的日子就在前面。