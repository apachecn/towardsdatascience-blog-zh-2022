# 如何编写定制的 Keras 模型，以便可以部署它来提供服务

> 原文：<https://towardsdatascience.com/how-to-write-a-custom-keras-model-so-that-it-can-be-deployed-for-serving-7d81ace4a1f8>

## 如何将定制层、建模、丢失、预处理、后处理改编成可服务的 API

如果您编写的唯一 Keras 模型是带有预建层(如 Dense 和 Conv2D)的顺序模型或功能模型，则可以忽略本文。但是在你 ML 生涯的某个时候，你会发现你是在子类化一个层或者一个模型。或者自己编写损失函数，或者在服务过程中需要自定义预处理或后处理。在这一点上，你会发现你不能很容易地使用

```
model.save(EXPORT_PATH)
```

即使您能够成功保存模型，您可能会发现模型无法成功部署到 TensorFlow 服务或包装 TF 服务的 Vertex AI 或 Sagemaker 等受管服务中。即使您成功地部署了它，您可能会发现结果不直观或者完全错误。

不幸的是，解释您必须做什么的文档分散在多个页面中。一些推荐的方法将会工作，如果你做的一切都是正确的，但不会报告错误，一些方法会导致训练中的戏剧性减速，另一些方法在服务中不灵活，一些方法会导致模型保存花费几个小时，并且通常错误消息可能很难理解。

![](img/0b961ac3515bf8558160cdb8f000fd6f.png)

图片由[康格设计](https://pixabay.com/users/congerdesign-509903/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=1976725)来自 [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=1976725)

这篇文章是关于如果你在 Keras 中有一个自定义的东西(层，模型，Lambda，Loss，预处理器，后处理器)你必须做什么。

## 示例模型

为了说明，我将使用 Keras 示例中的命名实体识别 [(NER)模型。基本上，一个被训练来识别名字和位置的 NER 会使用以下形式的句子:](https://keras.io/examples/nlp/ner_transformers/)

约翰去了巴黎

并返回:

*说出位置*

这个模型本身如何工作并不重要。只是它涉及自定义 Keras 层和自定义 Keras 模型(即，它们涉及子类层。层和 keras。型号):

```
class TransformerBlock(layers.Layer):
...class TokenAndPositionEmbedding(layers.Layer):
...class NERModel(keras.Model):
...
```

本文附带的[完整代码在 GitHub](https://github.com/lakshmanok/lakblogs/blob/main/deployable_ner.ipynb) 上。

这种 NLP 模型必须对输入文本进行一些定制的预处理。基本上，输入的句子被拆分成单词，小写，然后转换成基于词汇表的索引。该模型在词汇 id 上被训练:

```
def map_record_to_training_data(record):
    record = tf.strings.split(record, sep="\t") tokens = tf.strings.split(record[0])
    tokens = tf.strings.lower(tokens)
    tokens = vocab_lookup_layer(tokens) tags = tf.strings.split(record[1])
    tags = label_lookup_layer(tags)
    return tokens, tags
```

输入 tf.data 管道是:

```
train_dataset = (
    tf.data.TextLineDataset('train.csv')
    .map(map_record_to_training_data)
    .padded_batch(batch_size)
)
```

并且用定制损失来训练模型:

```
loss = CustomNonPaddingTokenLoss()
ner_model.compile(optimizer="adam", loss=loss)
ner_model.fit(train_dataset, epochs=10)
```

你可以明白我为什么选择这个例子——它有自定义*一切*。

## 出口

当我们试图导出这个模型来部署它时会发生什么？通常，这是它将涉及的内容:

```
model.save(EXPORT_PATH)
```

然后，我们将保存的模型交给 Sagemaker 或 Vertex AI 等服务，它会:

```
model = saved_model.load_model(EXPORT_PATH)
model.predict(...)
```

不幸的是，由于上面所有的自定义层和代码，这种 stratightforward 方法行不通。

当我们这样做时:

```
ner_model.save(EXPORT_PATH)
```

我们得到一个错误:

```
Unknown loss function: CustomNonPaddingTokenLoss. Please ensure this object is passed to the `custom_objects` argument. See https://www.tensorflow.org/guide/keras/save_and_serialize#registering_the_custom_object for details.
```

## 问题 1:未知损失函数

我们如何解决这个问题？我将在本笔记本的后面向您展示如何注册自定义对象。但是首先要意识到的是**我们不需要导出损失**。损失只是训练需要，部署不需要。

所以，我们可以做一件更简单的事情。只是消除损失:

```
# remove the custom loss before saving.
ner_model.compile('adam', loss=None)
ner_model.save(EXPORT_PATH)
```

成功！一句话:**在导出 Keras 模型进行部署之前，消除定制损失。**这是一行程序。

## 问题 2:错误的输入形状

现在，让我们做 TensorFlow Serving(或包装 TensorFlow Serving 的托管服务，如 Sagemaker 或 Keras)所做的事情:调用我们刚刚加载的模型的 predict()方法:

```
sample_input = [
     "Justin Trudeau went to New Delhi India",
     "Vladimir Putin was chased out of Kyiv Ukraine"
]
model.predict(sample_input)
```

不幸的是，我们得到一个错误:

```
Could not find matching concrete function to call loaded from the SavedModel. Got:
        * Tensor("inputs:0", shape=(None,), dtype=**string**)Expected:
        * TensorSpec(shape=(None, None), dtype=tf.**int64**, name='inputs')
```

**捕捉预处理**

这是因为我们试图发送一个完整的句子(一个字符串)，但我们的模型是在一组词汇 id 上训练的。这就是为什么期望的输入是一组 int64。

当我们调用 tf.strings.split()、tf.strings.lower 和 vocab_lookup_layer()时，我们在 tf.data()管道中做到了这一点:

```
def map_record_to_training_data(record):
    record = tf.strings.split(record, sep="\t") tokens = tf.strings.split(record[0])
    tokens = tf.strings.lower(tokens)
    tokens = vocab_lookup_layer(tokens)...
```

在预测过程中，我们也必须重复这种预处理。

怎么会？

嗯，我们可以在顶点 AI 上使用一个[预处理容器](https://cloud.google.com/blog/topics/developers-practitioners/add-preprocessing-functions-tensorflow-models-and-deploy-vertex-ai)或者一些类似的功能。但是这有点违背了我们拥有一个简单的、全方位部署的 Keras 模型的目的。

相反，我们应该重组我们的 tf.data 输入管道。我们想要的是有一个函数(在这里，我称之为 process_descr ),我们可以从 tf.data 管道和我们导出的模型中调用它:

```
def process_descr(descr):
  # split the string on spaces, and make it a rectangular tensor
  tokens = tf.strings.split(tf.strings.lower(descr))
  tokens = vocab_lookup_layer(tokens)
  max_len = MAX_LEN # max([x.shape[0] for x in tokens])
  input_words = tokens.to_tensor(default_value=0, shape=[tf.rank(tokens), max_len])
  return input_words
```

这使得我们的训练代码可以像以前一样工作。当我们准备好保存它时，我们需要用模型中包含的预处理代码创建一个层。

另一种方法是定义一个调用预处理函数的预测签名。然而，这是有问题的，因为如果您的定制模型中有错误，您将不会知道这些错误(问我我是如何知道的)。

**带预处理层的新模型**

一个更简单的方法是，告诉你错误并给你一个机会去修正它们，就像我们对损失函数所做的那样。定义一个新的标准模型，它有一个 lambda 层，在将它提供给定制模型之前进行预处理，并将其写出。

```
temp_model = tf.keras.Sequential([
  tf.keras.Input(shape=[], dtype=tf.string, name='description'),
  tf.keras.layers.Lambda(process_descr),
  ner_model                            
])
temp_model.compile('adam', loss=None)
temp_model.save(EXPORT_PATH)
!ls -l {EXPORT_PATH}
```

不幸的是，前面提到的未报告的错误现在出现了。什么错误？

## 问题 3:未追踪张量

我们得到的错误消息是:

```
Tried to export a function which references 'untracked' resource Tensor
```

这是怎么回事？这里的问题是:当你写一个自定义的 Keras 层或 Keras 损失或 Keras 模型，你是在定义代码。但是，当您导出模型时，您必须用它创建一个平面文件。代码会发生什么变化？丢了！那么预测是如何进行的呢？

你需要告诉 Keras 如何传入所有的构造函数参数等等。然后，Keras 将处理代码，恢复对象，并做正确的事情。

方法是定义一个 getConfig()方法，该方法包含所有的构造函数参数。基本上，自定义图层如下所示:

```
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings
```

必须是这样的:

```
**@tf.keras.utils.register_keras_serializable() # 1**
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, ****kwargs): # 2**
        super(TokenAndPositionEmbedding, self).__init__(****kwargs) # 3**
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )

        **#4 save the constructor parameters for get_config()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim**

        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings

   ** def get_config(self): # 5
        config = super().get_config()
        # save constructor args
        config['maxlen'] = self.maxlen
        config['vocab_size'] = self.vocab_size
        config['embed_dim'] = self.embed_dim
        return config**
```

有 5 个变化:

1.  添加注记以向 Keras 注册自定义图层
2.  将**kwargs 添加到构造函数参数中
3.  向超级构造函数添加一个**kwargs
4.  将构造函数参数保存为实例字段
5.  定义一个保存构造函数参数的 get_config 方法

一旦我们对我们的自定义层和模型类这样做了(查看 GitHub 中的笔记本以获得完整代码)，我们仍然无法保存。错误消息与之前完全相同。我们有一个未追踪的张量。但是我们只是完成了所有的定制课程，做了正确的事情。发生什么事了？

## 问题 Lambda 层中未跟踪的资源

我们还有一个问题。即使我们检查并修复了所有的定制层和模型，仍然有一段用户定义的代码。

我们用于预处理的 Lamda 层！它使用词汇表，并且 vocab_lookup_layer 是一个未被跟踪的资源:

```
def process_descr(descr):
  # split the string on spaces, and make it a rectangular tensor
  tokens = tf.strings.split(tf.strings.lower(descr))
  tokens = **vocab_lookup_layer**(tokens)
  max_len = MAX_LEN # max([x.shape[0] for x in tokens])
  input_words = tokens.to_tensor(default_value=0, shape=[tf.rank(tokens), max_len])
  return input_words
```

仅仅修改函数并不能解决这个问题。

一句话:Lambda 层是危险的，很难意识到我们忘记了什么资源。

我建议你去掉所有的 Lambda 图层，用自定义图层来代替。

让我们这样做吧，确保记住每个自定义层都需要的 5 个步骤:

```
[@tf](http://twitter.com/tf).keras.utils.register_keras_serializable(name='descr')
class PreprocLayer(layers.Layer):
    def __init__(self, vocab_lookup_layer, **kwargs):
        super(PreprocLayer, self).__init__(**kwargs) # save the constructor parameters for get_config() to work properly
       ** self.vocab_lookup_layer = vocab_lookup_layer** def call(self, descr, training=False):
        # split the string on spaces, and make it a rectangular tensor
        tokens = tf.strings.split(tf.strings.lower(descr))
        tokens = **self.vocab_lookup_layer(tokens)**
        max_len = MAX_LEN # max([x.shape[0] for x in tokens])
        input_words = tokens.to_tensor(default_value=0, shape=[tf.rank(tokens), max_len])
        return input_words def get_config(self):
        config = super().get_config()
        # save constructor args
        **config['vocab_lookup_layer'] = self.vocab_lookup_layer**
        return config
```

现在，我们的临时储蓄模型变成了:

```
temp_model = tf.keras.Sequential([
  tf.keras.Input(shape=[], dtype=tf.string, name='description'),
  PreprocLayer(vocab_lookup_layer),
  ner_model                            
])
temp_model.compile('adam', loss=None)
temp_model.save(EXPORT_PATH)
!ls -l {EXPORT_PATH}
```

当然，您应该返回并修复您的 tf.data 输入管道，以使用层而不是 process_descr 函数。幸运的是，这很容易。只需将 process_descr()调用替换为对 PreprocLayer()的调用:

```
PreprocLayer(vocab_lookup_layer)(['Joe Biden visited Paris'])
```

做的事情和:

```
process_descr(['Joe Biden visited Paris'])
```

## 后加工

现在，当我们加载模型并调用 predict 时，我们得到了正确的行为:

```
model = tf.keras.models.load_model(EXPORT_PATH)
sample_input = [
     "Justin Trudeau went to New Delhi India",
     "Vladimir Putin was chased out of Kyiv Ukraine"
]
model.predict(sample_input)
```

不过，这会返回一组概率:

```
array([[[7.6006036e-03, 4.3546227e-03, 9.7820580e-01, 1.3501652e-03,
         5.0268644e-03, 3.4619651e-03],
        [6.8284925e-03, 1.7240658e-02, 9.1373536e-04, 9.6674633e-01,
         5.9596724e-03, 2.3111277e-03],
```

真烦人。我们能对输出进行后处理吗？当然可以。我们必须找到这个数组的 argmax，然后查找对应于这个索引的标签。例如，如果第二项是最大概率，我们将得到 B-NAME 映射[1]。

既然我们在 Keras down 中有了自定义代码，那么应用自定义层方法就很简单了:

```
[@tf](http://twitter.com/tf).keras.utils.register_keras_serializable(name='tagname')
class OutputTagLayer(layers.Layer):
    def __init__(self, mapping, **kwargs):
        super(OutputTagLayer, self).__init__(**kwargs) # save the constructor parameters for get_config() to work properly
        self.mapping = mapping # construct
        self.mapping_lookup = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer( 
                tf.range(start=0, limit=len(mapping.values()), delta=1, dtype=tf.int64),
                tf.constant(list(mapping.values()))),
                default_value='[PAD]') def call(self, descr_tags, training=False):
        prediction = tf.argmax(descr_tags, axis=-1)
        prediction = self.mapping_lookup.lookup(prediction)
        return prediction def get_config(self):
        config = super().get_config()
        # save constructor args
        config['mapping'] = self.mapping
        return config
```

但是那个代码呢？那是什么@#$@R$@#啊？

在 Python 中，我们可以简单地做:

```
mapping[ np.argmax(prob) ]
```

我们必须进行映射。查找:

```
prediction = tf.argmax(descr_tags, axis=-1)
prediction = self.mapping_lookup.lookup(prediction)
```

mapping_lookup 本身就是一种资源，它是一个与 dict:

```
self.mapping_lookup = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer( 
                tf.range(start=0, limit=len(mapping.values()), delta=1, dtype=tf.int64),
                tf.constant(list(mapping.values()))),
                default_value='[PAD]')
```

在撰写本文时，TensorFlow 在存储关键字为整数的 dicts 时有一个 bug，所以我正在入侵对 tf.range()的调用。抱歉。

但是这样做的好处是，您可以简单地采用这个模型并部署它。没有额外的预处理容器、后处理容器等。代码全部运行在 GPU 上(加速了！)而且速度超快。

也很直观。我们请求一个字符串并返回识别出的单词。发送内容:

```
model = tf.keras.models.load_model(EXPORT_PATH)
sample_input = [
     "Justin Trudeau went to New Delhi India",
     "Vladimir Putin was chased out of Kyiv Ukraine"
]
model.predict(sample_input)
```

回馈:

```
array([[b'B-NAME', b'I-NAME', b'OUT', b'OUT', b'B-LOCATION',
        b'I-LOCATION', b'I-LOCATION', b'[PAD]', b'[PAD]', b'[PAD]',
        b'[PAD]', b'[PAD]', b'[PAD]', b'[PAD]', b'[PAD]', b'[PAD]'],
       [b'B-NAME', b'I-NAME', b'OUT', b'OUT', b'OUT', b'OUT',
        b'B-LOCATION', b'I-LOCATION', b'[PAD]', b'[PAD]', b'[PAD]',
        b'[PAD]', b'[PAD]', b'[PAD]', b'[PAD]', b'[PAD]']], dtype=object)
```

即所提供的句子中每个单词的标签。

尽情享受吧！

## 推荐阅读:

1.  [我的完整代码在 GitHub](https://github.com/lakshmanok/lakblogs/blob/main/deployable_ner.ipynb) 上。为了可读性，我省略了本文中的一些细节。请务必查阅笔记本。您可以在 Colab 或任何 Jupyter 环境中运行它。
2.  Keras 示例中的命名实体识别 [(NER)模型](https://keras.io/examples/nlp/ner_transformers/)。很好的说明模型。但是您将无法部署它。
3.  点击此处阅读关于 [Keras 定制图层](https://keras.io/guides/making_new_layers_and_models_via_subclassing/)的所有信息。不要错过关于序列化的“可选”部分。如果您想要部署自定义层，这是强制性的。
4.  一旦你写了你的自定义层，你必须做[自定义对象注册](https://keras.io/guides/serialization_and_saving/)。但是注册定制对象很容易出错。您可能会忘记一些，错误消息不会告诉您错过了哪一个。最好使用[标注快捷方式](https://keras.io/api/utils/serialization_utils/#registerkerasserializable-function)。希望你看到了——这是那一页上讨论的第三个选项。
5.  在这里阅读关于λ层的内容。没有任何关于导出带有 Lambda 层的模型的陷阱，这些 Lambda 层中有全局对象。希望您阅读了前面关于序列化模型的章节，并意识到它们同样适用于 Lambda 层包装的函数！

抱歉，打扰了。