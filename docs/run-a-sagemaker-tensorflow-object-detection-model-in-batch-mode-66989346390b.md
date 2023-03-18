# 以批处理模式运行 SageMaker TensorFlow 对象检测模型

> 原文：<https://towardsdatascience.com/run-a-sagemaker-tensorflow-object-detection-model-in-batch-mode-66989346390b>

## 如何使用 Sagemaker 批量转换作业处理大型图像

对于一个计算机视觉项目，我需要在一大组图像上应用对象检测模型。这篇博文描述了如何在 Amazon SageMaker 中通过 TensorFlow 对象检测模型 API 使用批量转换作业来实现这一点。

首先，基于一个 AWS 示例笔记本，我将解释如何使用 SageMaker 端点在单个图像上运行模型。对于小图像，这种方法是可行的，但是对于大图像，我们会遇到问题。为了解决这些问题，我改用批量转换作业。最后，我最后说几句话。

![](img/46d5bf76ed1f627f0d2fb676e28bc5d8.png)

用于检测徽标的对象检测。图片由作者提供。

# 起点:使用 SageMaker TensorFLow 对象检测 API 进行模型推断

AWS 在 GitHub 上提供了一些[如何使用 SageMaker 进行物体检测的好例子。我用这个例子通过一个使用 TensorFlow 对象检测 API 的对象检测模型进行预测:](https://github.com/aws-samples/)[https://github . com/AWS-samples/Amazon-sage maker-tensor flow-object-detection-API](https://github.com/aws-samples/amazon-sagemaker-tensorflow-object-detection-api)。

当模型被部署为端点时，您可以通过调用端点，一次一个图像地使用模型进行推理。这段代码摘自示例笔记本，展示了如何定义 TensorFlowModel 并将其部署为模型端点:

来源:[https://github . com/AWS-samples/Amazon-sage maker-tensor flow-object-detection-API/blob/main/3 _ predict/deploy _ endpoint . ipynb](https://github.com/aws-samples/amazon-sagemaker-tensorflow-object-detection-api/blob/main/3_predict/deploy_endpoint.ipynb)。

然后，将图像作为 NumPy 数组加载，并作为列表进行解析，以便将其传递给端点:

来源:[https://github . com/AWS-samples/Amazon-sage maker-tensor flow-object-detection-API/blob/main/3 _ predict/deploy _ endpoint . ipynb](https://github.com/aws-samples/amazon-sagemaker-tensorflow-object-detection-api/blob/main/3_predict/deploy_endpoint.ipynb)。

最后，调用端点:

来源:[https://github . com/AWS-samples/Amazon-sage maker-tensor flow-object-detection-API/blob/main/3 _ predict/deploy _ endpoint . ipynb](https://github.com/aws-samples/amazon-sagemaker-tensorflow-object-detection-api/blob/main/3_predict/deploy_endpoint.ipynb)。

同样，完整的笔记本可以在[这里](https://github.com/aws-samples/amazon-sagemaker-tensorflow-object-detection-api/blob/main/3_predict/deploy_endpoint.ipynb)找到。

# 问题:端点请求负载太大

这在使用小图像时非常好，因为 API 调用的请求负载足够小。但是，当使用较大的图片时，API 会返回 413 错误。这意味着有效负载超过了[允许的大小](https://docs.aws.amazon.com/general/latest/gr/sagemaker.html#limits_sagemaker)，即 6 MB。

当然，我们可以在调用端点之前调整图像的大小，但是我想使用[批处理转换作业](https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#run-a-batch-transform-job)。

# 解决方案:改用批处理转换作业

使用 SageMaker 批处理转换作业，您可以定义自己的最大有效负载大小，这样我们就不会遇到 413 错误。除此之外，这些作业可以用来一次性处理全套图像。

这些图像需要存储在 S3 桶上。所有图像都以批处理模式处理(顾名思义),预测也存储在 S3 上。为了使用批处理转换作业，我们再次定义了一个 TensorFlowModel，但是这次我们还定义了一个`entry_point`和一个`source_dir`:

来源:[https://github . com/AWS-samples/Amazon-sage maker-tensor flow-object-detection-API/blob/main/3 _ predict/deploy _ endpoint . ipynb](https://github.com/aws-samples/amazon-sagemaker-tensorflow-object-detection-api/blob/main/3_predict/deploy_endpoint.ipynb)并由作者改编。

`inference.py`代码转换模型的输入和输出数据，如[文档](https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#how-to-implement-the-pre-and-or-post-processing-handler-s)中所述。这段代码需要将请求负载(图像)更改为 NumPy 数组，解析为 list 对象。从这个例子开始，我修改了代码，让它加载图像并将其转换成 NumPy 数组。`inference.py`中`input_handler`功能的内容变更如下:

来源:[https://sage maker . readthe docs . io/en/stable/frameworks/tensor flow/using _ TF . html # how-to-implementation-the-pre-and-or-post-processing-handler-s](https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/using_tf.html#how-to-implement-the-pre-and-or-post-processing-handler-s)并由作者改编。

注意，我在上面的代码中排除了`output_handler`函数。确保在您的代码中也包含该函数(取自文档)。

该函数需要 Python 包 NumPy 和 Pillow，它们没有安装在运行批处理推理作业的映像上。我们可以创建自己的图像并使用那个图像(在初始化 TensorFlowModel 对象时使用`image_uri`关键字)，或者我们可以提供一个`requirements.txt`并将它存储在与您的笔记本相同的文件夹中(称为`source_dir='.’`)。该文件在映像启动过程中使用，以便使用 pip 安装所需的软件包；内容是:

内容由作者提供。

首先，我想使用 OpenCV(就像端点示例中一样)，但是这个包不太容易安装。

我们现在使用模型来创建一个 transformer 对象，而不是将模型部署为模型端点:

作者代码。

最后，使用输入路径调用转换器:

作者代码。

瞧！图像由模型处理，结果将作为 JSON 文件保存在`output_path`桶中。命名等同于输入文件名，后跟一个`.out`扩展名。顺便说一下，您还可以调整和优化实例类型、最大负载等。

![](img/580bd425b036eb3e6ce8d4b28030946b.png)

布鲁斯·马尔斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片。

# 最后的想法

这很可能不是最划算的方法，因为我们将图像作为 NumPy 数组传递给转换器。从[的另一个示例笔记本](https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker_batch_transform/tensorflow_open-images_jpg/tensorflow-serving-jpg-python-sdk.ipynb)中，我发现建议您的 SavedModel 应该接受二进制数据的 base-64 编码字符串，因为二进制数据的 JSON 表示可能很大。

此外，我们可以调整`inference.py`中的`output_handler`函数来压缩返回并存储在 S3 上的 JSON，或者只返回相关的检测。