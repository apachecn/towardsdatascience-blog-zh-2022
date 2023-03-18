# 使用 PyTorch 下载和使用 ImageNet 数据集

> 原文：<https://towardsdatascience.com/downloading-and-using-the-imagenet-dataset-with-pytorch-f0908437c4be>

## 使用最流行的研究数据集训练您的影像分类模型

![](img/54162637934b3ff2a4e4def911b8a8d5.png)

由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的[离子场效应晶体管](https://unsplash.com/@ionfet?utm_source=medium&utm_medium=referral)拍照

ImageNet 是计算机视觉研究中最流行的数据集。图像数据集包含在 WordNet 层次结构中找到的所有种类的收集图像。168 GB 的大型数据集包含 130 万张图像，分为 1，000 个类别，具有不同的标签分辨率粒度。例如，它包含飞机和狗的类别，但也包含不同狗品种的类别，这些类别甚至很难对人类进行分类。ImageNet 可用于分类和对象检测任务，并在默认情况下提供训练、验证和测试分割。

你可能听过 ImageNet、ImageNet1k、ImNet、ILSVRC2012、ILSVRC12 等术语。被利用了。它们都引用了为 ILSVRC 2012 竞赛引入的相同数据集。但是，我应该提到，它只是完整 ImageNet 的一个子集，以“ImageNet21k”的名称存在。ImageNet21k 偶尔用于预训练模型。

最初，ImageNet 托管在 www.image-net.org 的[，](http://www.image-net.org,)，然后数据集私有化，网站进入维护阶段，最后再次公开，但现在只能根据请求下载。在过去的几年里，我肯定申请了十几次，但都没有成功。下载 ImageNet 似乎是一次漫长的旅程。

最近，组织者举办了一场基于原始数据集的 Kaggle 挑战赛，增加了用于对象检测的标签。因此，数据集是半公开的:[https://www . ka ggle . com/competitions/imagenet-object-localization-challenge/](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/)

要下载数据集，您需要注册一个 Kaggle 帐户并加入挑战。请注意，这样做意味着您同意遵守[竞赛规则](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/rules)。特别是，您只能将数据集用于非商业研究和教育目的。

然后，安装 Kaggle CLI:

```
pip install kaggle
```

现在您需要设置您的凭证。这一步非常重要，否则，你将无法开始下载。请遵循官方[指南](https://github.com/Kaggle/kaggle-api#api-credentials):

> 要使用 Kaggle API，请在[https://www.kaggle.com](https://www.kaggle.com/)注册一个 Kaggle 账户。然后转到您的用户配置文件(`https://www.kaggle.com/<username>/account`)的“帐户”选项卡，并选择“创建 API 令牌”。这将触发下载`kaggle.json`，一个包含您的 API 凭证的文件。将这个文件放在位置`~/.kaggle/kaggle.json`(在 Windows 上的位置`C:\Users\<Windows-username>\.kaggle\kaggle.json`——你可以用`echo %HOMEPATH%`检查确切的位置，sans drive)。您可以定义一个 shell 环境变量`KAGGLE_CONFIG_DIR`来将这个位置更改为`$KAGGLE_CONFIG_DIR/kaggle.json`(在 Windows 上是`%KAGGLE_CONFIG_DIR%\kaggle.json`)。

完成后，您就可以开始下载了。请注意，此文件非常大(168 GB)，下载将需要几分钟到几天的时间，这取决于您的网络连接。

```
kaggle competitions download -c imagenet-object-localization-challenge
```

下载完成后，你解压文件。对于 Unix，只需使用`unzip`。请注意，这也需要一段时间。

```
unzip imagenet-object-localization-challenge.zip -d <YOUR_FOLDER>
```

我们还需要两个小的辅助文件。您可以独立地重写下面的代码，但是简单地使用这些文件会更快更简单。因此，只需将它们下载到 ImageNet 根文件夹(包含 ILSVRC 文件夹的那个文件夹)中。如果你在 Unix 系统下，你可以使用`wget`:

```
cd <YOUR_FOLDER>
wget [https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json](https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json)
wget [https://gist.githubusercontent.com/paulgavrikov/3af1efe6f3dff63f47d48b91bb1bca6b/raw/00bad6903b5e4f84c7796b982b72e2e617e5fde1/ILSVRC2012_val_labels.json](https://gist.githubusercontent.com/paulgavrikov/3af1efe6f3dff63f47d48b91bb1bca6b/raw/00bad6903b5e4f84c7796b982b72e2e617e5fde1/ILSVRC2012_val_labels.json)
```

我们现在需要做的就是为 PyTorch 编写一个`Dataset`类。我认为实际的代码加载起来很无聊，所以我就不赘述了。

```
import os
from torch.utils.data import Dataset
from PIL import Image
import jsonclass ImageNetKaggle(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target) def __len__(self):
            return len(self.samples) def __getitem__(self, idx):
            x = Image.open(self.samples[idx]).convert("RGB")
            if self.transform:
                x = self.transform(x)
            return x, self.targets[idx]
```

现在，我们可以通过为预训练的 ResNet-50 模型运行验证时期来测试它。

```
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torchvision
from tqdm import tqdmmodel = torchvision.models.resnet50(weights="DEFAULT")
model.eval().cuda()  # Needs CUDA, don't bother on CPUs
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
dataset = ImageNetKaggle(<YOUR_FOLDER>, "val", val_transform)
dataloader = DataLoader(
            dataset,
            batch_size=64, # may need to reduce this depending on your GPU 
            num_workers=8, # may need to reduce this depending on your num of CPUs and RAM
            shuffle=False,
            drop_last=False,
            pin_memory=True
        )
correct = 0
total = 0
with torch.no_grad():
    for x, y in tqdm(dataloader):
        y_pred = model(x.cuda())
        correct += (y_pred.argmax(axis=1) == y.cuda()).sum().item()
        total += len(y)print(correct / total)
```

这应该输出 0.80342，这是模型的精度(80.342%)。

# ImageNet 的替代产品

对于大多数人来说，使用 ImageNet 进行培训仍然过于昂贵。然而，有许多基于 ImageNet 的替代数据集，其分辨率和/或样本和标签数量有所减少。这些数据集可以用于训练，只需花费一小部分成本。一些例子有[图像网](https://github.com/fastai/imagenette)、[微型图像网](https://www.kaggle.com/c/tiny-imagenet)、[图像网 100](https://www.kaggle.com/datasets/ambityga/imagenet100) 和 [CINIC-10](https://github.com/BayesWatch/cinic-10) 。

# 参考

[1]邓，董，李，李，，，“ImageNet:一个大规模的层次图像数据库”， *2009 年 IEEE 计算机视觉与模式识别会议*，2009，第 248–255 页，doi: 10.1109/CVPR.2009.5206848

该数据集是免费的，用于非商业研究和教育目的。

*感谢您阅读这篇文章！如果你喜欢它，请考虑订阅我的更新。如果你有任何问题，欢迎在评论中提出。*