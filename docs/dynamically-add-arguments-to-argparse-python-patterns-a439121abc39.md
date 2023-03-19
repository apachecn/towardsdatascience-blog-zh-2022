# 向 Argparse | Python 模式动态添加参数

> 原文：<https://towardsdatascience.com/dynamically-add-arguments-to-argparse-python-patterns-a439121abc39>

## 如何使用 argparse.ArgumentParser 根据用户输入指定不同的参数。

![](img/66da7a5805c0331ad63c5027a5a6216a.png)

由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [Shubham Dhage](https://unsplash.com/@theshubhamdhage?utm_source=medium&utm_medium=referral) 拍摄的照片

# 介绍

当数据科学家的需求超出 Jupyter 笔记本电脑所能提供的范围时，命令行界面(cli)工具就是他们的面包和黄油。Python 的来自标准库的 *argparse* 是我们在 Python 旅程中遇到的第一个相对容易地构建这种接口的工具。然而，尽管只使用 ArgumentParser 类的三个方法就可以很容易地构建一个带有 argparse 的小型易用 cli，但是当接口增长并变得更加复杂时，就需要进行一些额外的规划。在本文中，我们探索了将 cli 划分为单独的子命令的方法，这些子命令带有可以在运行时加载的参数，以及如何将它们与设计模式结合起来，以便更容易地扩展它。

# 多个子命令

大多数 cli 工具都提供了多个命令，每个命令都有自己独特的一组参数。让我们以 git 为例，下面是您可以运行的所有 git 命令的列表:

```
$ git --help
[...]
These are common Git commands used in various situations:

start a working area (see also: git help tutorial)
   clone     Clone a repository into a new directory
   init      Create an empty Git repository or reinitialize an existing one

work on the current change (see also: git help everyday)
   add       Add file contents to the index
   mv        Move or rename a file, a directory, or a symlink
   restore   Restore working tree files
   rm        Remove files from the working tree and from the index

examine the history and state (see also: git help revisions)
   bisect    Use binary search to find the commit that introduced a bug
   diff      Show changes between commits, commit and working tree, etc
   grep      Print lines matching a pattern
   log       Show commit logs
   show      Show various types of objects
   status    Show the working tree status

grow, mark and tweak your common history
   branch    List, create, or delete branches
   commit    Record changes to the repository
   merge     Join two or more development histories together
   rebase    Reapply commits on top of another base tip
   reset     Reset current HEAD to the specified state
   switch    Switch branches
   tag       Create, list, delete or verify a tag object signed with GPG

collaborate (see also: git help workflows)
   fetch     Download objects and refs from another repository
   pull      Fetch from and integrate with another repository or a local branch
   push      Update remote refs along with associated objects
```

这些是其中一些的论据:

```
$ git add --helpSYNOPSIS*git add* [--verbose | -v] [--dry-run | -n] [--force | -f] [--interactive | -i] [--patch | -p] [--edit | -e] [--[no-]all | --[no-]ignore-removal | [--update | -u]] [--sparse] [--intent-to-add | -N] [--refresh] [--ignore-errors] [--ignore-missing] [-renormalize] [--chmod=(+|-)x] [--pathspec-from-file=<file> [--pathspec-file-nul]] [--] [<pathspec>…​]$ git commit --helpSYNOPSIS*git commit* [-a | --interactive | --patch] [-s] [-v] [-u<mode>] [--amend]
           [--dry-run] [(-c | -C | --squash) <commit> | --fixup [(amend|reword):]<commit>)]
           [-F <file> | -m <msg>] [--reset-author] [--allow-empty]
           [--allow-empty-message] [--no-verify] [-e] [--author=<author>]
           [--date=<date>] [--cleanup=<mode>] [--[no-]status]
           [-i | -o] [--pathspec-from-file=<file> [--pathspec-file-nul]]
           [(--trailer <token>[(=|:)<value>])…​] [-S[<keyid>]]
           [--] [<pathspec>…​]
```

你可以看到它们重叠但不相同。如果您尝试将一个命令与另一个命令的参数一起使用，显然会失败:

```
$ git add --reset-author
error: unknown option `reset-author'
usage: git add [<options>] [--] <pathspec>...
```

假设你正在用 Python 为你的数据科学工作编写一个 cli 工具，也许你正在用深度学习编写一个工具来解决你领域中的问题

[](/pick-your-deep-learning-tool-d01fcfb86845)  

然后你很可能用标准库中的 *argparse* 编写你的 cli，因为它可能不是最好的，但它肯定是你学习的第一个。

构建一个参数解析器来使用子命令非常简单，我们将用一个例子来展示它。我们有一个示例应用程序，命令 *train* 和 *infer，*各有不同的参数，可以是强制的，也可以是可选的。示例代码可在[https://github.com/mattiadg/example-cli](https://github.com/mattiadg/example-cli/blob/main/src/sub_commands.py)获得，为方便起见在此复制:

```
# src/sub_commands.pyimport argparse

from commands import train, infer

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Example v2")
    subparsers = parser.add_subparsers(help="Sub-commands help")

    parser_train = subparsers.add_parser("train", help="Train a model")
    parser_train.add_argument(
        "--model",
        "-m",
        required=True,
        help="name of the deep learning architecture to use",
    )
    parser_train.add_argument(
        "--save_model_path",
        required=True,
        help="Path to the directory where to save the model",
    )
    parser_train.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout value, equal for each value",
    )
    parser_train.add_argument(
        "--batch_size", type=int, help="Batch size during training"
    )
    parser_train.set_defaults(func=train)

    parser_infer = subparsers.add_parser("infer", help="Use a model for inference")
    parser_infer.add_argument(
        "--model_path", required=True, help="Path to the model to use for inference"
    )
    parser_infer.add_argument(
        "--batch_size", type=int, help="Batch size during inference"
    )
    parser_infer.set_defaults(func=infer)

    args = parser.parse_args()
    args.func(args)
```

argparse 中的子参数将完成这项工作。我们首先需要声明我们想要使用子参数:

```
subparsers = parser.add_subparsers(help="Sub-commands help")
```

然后，我们用 add_parser()方法为命令“train”和“infer”声明子参数:

```
parser_train = subparsers.add_parser("train", help="Train a model")
parser_infer = subparsers.add_parser("infer", help="Use a model for inference")
```

然后，子解析器的行为将与任何其他解析器完全一样。让我们测试一下我们的 cli:

```
$ python .\src\sub_commands.py train -m transformer --save_model_path $HOME/my_model_path/
Training model with:
model=transformer
save_model_path=/home/me/my_model_path/
dropout=0.1
batch_size=None
func=<function train at 0x000001F9433493A0>
```

它不接受特定于推断的参数:

```
$ python .\src\correct.py train -m transformer --save_model_path $HOME/my_model_path/ --model_path .
usage: Example v2 [-h] {train,infer} ...
Example v2: error: unrecognized arguments: --model_path .
```

但是—当使用“推断”命令时，模型路径可以正常工作

```
$ python .\src\correct.py infer --model_path $HOME/my_model_path/
Inferring with model with:
model_path=/home/me/my_model_path/
batch_size=None
func=<function infer at 0x000001A1881AD1F0>
```

# 动态参数

现在，我们想要为我们想要训练的模型指定一些参数。在我们的例子中，我们有两个不同的模型，transformer 和 lstm。Lstm 将输出向量的大小和状态向量的大小作为参数。变压器具有子层的大小，并且对于大的前馈子层具有不同的大小。两种模型类型都将层数作为参数。

我没有找到将子解析器附加到命名参数(如 model)的方法，所以我们必须遵循不同的路线。

首先，为了保持解析器代码的可管理性，我们不希望所有的参数都出现在同一个主文件中。我们将把新的参数保存在文件中，在文件中我们使用一个函数来定义各自的模型，该函数将解析器作为输入并添加新的参数。

Lstm 看起来是这样的:

```
# src/models/lstm.pydef add_arguments(parser):
    parser_lstm = parser.add_argument_group("lstm")
    parser_lstm.add_argument("--num_layers", type=int, help="Number of LSTM layers")
    parser_lstm.add_argument("--forward_size", type=int, help="Number of units in the forward propagation")
    parser_lstm.add_argument("--state_size", type=int, help="Number of units for the state vector")
```

以下为变压器

```
# src/models/transformer.pydef add_arguments(parser):
    parser_trafo = parser.add_argument_group("transformer")
    parser_trafo.add_argument("--num_layers", type=int, help="Number of Transformer layers")
    parser_trafo.add_argument("--forward_size", type=int, help="Number of units in the forward propagation")
    parser_trafo.add_argument("--hidden_size", type=int, help="Number of units in the hidden FF layers")
```

现在，我们不能在启动时添加来自不同模型的所有参数，因为它们有冲突的名称(— num_layers)，这会使解析器崩溃。在 duplicate.py 文件中，我们复制了上一节中的代码，并添加了来自两个模型的参数，您可以在[https://github . com/matti adg/example-CLI/blob/main/src/duplicate _ arguments . py](https://github.com/mattiadg/example-cli/blob/main/src/duplicate_arguments.py)找到:

```
# src/duplicate_arguments.pyimport argparse

from commands import train, infer
import models.lstm
import models.transformer

if __name__ == "__main__":
 .
 .
 .models.lstm.add_arguments(parser)
models.transformer.add_arguments(parser)

args = parser.parse_args()
args.func(args)
```

如果我们现在跑

```
$ python .\src\duplicate_arguments.py infer --model_path $HOME/my_model_path/
```

我们得到预期的误差

```
argparse.ArgumentError: argument --num_layers: conflicting option string: --num_layers
```

很明显，我们需要一种方法来动态地向我们的解析器*，*添加参数，也就是说，在运行时根据用户的输入。然而，ArgumentParser 的任何方法都不允许我们直接这样做*，我们需要利用一点小技巧。*

*我们需要解析参数两次，第一次获取模型值，然后加载相应的参数，第二次解析我们可以获取特定于命令的参数。*

*为此，我们需要解析器忽略用户插入的、尚未作为参数添加的参数。幸运的是，parse_known_args()恰恰做到了这一点！*

```
*# src/model_loader.pyimport argparse

from commands import train, infer

# src/model_loader.pyfrom commands import train, infer
from models.loader import load_model_args
.
.
.args_, _ = parser.parse_known_args()

load_model_args(parser_train, args_.model)

args = parser.parse_args()
args.func(args)*
```

*其中 parser_train 是 train 命令的子参数，如前所述，而游戏规则的改变者是这一行:*

```
*args_, _ = parser.parse_known_args()*
```

*它需要两个返回值，因为 parse_known_args 返回一个元组，在第一个位置包含已解析的参数，在第二个位置包含所有其他的参数。这是我们通过打印 parser.parse_known_args()的结果得到的结果:*

```
*$ python .\src\model_loader.py train -m transformer --save_model_path $HOME/my_model_path/ --num_layers 6 --forward_si
ze 512 --hidden_size 2048(Namespace(model='transformer', save_model_path='/home/me/my_model_path/', dropout=0.1, batch_size=None, func=<function train at 0x000001E0125DA3A0>), ['--num_layers', '6', '--forward_size', '512', '--hidden_size', '2048'])*
```

*通过最新的修改，我们最终得到了预期的结果:*

```
*$ python .\src\model_loader.py train -m transformer --save_model_path $HOME/my_model_path/ --num_layers 6 --forward_si
ze 512 --hidden_size 2048Training model with:
model=transformer
save_model_path=C:\Users\matti/my_model_path/
dropout=0.1
batch_size=None
num_layers=6
forward_size=512
hidden_size=2048
func=<function train at 0x000001E0125DA3A0>*
```

*缺少的步骤只有 load_model_args()函数，它为正确的模型加载参数。这是:*

```
*# src/models/loader.pyimport importlib

def load_model_args(parser, model):
    module = importlib.import_module("."+model, "models")
    module.add_arguments(parser)*
```

*这是一个基本的实现，它只接受模型名，在“model”包中导入同名的模块，然后在作为输入接收的 parser_train 上调用其中的 add_arguments 函数(如上所示)。*

*这里缺少的是可用模型的列表。如果给定一个随机的模型名，我们的代码就会失败，因为它找不到相应的模块。另一种方法是预先将所有模型的所有 add_arguments()函数和它们相应的名称添加到注册表中。然后，我们可以在— model 参数中提供这些选项作为“[选择](https://docs.python.org/3/library/argparse.html#the-add-argument-method)”。使用 Register 模式可以很容易地做到这一点，我在上一篇文章中描述过:*

*[](/python-polymorphism-with-class-discovery-28908ac6456f)  

为什么我们要使用这种似乎会增加代码复杂性的模式呢？答案是，这里的代码非常简单，但是当您有几个模型时，它会变得越来越复杂，并且还希望为不同的优化器、数据加载器、搜索算法和代码中需要参数化的任何东西提供特定的参数。然后，代码变得固有地复杂，最好通过为每个类别设置一个单一的入口点(注册表)来隔离更改，并使特定的代码只与通用代码接口。此外，主函数将只与通用代码(load_model_args、load_optimizer_args 等)通信，代码库变得更容易推理。

# 结论

在本文中，我们已经看到了如何使用 Python 标准库中的 argparse 构建越来越复杂的 CLI。我们从简单地使用子参数向程序添加子命令开始，然后我们看到了如何使用 parse_know_args 根据用户的选择加载新的参数。最后，我们讨论了 Register 模式如何帮助我们的解析器实现关注点隔离。

作为最新的评论，如果你对开发 cli 工具感兴趣，考虑使用提供比 argparse 更多特性的库，比如 [cloup](https://github.com/janluke/cloup) 。

# 中等会员

你喜欢我的文章吗？你是否正在考虑申请一个中级会员来无限制地阅读我的文章？

如果您决定通过此链接订阅，您将通过您的订阅支持我，无需为您支付额外费用【https://medium.com/@mattiadigangi/membership*