# 将 Jupyter 笔记本作为后台服务运行

> 原文：<https://towardsdatascience.com/run-jupyter-notebook-as-a-background-service-on-ubuntu-c5d6298ed1e>

## 再也不会丢失远程笔记本服务器，甚至可以让笔记本服务器在系统重新启动时启动

![](img/2b7878d008cd92d6f29b15f7ee614747.png)

克里斯托弗·高尔在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

如何在后台运行 Jupyter 笔记本，而不需要为它们打开一个终端？在本文中，我将向您展示两种方法，一种是使用`nohup`的简单的一行程序，另一种是使用守护程序服务的更高级的方法。

在本地运行时，避免为您的笔记本电脑打开终端是非常有用的，但在可能暂时断开连接的远程笔记本服务器上更是如此。如果我们可以将笔记本服务器作为一个长期的后台任务作为系统服务来运行，会怎么样？因此我们可以随时连接到它，它甚至可以在任何故障或机器重启时重新启动。

本教程假设你在基于 unix 的系统上，我在 ubuntu 上，并且已经测试了 18.04 和 20.04 版本。第二种方式需要 root 权限，第一种方式不需要。

# 简单的一行程序

不要使用任何选项来运行`jupyter-notebook ...`,比如指定所使用的端口，而是在您希望笔记本根目录所在的目录中使用以下内容:

```
nohup jupyter-notebook --no-browser --port=8888 &
```

注意命令末尾的`&`字符，这使得它在后台运行。默认情况下，任何输出都将被写入一个名为`nohup.out`的文件中，您可以在那里查看笔记本日志。

像任何其他进程一样，这个进程也有一个 PID，以后可以用它来停止。(通常`jupyter-notebook stop <port>`不会用这种方法。)PID 是在执行上述命令时打印出来的，虽然只是在屏幕上显示，但没有必要记下来，因为以后可以找到它:

您可以检查打开了输出文件的进程的 PID:

```
$ lsof nohup.outCOMMAND     PID  USER   FD   TYPE DEVICE SIZE/OFF     NODE NAME
jupyter-n 44361   username    1w   REG   0,51      884 19846171 nohup.out
```

或者使用`ps`查找计算机所有运行进程之间的进程:

```
$ ps au | grep jupyter
```

最后，使用

```
$ kill -9 <PID>
```

扼杀这一进程。

这相对简单，允许您随时检查该目录中的日志，即使终端会话结束，也能保持笔记本运行。尽管服务器在机器重新启动时也不会重新启动，但是不需要 root 权限。

![](img/a807a082fc6da4161068355c861402d8.png)

Avi Richards 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# 系统上的守护进程

如果您有 root 权限，您可以使笔记本服务器作为系统服务运行，甚至在重新启动时启动。这是针对基于 linux 的系统，使用内置的服务管理器。

## 第 1 步:找到您的`jupyter-notebook`可执行文件的路径

您需要在激活正确的 python 环境后才能这样做。

```
$ which jupyter-notebook/home/username/.local/share/virtualenvs/notebook/bin/jupyter-notebook
```

## (可选)步骤 1.1:设置笔记本密码和任何其他配置

如果您依赖于安全的密码集，而不是复制服务器在启动时生成的令牌，您会发现以后使用该服务会更容易。

```
$ jupyter-notebook password[... password prompt, type it in twice] [NotebookPasswordApp] Wrote hashed password to /home/username/.jupyter/jupyter_notebook_config.json
```

您需要输入两次密码，然后密码以种子格式存储在本地 jupyter config 目录中。

## 步骤 2:编写服务设置文件

系统服务的设置和属性在类似 INI 的文件中指定。这些存储在`/etc/systemd/system/`目录中。

我们来写一个服务文件吧！这是一个相对简单的例子，我们已经为 jupyter 指定了可执行文件和参数，包括启动笔记本目录的目录，该目录与执行服务的目录相同。将用户和组设置更改为您自己的用户，确保它不是 root 用户。

注意`Restart`部分，它定义了如果出现故障，服务应该重启，守护进程等待 60 秒。

## 步骤 3:启用并启动服务

我们希望将该服务设置文件放入正确的系统目录中，然后“启用”并“启动”该服务。Enable 表示该服务在计算机重启时启动，否则需要手动启动。

```
# add the service file
$ sudo cp jupyter-notebook.service /etc/systemd/system/# let the daemon reload all the service files
$ sudo systemctl daemon-reload# (optional) "enable" -> started when the computer boots
$ sudo systemctl enable jupyter-notebook.service# start the service
$ sudo systemctl start jupyter-notebook.service
```

完成后，您可以查看服务状态。请注意，显示状态不需要 root 权限。

```
$ systemctl status jupyter-notebook
```

如果你想停止服务，你可以使用`sudo systemctl stop ...`来禁用启动启动使用`sudo systemctl disable ...`

## 第四步:享受！

现在，您有了一个笔记本守护程序服务，它也可以在重新启动时运行，并且您可以随时访问它。如果服务退出，系统会自动重启，这样你就可以安心工作了。

例如，如果您在一台远程机器上运行您的笔记本，比如您的家庭/工作计算机或云中，您只需要通过 SSH 连接到它，并让笔记本启动和运行。

# 结论

我们已经看到了在后台运行 Jupyter 笔记本服务器的两种方法，一种简单的方法适用于非 root 用户，另一种更高级的方法适用于 root 用户。第一个需要一点监控来启动和停止，而后者应该让你在服务器上一劳永逸。

我在不同的计算机上使用这两种方法，这取决于我拥有的用户权限。例如，第一个在科学计算集群上是最好的，而后者对于 AWS EC2 实例来说是惊人的，该实例用于笔记本和数据分析，可以在任何地方进行。

![](img/45428c3d16e13ef8d2355ea73c6cbfbc.png)

由[乔治·克罗克](https://unsplash.com/@gmk?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

## 参考和有用的链接

*   `man systemd.service`和`man systemd.unit`提供了系统服务文件选项的综合文档，[本](https://www.shellhacks.com/systemd-service-file-example/)是其摘要
*   这个教程很久以前我第一次使用它来建立一个系统服务，后来在多台机器上进行了修改