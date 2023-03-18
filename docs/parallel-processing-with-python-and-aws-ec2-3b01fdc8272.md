# 使用 Python 和 AWS EC2 进行并行处理

> 原文：<https://towardsdatascience.com/parallel-processing-with-python-and-aws-ec2-3b01fdc8272>

## 更快地运行您的脚本

![](img/db80885f54569564f519426446cf6407.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上 [Max Duzij](https://unsplash.com/@max_duz?utm_source=medium&utm_medium=referral) 拍摄的照片

# 介绍

在进行数据科学项目时，我经常遇到的一个问题是需要加快脚本的速度。这可能是在探索性数据分析期间，或者可能是在自然语言处理时获得文档相似性。无论如何，在这种情况下，我可怜的计算机要么在我运行脚本几个小时时变慢，要么在我需要再次使用计算机之前花费太长时间。然后等了这么久，我甚至可能需要回去编辑，重新运行剧本！

有几种方法可以解决这个问题，但在本文中，我将首先直接用 Python 介绍我的方法，然后引入 AWS EC2。

# 多重处理

你电脑的 CPU(很可能)有多个内核，每个内核可以执行不同的任务。这意味着当你运行一个 Python 脚本时，你可以继续检查空闲时间和回复邮件，但是我们经常希望尽快得到我们的脚本结果。为了实现这一点，我们可以使用 Python 的多处理库来并行使用我们的所有(或 1 个以上)CPU 内核。

下面的要点给出了一个简单的方法来实现函数`process`的多重处理，该函数为`i_list`中的每个参数`i`运行。

# AWS EC2

然而，我们可能并不总是拥有生成结果的本地计算能力，甚至不想让脚本在本地运行。Amazon EC2 也被称为 Amazon Elastic Compute Cloud，它支持创建具有不同计算能力级别的虚拟机，这对于将我们的本地脚本运行扩展到云是完美的。

1.  创建一个虚拟机，并按照[标准步骤连接](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html)。

```
1\. Open an SSH client.
2\. Locate your private key file. The key used to launch this instance is temp-cluster.pem
3\. Run this command, if necessary, to ensure your key is not publicly viewable.
	chmod 400 temp-cluster.pem
4\. Connect to your instance using its Public DNS:
	ec2-1234.eu-west-1.compute.amazonaws.com
5\. Example: 
	ssh -i "temp-cluster.pem" ubuntu@ec2-1234.eu-west-1.compute.amazonaws.com
```

2.用 Python 设置 VM，并安装所需的 Python 包。在我的上一个项目中，我做了一些自然语言处理，需要去除停用词。

```
sudo apt update
sudo apt install python3-pippip install nltk
python3 -m nltk.downloader stopwords
pip install numpy pandas sklearn scipy
```

3.要将文件(通常是我们的脚本)移动到 EC2 实例，我们可以使用 secure copy 命令。默认路径是/home/ubuntu。

```
scp -i .ssh/temp-cluster.pem Documents/nlp_script.py ubuntu@ec2-1234.eu-west-1.compute.amazonaws.com:/home/ubuntu
```

4.现在，如果需要，我们可以创建一个文件夹来保存脚本输出，然后运行 Python3 脚本。

```
sudo mkdir script_output_folder
python3 nlp_script.py

# List directory folders
ls -d */
# List folder contents
ls -R
```

5.为了在会话断开或计算机关闭的情况下保持脚本运行，我们可以使用 tmux。

```
# Install tmux
sudo apt-get install tmux# Start a terminal session
tmux new -s mywindow# Restart local computer, ssh to reconnect to the EC2 instance, and 
resume the terminal window
tmux a -t mywindow
```

6.最后，从 EC2 实例下载文件。

```
# To copy the script back to the local "Documents" folder
scp -i .ssh/temp-cluster.pem ubuntu@ec2-1234.eu-west-1.compute.amazonaws.com:/home/ubuntu/nlp_script.py Documents

# Zip a folder for easy download
zip -r squash.zip /home/ubuntu/script_output_folder/

# Use wildcards to download many files
scp -i .ssh/temp-cluster.pem ubuntu@ec2-1234.eu-west-1.compute.amazonaws.com:/home/ubuntu/script_output_folder/* Documents
```

一旦我们所有的处理完成并下载回我们的本地驱动器，EC2 实例就可以删除了。我们现在可以正常访问新文件了。

# 最后的想法

您可能是在试图弄清楚 Python 中的并行处理或者如何使用 Amazon EC2 实例来实现它的时候偶然看到这篇文章的，在这种情况下，我希望这对您有所帮助！