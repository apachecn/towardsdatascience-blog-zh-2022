# 云基础知识:与 AWS 交互

> 原文：<https://towardsdatascience.com/cloud-basics-interacting-with-aws-da179d3f5829>

## 关于使用各种接口创建云资源的教程

![](img/ea0184325e5f0791e0fd7c201b850301.png)

在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上 [engin akyurt](https://unsplash.com/@enginakyurt?utm_source=medium&utm_medium=referral) 拍摄的照片

有基于编程和 GUI/web 的方式与云提供商进行交互，以创建和管理云中的资源。在本实验教程中，我们将通过以下四种方式与 Amazon Web Services(AWS)API 进行交互，以创建 AWS EC2 实例(虚拟化服务器)和 S3(云存储)。

1.  AWS 管理控制台(浏览器)
2.  AWS CLI(命令行界面)
3.  交互式开发环境
4.  自动气象站云形成

通过控制台进行更改对于实验来说很好，但除此之外并不可取。它也不支持自动化和版本控制。CloudFormation 之类的基础设施代码解决方案应该用于实验之外的任何事情。

![](img/0aa16de7e657fdc4f01fe4450fdd74e1.png)

表 1:比较不同的云 API 交互方法

# 先决条件

**AWS 设置**

1.  AWS 设置
    -设置自由层 AWS 帐户( [*链接)*](https://aws.amazon.com/free/)
2.  使用 root 用户登录 AWS 控制台，并创建 IAM 用户。
    -按照[链接](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html)中的*创建 IAM 用户(控制台)*下的步骤操作。为两个*程序性* & *AWS 管理控制台访问*
    创建凭证——在第 6 步(*权限*，选择“*直接附加现有策略*”&添加“*管理员访问*”，并完成其余步骤
3.  为 EC2
    创建密钥对——确保您位于 AWS 控制台
    中的 *us-east-1* (北弗吉尼亚)地区——使用[链接](https://docs.aws.amazon.com/AWSEC2/latest/WindowsGuide/ec2-key-pairs.html)中的“*创建密钥对”下的说明创建 *ec2 密钥对*。*
4.  安装 AWS CLI — [链接](https://aws.amazon.com/cli/)
    —使用“ *aws configure* ”命令配置 aws-cli。在这里使用上面创建的 IAM 凭据。参见[链接](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)寻求帮助。下面的例子:

```
$ aws configure
```

```
AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
AWS Secret Access Key [None]: wJalrXUtnFEMMPLEKEY
Default region name [None]: us-east-1
Default output format [None]: json
```

5.这将存储在 *~/中。AWS/凭证*文件为:

```
[default]
aws_access_key_id = AKIAIOSFODNN7EXAMPLE
aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
region = us-east-1
format = json
```

6.使用以下命令测试 AWS CLI 凭据和配置。

```
aws sts get-caller-identity 
```

如果对您的 AWS 帐户的身份验证成功，不管您的授权级别是什么，即 IAM 权限，它都应该返回以下输出。

```
{
    "Account": "123456", 
    "Arn": "arn:aws:sts::123456:assumed-role/role-name/role-session-name",
    "UserId": "AR#####:#####"
} 
```

**GitHub 账户设置&安装**

1.  创建免费 GitHub.com 账户— [加入 GitHub](http://github.com/signup?)
2.  从[这里](https://git-scm.com/downloads)下载并安装 *git* 到你的电脑上
3.  在您的终端/shell 中使用以下命令配置您的用户名和电子邮件。

```
$ git config — global user.name "Your name here" 
$ git config — global user.email "your_email@example.com"
```

4.在您的终端/shell 中使用以下命令在您的计算机上创建 SSH 密钥。只需接受默认值并一直按 enter 键，直到生成了 2 个文件(私钥和公钥)(如果您愿意的话)。

```
$ ssh-keygen -t rsa -C "your_email@example.com"
```

5.上面的步骤将在/中创建 2 个文件作为 *~/。ssh/id_rsa & ~/。ssh/id_rsa.pub* 。

6.将您的公共 ssh 公钥添加到您的 github 帐户。
-转到您的 github [*帐户设置*](https://github.com/settings/profile)
-单击“ [*SSH 密钥*](https://github.com/settings/ssh)”
-单击“*添加 SSH 密钥*”
-给出一个标签(例如，“我的笔记本电脑”)并将公钥粘贴到文本框
-通过键入以下内容来测试与您的 github.com 帐户的连接:

```
$ ssh -T git@github.com
```

> 它应该是这样的:

```
Hi <your username>! You've successfully authenticated, but GitHub does not provide shell access.
```

**Vistual Studio 代码和 AWS 工具包**

1.  安装 IDE(如 Visual Studio 代码)[https://code.visualstudio.com/](https://code.visualstudio.com/)
2.  为 VS 代码安装 AWS 工具包—[https://aws.amazon.com/visualstudiocode/](https://aws.amazon.com/visualstudiocode/)

# AWS 管理控制台

这是 AWS 的 web 界面。

1.  转到[https://aws.amazon.com/console/](https://aws.amazon.com/console/)
2.  在 AWS 控制台中，确保您位于 us-east-1(北弗吉尼亚)地区
3.  从控制台启动 Windows 实例
4.  遵循[https://docs . AWS . Amazon . com/AWS C2/latest/windows guide/EC2 _ get started . html](https://docs.aws.amazon.com/AWSEC2/latest/WindowsGuide/EC2_GetStarted.html)中的步骤
5.  使用实例的公共 IP 连接(RDP)到您刚刚创建的 Windows 实例。
6.  确保终止 EC2 实例，以避免充电

![](img/0b982104d54cc62f043242501fe17ed5.png)

图 1:显示运行 ec2 实例细节的屏幕截图

# AWS CLI

1.  使用以下命令从命令行界面启动 EC2 Linux 实例。用您的 ec2 密钥对名称替换<key-pair-name>。</key-pair-name>

```
aws ec2 run-instances --image-id ami-08e4e35cccc6189f4 --count 1 --instance-type t2.micro --key-name <Key-Pair-Name> --security-groups default
```

> 暂时向所有 IP 开放实例的安全组上的 SSH 端口(请注意，这不是一个好的安全做法)。请参见[https://docs . AWS . Amazon . com/AWS C2/latest/user guide/authoring-access-to-an-instance . html](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/authorizing-access-to-an-instance.html)的“向 Linux 实例添加入站 SSH 流量规则”一节

2.从控制台使用“ *EC2 实例连接*连接到实例。

3.确保终止 EC2 实例，以避免充电

![](img/4f6f59c0a8e305fa9613018b8cba85ec.png)

图 2:显示创建 ec2 实例的 CLI 命令后终端输出的屏幕截图

# 交互式开发环境(IDE)

AWS Toolkit for Visual Studio 包括 AWS SDK。既然您在 VS 代码上安装了工具包，我们将在 VS 代码中使用 AWS 扩展来创建 AWS S3。

1.  打开 Visual Studio 代码
2.  打开 AWS 浏览器
3.  右键单击 S3 并创建一个桶
4.  从控制台验证铲斗

![](img/64d2a2fb2cce975524e94bf56b297da8.png)

图 3:显示 VS 代码 UI 中 S3 创建链接的屏幕截图

# 自动气象站云形成

CloudFormation 是 AWS 的 IaC(基础设施即代码)服务，它允许我们建模、创建和管理 AWS 和第三方资源。

1.  从云形成模板启动 EC2
2.  去 https://github.com/shrestha-ajay/cloud-basics1.git
3.  遵循"**自述文件中的步骤。MD** 文件中的回购或遵循以下步骤
4.  通过点击 GitHub UI 中的“Fork”来分叉这个库
5.  通过运行以下命令，将此存储库克隆到您的本地计算机上:

```
$ git clone git@github.com:YOUR-USERNAME/YOUR-REPOSITORY.git
$ cd YOUR-REPOSITORY-Folder
```

6.在本地计算机中配置 AWS CLI

7.运行以下命令，在您的 AWS 帐户中创建 ec2 实例和安全组

```
aws cloudformation deploy \
  --stack-name my-cloudbasic-ec2 \
  --template-file ec2_securitygroup.template \
  --parameter-overrides KeyName=General-key-1
```

> 注意:用您的 EC2 密钥对名称替换“General-key-1”密钥名称

8.确保终止 EC2 实例，以避免充电

![](img/ac8b200d06349333e62e8f687c8db797.png)

图 4:显示带有 EC2 实例的 CloudFormation 堆栈的屏幕截图

# **总结**

在本实验中，我们使用四种方式与 AWS 交互来创建云资源(ec2 和 s3):控制台、CLI、IDE 和 CloudFormation。存在一种方法可能优于另一种方法的用例，所以熟悉每种方法是有好处的。

## 打扫

温和地提醒终止资源，以避免被收取费用。

# 资源

1.  [https://aws.amazon.com/free](https://aws.amazon.com/free)
2.  [https://docs . AWS . Amazon . com/IAM/latest/user guide/id _ users _ create . html](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html)
3.  [https://docs . AWS . Amazon . com/AWS C2/latest/windows guide/ec2-key-pairs . html](https://docs.aws.amazon.com/AWSEC2/latest/WindowsGuide/ec2-key-pairs.html)
4.  [https://AWS . Amazon . com/CLI/](https://aws.amazon.com/cli/)
5.  [https://docs . AWS . Amazon . com/CLI/latest/user guide/CLI-configure-files . html](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)
6.  [http://github.com/signup](http://github.com/signup) ？
7.  [https://git-SCM . com/downloads](https://git-scm.com/downloads)
8.  [https://AWS . Amazon . com/SDK-for-python/](https://aws.amazon.com/sdk-for-python/)
9.  [https://code . visualstudio . com/](https://code.visualstudio.com/)
10.  [https://AWS . Amazon . com/visualstudio code/](https://aws.amazon.com/visualstudiocode/)
11.  [https://AWS . Amazon . com/console/](https://aws.amazon.com/console/)
12.  [https://docs . AWS . Amazon . com/awsec 2/latest/windows guide/ec2 _ getstarted . html](https://docs.aws.amazon.com/AWSEC2/latest/WindowsGuide/EC2_GetStarted.html)
13.  [https://github . com/sh restha-ajay/cloud-basics 1 . git](https://github.com/shrestha-ajay/cloud-basics1.git)