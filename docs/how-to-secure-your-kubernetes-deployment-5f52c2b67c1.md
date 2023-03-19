# 如何保护您的 Kubernetes 部署

> 原文：<https://towardsdatascience.com/how-to-secure-your-kubernetes-deployment-5f52c2b67c1>

## 建立一个声誉需要 20 年，而几分钟的网络事件就能毁掉它。— **夏羽·纳珀**

![](img/a3411124cc44bb75ad9bce40adc767d3.png)

由[马特·阿特兹](https://unsplash.com/@mattartz?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

不幸的是，Kubernetes 部署在默认情况下并不安全。一个臭名昭著的例子是针对特斯拉汽车公司发起的密码劫持攻击。几年前，恶意攻击者渗透进他们的 Kubernetes 集群，并从其中一个 pod 内部开始了一个加密挖掘过程。

每个 Kubernetes 分布是不同的；因此，这里没有免费的午餐。因此，为了确保 Kubernetes 的安全，您应该采取额外的措施，遗憾的是，大多数措施都是手动过程。那么，你怎么知道从哪里开始呢？

在这个故事中，我们将介绍一款围棋检测工具 [kube-bench](https://github.com/aquasecurity/kube-bench) ，它将帮助你做到这一点。然后，我们将在[Rancher Kubernetes Engine](https://rancher.com/docs/rke/latest/en/)(RKE)中运行 kube-bench，这是一个 CNCF 认证的 Kubernetes 发行版，完全在 Docker 容器中运行。

> [学习率](https://www.dimpo.me/newsletter?utm_source=medium&utm_medium=article&utm_campaign=kube-bench)是为那些对 AI 和 MLOps 的世界感到好奇的人准备的时事通讯。你会在每个月的第一个星期六收到我关于最新人工智能新闻和文章的更新和想法。在这里订阅！

# 库贝长凳

kube-bench 与 docker-bench 类似，是一个 Go 检测工具，而不是强制工具。它通过运行由 [Kubernetes CSI 基准](https://www.cisecurity.org/benchmark/kubernetes)提供的检查来检查您的 Kubernetes 集群是否安全部署。

[](https://medium.com/geekculture/your-docker-setup-is-like-a-swiss-cheese-heres-how-to-fix-it-cd1f49f40256)  

CIS Kubernetes 基准测试的工作方式与 CIS Docker 基准测试相同。然而，由于有许多 Kubernetes 发行版，CIS Kubernetes 基准的具体实现是为每个发行版量身定制的。

kube-bench 不采取措施强化您的 Kubernetes 集群；它可以识别您的部署中的安全漏洞，但是您必须付出额外的努力来填补这些漏洞。

测试被配置到不同的 YAML 文件中。许可和强化配置文件都有测试文件。

最后，kube-bench 运行在 Linux 上，所以除非您使用 Linux 发行版，否则一些审计检查不会起作用。但是，在您的 Kubernetes 集群中，始终可以选择将 kube-bench 作为 pod 运行。我们将探讨这个选项。

## RKE·库贝-长凳模式

针对特定的 Kubernetes 实现，有不同的 CIS Kubernetes 基准实现。我们将在这里研究 RKE 的实现。

RKE·库贝-长凳模式与多克-长凳模式具有相同的结构:

```
- id: < test id>
  text: < what to test>
  audit: < command to test>
  tests:
    test_items: <what to test for>
    — flag:
      set: <true | false>
  remediation: <how to fix>
  scored: <true | false>
```

该模式的主要部分包括:

*   测试 id
*   测试的简短描述
*   包含要运行的命令及其标志的审核部分
*   补救部分，包含修复失败测试的说明

## 评估您的 Kubernetes 部署

我们将 kube-bench 安全工具作为一个容器来运行。更具体地说，在这个例子中，我们将评估一个 RKE 部署。不同的 Kubernetes 发行版需要稍微不同的方法。如果您想跟进，请按照官方的[文档](https://rancher.com/docs/rke/latest/en/installation/)建立一个 RKE 集群。

**评估控制平面**

首先，您将评估集群的控制平面。为此，你需要先`ssh`进入你的主节点。如何做到这一点取决于您的系统。通常，您应该运行类似于下面的命令:

```
ssh USER@HOST
```

一旦你进入，你将运行由兰斯和`exec`提供的`rancher/security-scan`容器来得到结果:

```
docker run --pid=host -v /etc:/node/etc:ro /var:/node/var:ro -it rancher/security-scan:v0.2.2 bash
```

这个容器已经被配置为在适当的 Kubernetes 上下文中使用 kube-bench。因此，要检查控制面板的安全状态，您只需运行以下命令:

```
kube-bench run --targets etcd,master,controlplane,policies --score --config-dir=/etc/kube-bench/cfg --benchmark rke-cis-1.6-hardened
```

请注意，该命令运行对硬化配置文件的检查。您总是可以使用宽容的概要文件，它更宽容。为此，用`--benchmark rke-cis-1.6-permissive`替换`--benchmark rke-cis-1.6-hardened` 。

在执行安全扫描命令时，kube-bench 将针对不同的类别运行多项检查。为了收集故障并仔细调查它们，grep for `FAIL`:

```
kube-bench run --targets etcd,master,controlplane,policies --scored --config-dir=/etc/kube-bench/cfg --benchmark rke-cis-1.6-hardened | grep FAIL
```

**评估工人节点**

评估 worker 节点遵循类似的过程。同样，第一步是将`ssh`放入 worker 节点，运行`rancher/security-scan`容器并将`exec`放入其中:

```
docker run --pid=host -v /etc:/node/etc:ro -v /var:/node/var:ro -ti rancher/security-scan:v0.2.2 bash
```

一旦进入容器上下文，您需要运行的命令如下所示:

```
kube-bench run --targets node --scored --config-dir=/etc/kube-bench/cfg --benchmark rke-cis-1.6-hardened | GREP FAIL
```

注意，唯一的区别是`--targets`标志。您现在将其设置为`node`。

就是这样！现在，您已经检查了主节点和工作节点的安全问题。

# 结论

Kubernetes 部署在默认情况下是不安全的，您应该多做一点来保护大门。幸运的是，像 kube-bench 这样的工具让我们将注意力集中在集群的特定区域。

现在，您已经看到了如何使用 docker-bench 保护 Docker 设置，以及如何使用 kube-bench 保护 Kubernetes 部署。然而，在容器内部运行什么同样重要。因此，在下一篇文章中，我们将看到您可以判断出容器图像是可以安全使用的。

# 关于作者

我叫[迪米特里斯·波罗普洛斯](https://www.dimpo.me/?utm_source=medium&utm_medium=article&utm_campaign=kube-bench)，我是一名为[阿里克托](https://www.arrikto.com/)工作的机器学习工程师。我曾为欧洲委员会、欧盟统计局、国际货币基金组织、欧洲央行、经合组织和宜家等主要客户设计和实施过人工智能和软件解决方案。

如果你有兴趣阅读更多关于机器学习、深度学习、数据科学和数据运算的帖子，请在 Twitter 上关注我的 [Medium](https://towardsdatascience.com/medium.com/@dpoulopoulos/follow) 、 [LinkedIn](https://www.linkedin.com/in/dpoulopoulos/) 或 [@james2pl](https://twitter.com/james2pl) 。

所表达的观点仅代表我个人，并不代表我的雇主的观点或意见。