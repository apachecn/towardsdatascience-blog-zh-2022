# 如何在云中运行机器学习超参数优化—第 2 部分

> 原文：<https://towardsdatascience.com/how-to-run-machine-learning-hyperparameter-optimization-in-the-cloud-part-2-23b1dac5ebed>

## 在专用射线簇上进行调谐的两种方法

![](img/fc62711d52a23f8ca1cb6bc4a088cc60.png)

大卫·坎特利在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

这是关于云中的超参数调整(HPT)机器学习模型的三部分帖子的第二部分。在[第 1 部分](/how-to-run-machine-learning-hyperparameter-optimization-in-the-cloud-part-1-7877cdd6e879)中，我们通过引入问题并定义一个我们将在调试演示中使用的玩具模型来搭建舞台。在这一部分中，我们将回顾基于云的优化的两个选项，这两个选项都涉及在专用调优集群上的并行实验。

# 选项 1:云实例集群上的 HPT

我们考虑在云中执行 HPT 的第一个选项是基于云实例的集群。实际上，有几十种不同的方式来设置实例集群。例如，要在 AWS 上创建一个集群，您可以:1)简单地通过 EC2 控制台启动您想要的数量的 [Amazon EC2](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance) 实例，2)使用容器编排框架(如 [Kubernetes](https://kubernetes.io/) )构建和管理一个集群，3)使用 Amazon 的*托管的* Kubernetes 服务、 [Amazon EKS、](https://aws.amazon.com/eks/)等。

这种 HPT 选项的主要优点是它提供的灵活性。当您启动自己的集群时，您可以随意设置它。这意味着以你想要的任何方式执行 HPT 的自由——使用任何框架、任何算法、任何特性(例如每个实例多次尝试)、任何[自动缩放](https://docs.aws.amazon.com/eks/latest/userguide/autoscaling.html)机制等等。特别是，您可以以这样一种方式设置您的集群，即头节点运行在一个可靠的非现场(不可预占的)实例上，但是所有工作节点都运行在现场实例上以降低成本。

这种灵活性和自由的代价是设置和维护该选项所需的工作。启动、配置和管理集群需要一定程度的专业知识。选择这种方式的组织通常会有一个完全致力于集群管理的(devops)团队。有些组织可能已经制定了指导方针(例如出于安全原因)，您需要遵守这些指导方针。这可能使这种类型的解决方案的使用更加复杂，或者对其上述自由度引入限制。

在这篇文章中，我们将演示使用[射线](https://docs.ray.io/en/latest/index.html)框架创建集群。Ray 包含了对[在 AWS](https://docs.ray.io/en/latest/cluster/vms/getting-started.html#vm-cluster-quick-start) 上启动集群的内置支持。为了启动 HPT 的集群，我们在 tune.yaml YAML 文件中放置了以下集群配置。

```
**cluster_name**: hpt-cluster
**provider**: {type: aws, region: us-east-1 }
**auth**: {ssh_user: ubuntu}
**min_workers**: 0
**max_workers**: 7
**available_node_types:
**  **head_node**:
    **node_config**: {InstanceType: g4dn.xlarge, 
                  ImageId: ami-093e10b196d7cc7f0}
  **worker_nodes**:
    **node_config**: {InstanceType: g4dn.xlarge, 
                  ImageId: ami-093e10b196d7cc7f0}
**head_node_type**: head_node
**setup_commands:
**    - echo 'export
            PATH="$HOME/anaconda3/envs/pytorch_p39/bin:$PATH"' >> 
            ~/.bashrc
    - conda activate pytorch_p39 && 
            pip install "ray[tune]" "ray[air]" &&
            pip install mup transformers evaluate datasets
```

这个 YAML 文件定义了一个 HPT 环境，其中有多达八个 Amazon EC2 g4dn.xlarge 实例，每个实例都有 65.3 版本的 [AWS 深度学习 AMI](https://aws.amazon.com/machine-learning/amis/) ，预配置为使用专用 PyTorch conda 环境，并且预安装了所有 Python 依赖项。(为了简单起见，我们从脚本中省略了 Python 包版本。)

启动该集群的命令是:

```
ray up tune.yaml -y
```

假设正确配置了所有 AWS 帐户设置，这将创建一个集群头节点，我们将在运行 HPT 时连接到该节点。在下面的代码块中，我们演示了如何使用[光线调节](https://docs.ray.io/en/latest/tune/index.html)库来运行 HPT。这里我们选择使用带有随机参数搜索的 [ASHA](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#asha-tune-schedulers-ashascheduler) 调度算法。该算法被配置为总共运行 32 个实验，每个实验最多运行 8 个时期，并且每个实验都有不同的优化器*学习速率*候选。一次最多可以运行八个平行实验。实验将由报告的评估准确度来衡量。根据 [ASHA 的提前停止算法](https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/)，表现不佳的实验将被提前终止。

```
def hpt():
  from ray import tune

 **# define search space**
  config = {
    "lr": tune.loguniform(1e-6, 1e-1),
  } **# define algorithm**
  from ray.tune.schedulers import ASHAScheduler
  scheduler = ASHAScheduler(
    max_t=8,
    grace_period=1,
    reduction_factor=2,
    metric="accuracy",
    mode="max") gpus_per_trial = 1 if torch.cuda.is_available() else 0
  tuner = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(**train**), **# our train function** 
        resources={"cpu": 4, "gpu": gpus_per_trial}),
    tune_config=tune.TuneConfig(num_samples=32,
                                max_concurrent_trials=8,
                                scheduler=scheduler,
                                ),
    param_space=config,
  ) **# tune**  
  results = tuner.fit()
  best_result = results.get_best_result("accuracy", "max")
  print("Best trial config: {}".format(best_result.config))
  print("Best final validation accuracy: {}".format(
      best_result.metrics["accuracy"]))**if __name__ == "__main__":
  import ray
  ray.init()
  hpt()**
```

最后一个环节是[会议记者](https://docs.ray.io/en/latest/tune/api_docs/trainable.html#training-tune-trainable-session-report)。为了让 Ray Tune 能够跟踪实验的进度，我们将以下回调添加到了[拥抱脸训练器*回调*的列表中:](https://huggingface.co/transformers/v4.0.1/main_classes/callback.html#trainercallback)

```
from transformers import TrainerCallback
class **RayReport**(TrainerCallback):
  def on_evaluate(self, args, state, control, metrics, **kwargs):
    from ray.air import session
    **session.report({"loss": metrics['eval_loss'],
                    "accuracy": metrics['eval_accuracy']})**
```

以下命令将触发 HPT 作业:

```
ray submit tune.yaml train.py
```

为了支持八个并行实验的需求，集群自动缩放器将启动七个额外的(工作)节点。HPT 的工作将总共运行 32 个实验，根据 ASHA 的提前停止算法，其中一些可能会提前停止。

以下命令将停止集群(尽管它不会终止 EC2 实例):

```
ray down tune.yaml -y
```

请不要被我们这里描述的相对简单的流程所迷惑。在实践中，如上所述，适当配置您的云环境、适当配置您的 YAML 文件，以及适当管理您的集群是非常棘手的。我们将讨论的其余基于云的 HPT 解决方案将通过使用高级云培训服务来管理计算实例，从而绕过这些复杂性。

## 结果

我们的 HPT 作业运行了大约 30 分钟，产生了以下结果:

```
**Total run time: 1551.68 seconds (1551.47 seconds for the tuning loop).
Best trial config: {'lr': 2.393250830770165e-05}
Best final validation accuracy: 0.7669172932330827**
```

# 选项 2:在管理培训环境中的 HPT

专门的云训练服务，比如亚马逊 SageMaker，为机器学习模型开发提供了很多便利。除了简化启动和管理培训实例的流程之外，它们还可能包括一些引人注目的功能，如加速数据输入流、分布式培训 API、高级监控工具等。这些特性使得托管训练环境成为许多机器学习开发团队的首选解决方案。挑战在于如何扩展这些环境以支持 HPT。在本节中，我们将介绍三种解决方案中的第一种。第一种方法将演示如何在 Amazon SageMaker 培训环境中运行 Ray Tune HPT 作业。与前面的方法相反，在前面的方法中，我们显式地定义并启动了实例集群，这里我们将依靠 Amazon SageMaker 来完成这项工作。虽然前面的方法包括一个自动缩放器，用于根据 HPT 调度算法来缩放集群，但在此方法中，我们创建了一个固定大小的实例集群。我们将使用上面定义的相同的基于 Ray Tune 的解决方案和相同的 *hpt()* 函数，并修改入口点以在由托管服务启动的 EC2 集群上设置 Ray 集群:

```
if __name__ == "__main__":
  # utility for identifying the head node
  def get_node_rank() -> int:
    import json, os
    cluster_inf = json.loads(os.environ.get('SM_RESOURCE_CONFIG'))
    return cluster_inf['hosts'].index(cluster_inf['current_host']) # utility for finding the hostname of the head node
  def get_master() -> str:
    import json, os
    cluster_inf = json.loads(os.environ.get('SM_RESOURCE_CONFIG'))
    return cluster_inf['hosts'][0] if get_node_rank() == 0:
 **# the head node starts a ray cluster and starts the hpt**    import subprocess
    p = subprocess.Popen('**ray start --head --port=6379**',  
                         shell=True).wait()
    import ray
    ray.init()
    hpt()
  else:
    **# worker nodes attach to the hpt cluster**
    import time
    import subprocess
    p = subprocess.Popen(
           f"**ray start --address='{get_master()}:6379**'",
           shell=True).wait()
    import ray
    ray.init()
    try:
 **# keep node alive until the hpt process on head node completes**      while ray.is_initialized():
        time.sleep(10)
    except:
      pass
```

下面的代码块演示了如何使用我们的 HPT 脚本所需的资源来设置 Amazon SageMaker 培训作业。我们不是启动八个单 GPU 实例来支持八个并行实验，而是请求两个四 GPU 实例来展示我们在每个实例上运行多个(四个)实验的能力。当使用我们将要讨论的下面两种 HPT 方法时，这是不可能的。我们还将每个实例的资源调整到 12 个 CPU(而不是 4 个)。

```
from sagemaker.pytorch import PyTorch 
estimator=PyTorch(
           entry_point='train.py',
           source_dir='./' #contains train.py and requirements file
           role=<role>,
           **instance_type='ml.g4dn.12xlarge', # 4 gpus 
           instance_count=2,
**           py_version='py38',
           pytorch_version='1.12')
estimator.fit()
```

*source_dir* 应该指向包含 train.py 脚本和 requirements.txt 文件的本地目录，该文件包含所有 Python 包依赖项:

```
ray[air]
ray[tune]
mup==1.0.0
transformers==4.23.1
datasets==2.6.1
evaluate==0.3.0
```

## 利弊

这种调优方法分享了上一节中基于集群的方法的许多优点——我们可以非常自由地运行我们想要的任何 HPT 框架和任何 HPT 算法。这种方法的主要缺点是缺乏自动伸缩性。集群实例的数量需要预先确定，并在整个培训工作期间保持不变。如果我们使用 HPT 算法，其中并行实验的数量在调整过程中会发生变化，我们可能会发现一些(昂贵的)资源会闲置一段时间。

另一个限制与使用折扣 spot 实例的能力有关。虽然 Amazon SageMaker 支持从 spot 中断中恢复[，但是在撰写本文时，spot 配置已经应用于集群中的所有实例。您不能选择将头节点配置为持久节点，而只能将工作节点配置为现场实例。此外，单个实例的现场中断将触发整个集群的重启。虽然这并没有完全禁止使用 spot 实例，但确实使它变得更加复杂。](https://docs.aws.amazon.com/sagemaker/latest/dg/model-managed-spot-training.html)

## 结果

我们在亚马逊 SageMaker 内部的 HPT 运行结果总结如下:

```
**Total run time: 893.30 seconds (893.06 seconds for the tuning loop).
Best trial config: {'lr': 3.8933781751481333e-05}
Best final validation accuracy: 0.7894736842105263**
```

# 下一个

由三部分组成的帖子的最后一部分[将探讨基于云的 HPT 的另外两种方法，一种是使用托管 HPT 服务，另一种是将托管培训工作打包到 HPT 解决方案中。](/how-to-run-machine-learning-hyperparameter-optimization-in-the-cloud-part-3-f66dddbe1415)