# 为 AWS Kubernetes 集群处理 DNS 和 SSL/TLS

> 原文：<https://towardsdatascience.com/handling-dns-and-ssl-tls-for-your-aws-kubernetes-cluster-f3ecd0991e6a>

## 在 AWS 上将 Kubernetes 服务连接到您的域的快速纲要

![](img/55615fe29aee57117b5482450f4d8e67.png)

照片由[在](https://unsplash.com/@comparefibre?utm_source=medium&utm_medium=referral) [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上对比纤维

因此，您已经在 AWS 上创建了一个 Kubernetes 集群，在其上部署了一些应用程序，并最终准备向世界展示您所构建的东西。但是您需要了解如何将域连接到您的应用程序并设置 SSL/TLS。我以前也有过类似的经历，所以我写了这个快速的总结，这样你就不用去库伯内特兔子洞找答案了。

*我会为* [*SSL/TLS 设置*](/ssl-tls-for-your-kubernetes-cluster-with-cert-manager-3db24338f17) *写一篇单独的文章，并在这里链接，所以这篇文章不会太长。*

## 将您的域迁移到 AWS 路由 53

Route53 是 AWS 提供的 DNS 服务。它的工作是将请求路由到您的域和子域，再路由到附属于它们的 IP。

您可能已经通过 Route53 注册了您的域名，在这种情况下，您不需要在这里做任何事情。但是，如果您从另一个提供商处购买了您的域名，例如 Namecheap，那么您需要将其转让给 Route53 来管理您的 DNS 记录和路由流量。

要通过 AWS 控制台执行此操作，请导航到 AWS Route53 并创建一个托管区域。托管区域保存有关您希望如何路由流量的记录。创建托管区域时，您将被要求提供一个域，例如 example.com，但如果您不想迁移整个域，也可以放入一个子域，例如 test.example.com。

您的新托管区域将有一个包含 4 台服务器的 NS 记录。进入您的域的管理部分，在它当前所在的提供商中。它将有一节添加自定义域名。将 4 个名称服务器复制到自定义 DNS 部分。你完了。现在，您需要等待 24 小时才能完成迁移。

## 使用 Kubernetes 集群配置外部 DNS

为什么我们把你的域名迁移到 Route53？因为我们希望在您的 DNS 上为您所有公开的 Kubernetes 服务和入口自动创建新记录，如果您的 DNS 位于 AWS 上，这将更容易实现。ExternalDNS 将为我们做这件事。它基本上是我们将在您的群集上部署的一个单元。它将在 route53 中寻找需要 DNS 记录的资源列表，并自动创建这些记录。

以下是外部 DNS 的 K8S 模板:

```
apiVersion: v1
kind: ServiceAccount
metadata:
  name: external-dns
  # If you're using Amazon EKS with IAM Roles for Service Accounts, specify the following annotation.
  # Otherwise, you may safely omit it.
  annotations:
    # Substitute your account ID and IAM service role name below.
    **eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT-ID:role/IAM-SERVICE-ROLE-NAME**
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: external-dns
rules:
- apiGroups: [""]
  resources: ["services","endpoints","pods"]
  verbs: ["get","watch","list"]
- apiGroups: ["extensions","networking.k8s.io"]
  resources: ["ingresses"]
  verbs: ["get","watch","list"]
- apiGroups: [""]
  resources: ["nodes"]
  verbs: ["list","watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: external-dns-viewer
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: external-dns
subjects:
- kind: ServiceAccount
  name: external-dns
  namespace: default
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: external-dns
spec:
  strategy:
    type: Recreate
  selector:
    matchLabels:
      app: external-dns
  template:
    metadata:
      labels:
        app: external-dns
      # If you're using kiam or kube2iam, specify the following annotation.
      # Otherwise, you may safely omit it.
      annotations:
        **iam.amazonaws.com/role: arn:aws:iam::ACCOUNT-ID:role/IAM-SERVICE-ROLE-NAME**
    spec:
      serviceAccountName: external-dns
      containers:
      - name: external-dns
        image: k8s.gcr.io/external-dns/external-dns:v0.7.6
        args:
        - --source=service
        - --source=ingress
        - --domain-filter=external-dns-test.my-org.com # will make ExternalDNS see only the hosted zones matching provided domain, omit to process all available hosted zones
        - --provider=aws
        - --policy=upsert-only # would prevent ExternalDNS from deleting any records, omit to enable full synchronization
        - --aws-zone-type=public # only look at public hosted zones (valid values are public, private or no value for both)
        - --registry=txt
        - --txt-owner-id=my-hostedzone-identifier
      securityContext:
        fsGroup: 65534 # For ExternalDNS to be able to read Kubernetes and AWS token files
```

您会注意到几行粗体字表示 AWS IAM 角色 Arn 的注释。IAM 角色将由 ExternalDNS 承担。它应该有权向 Route53 上的托管区域添加记录。以下是创建该角色的方法。

将这个内联策略保存到一个 json 文件。该策略为角色提供了路由 53 托管区域的权限。

```
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "route53:ChangeResourceRecordSets"
      ],
      "Resource": [
        "arn:aws:route53:::hostedzone/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "route53:ListHostedZones",
        "route53:ListResourceRecordSets"
      ],
      "Resource": [
        "*"
      ]
    }
  ]
}
```

使用 AWS CLI 创建角色。

```
aws iam create-role — role-name role-example — assume-role-policy-document path/to/policy
```

将上面命令返回的 Arn 替换到 ExternalDNS 模板中的粗体部分，并部署。

```
kubectl -n external-dns apply template.yaml
```

好了，我们完成了外部 DNS 的设置。现在，将为您部署的每个入口或服务自动创建一个 DNS 记录。

## 它是如何工作的一个例子

这里有一个指向 mynginx.your-domain.com 的 nginx 服务的模板。

```
apiVersion: v1
kind: Service
metadata:
  name: nginx
  annotations:
    **external-dns.alpha.kubernetes.io/hostname: mynginx.your-domain.com**
spec:
  type: LoadBalancer
  ports:
  - port: 80
    name: http
    targetPort: 80
  selector:
    app: nginx---apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - image: nginx
        name: nginx
        ports:
        - containerPort: 80
          name: http
```

为了测试，你可以用`kubectl apply -f [your-file-name.yaml]`应用上面的模板，然后在 route53 中检查你的托管区域下的记录。你会发现一个新的 A 记录自动为 mynginx.your-domain.com 创建，指向这个 nginx 服务的 DNS 名称。

这就是连接您的域。总的来说，我们已经在您的集群上部署了 ExternalDNS，以便它可以自动将指向您集群上的服务和入口的新记录添加到 Route53 上托管的域的 DNS 中。我们还将您的域迁移到了 Route53，因此它可以由 AWS 管理，并创建了一个新角色来授予 route 53 external DNS 权限。接下来，您需要为您的服务设置 TLS/SSL。我将在下一篇文章中讨论这个问题，并在这里链接。