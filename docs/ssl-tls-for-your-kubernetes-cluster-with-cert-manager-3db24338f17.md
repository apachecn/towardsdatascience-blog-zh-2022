# 带证书管理器的 Kubernetes 集群的 SSL/TLS

> 原文：<https://towardsdatascience.com/ssl-tls-for-your-kubernetes-cluster-with-cert-manager-3db24338f17>

## 为浏览您的域的用户设置安全连接

![](img/7e557ae3d69fe69b4f89f361ed159296.png)

丹尼·米勒在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

如果您阅读了来自 [*的这篇文章，为您的 AWS Kubernetes 集群*](/handling-dns-and-ssl-tls-for-your-aws-kubernetes-cluster-f3ecd0991e6a) 处理 DNS 和 SSL/TLS，那么您已经为集群上的一些服务获得了域路由流量，但是您的用户仍然会在他们的浏览器上看到讨厌的消息，告诉他们这是一个不安全的连接。让我们解决这个问题。

需要 SSL 证书，以便浏览器可以创建与您的服务的安全连接。在 Kubernetes 中，SSL 证书被存储为 Kubernetes 的秘密。证书通常在一到两年内有效，过期后就会失效，因此会有很大的管理开销和一些停机的可能性。我们需要一个自我管理并自动更新过期证书的设置。

这就是 C [ert 经理](https://cert-manager.io/docs/)的用武之地。Cert-manager 是我们在您的集群中部署的一种资源，它可以与像 [Let's Encrypt](https://letsencrypt.org/) (这是免费的)这样的认证机构对话，为您的域生成证书。因此，在我们深入探讨之前，让我们在您的集群中部署 cert-manager

我将使用 v1.8.0，但是您可以在这里查看最新的 cert-manager 版本。

让我们从 releases 下载 cert-manager.yaml 文件。

```
curl -LO [https://github.com/jetstack/cert-manager/releases/download/v1.8.0/cert-manager.yaml](https://github.com/jetstack/cert-manager/releases/download/v1.8.0/cert-manager.yaml)
```

然后，让我们将 cert-manager 部署到一个名为 cert-manager 的名称空间。

```
kubectl create namespace cert-managerkubectl apply --validate=false -f cert-manager.yaml
```

为了将 cert-manager 连接到一个认证机构，比如让我们加密另一个 Kubernetes 对象，需要部署一个名为 Issuer 的对象。基本上，当我们从 cert-manager 请求证书时，它会创建一个证书请求对象，要求发行者从证书颁发机构请求一个新的证书。

现在，在部署我们的发行者之前，我们需要部署一个 nginx 入口控制器。为什么？因为当 cert-manager 向像 Let's Encrypt 这样的发行者请求证书时，Let's Encrypt 会发送一个 http 质询，需要 cert-manager 完成该质询才能提供证书。因此，要让 cert-manager 与 Let's Encrypt 对话，我们需要使用 nginx 入口控制器将其开放给互联网。

让我们将入口控制器部署到 ingress-nginx 名称空间。

```
kubectl create ns ingress-nginxkubectl -n ingress-nginx apply -f [https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v0.1.2.0/deploy/static/provider/cloud/deploy.yaml](https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v0.1.2.0/deploy/static/provider/cloud/deploy.yaml)
```

接下来，我们将设置发行者。颁发者指定证书颁发机构的服务器和存储颁发者密钥的 Kubernetes 密钥参考的名称。在我们的例子中，我们使用 **acme** 作为发布者(也就是让我们加密),因此我们还指定了我们希望如何解决 solvers 下的让我们加密挑战。你可以在这里了解更多关于[的信息。这是发行者模板。](https://cert-manager.io/docs/configuration/acme/)

```
apiVersion: cert-manager.io/v1                             
kind: ClusterIssuer                             
metadata:                               
  name: letsencrypt-cluster-issuer                             
spec:                            
  **acme: **                                
    **server:** [**https://acme-v02.api.letsencrypt.org/directory**](https://acme-v02.api.letsencrypt.org/directory)
    email: your-email@email.com
    **privateKeySecretRef: **                                                                  
      **name: letsencrypt-cluster-issuer-key**
    **solvers:**
    - http01:
        ingress:
          class: nginx
```

让我们部署发行者。

```
kubectl apply -f cert-issuer.yaml# view the 
kubectl describe clusterissuer letsencrypt-cluster-issuer
```

我们已经有了证书管理器和发行者。现在我们可以申请证书了。这是证书对象的模板。

```
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: example-cert  #name of this object
  namespace: default #same namespace as 
spec:
  **dnsNames:
    - example.com**
  **secretName: example-tls-cert**
  **issuerRef:
    name: letsencrypt-cluster-issuer
    kind: ClusterIssue**r
```

在模板中，我们指定需要证书的 DNS 名称、存储证书的 Kubernetes secrets 中的秘密名称以及对我们之前部署的发行者的引用。还要确保使用相同的名称空间，您将在其中部署将使用该证书的服务。

让我们部署它:

```
kubectl apply -f certificate.yaml
```

一旦证书被颁发，你应该看到它在 Kubernetes 的秘密。

```
kubectl get secrets
```

## 利用证书

假设您已经部署了一个应用程序及其服务，您希望通过入口将该应用程序公开到 internet，并使用我们上面发布的证书为其设置 TLS。以下是您可以使用的模板:

```
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
  annotations:
    **kubernetes.io/ingress.class: nginx**
spec:
  **tls:**
  **- hosts:
    - example.com**
  **secretName: example-tls-cert**

  rules:
  - **host: example.com**
    http:
      paths:
        - path: /
          pathType: Exact
          backend:
            **service:**
              **name: backend-service**
              port:
                number: 80
```

在 tls 部分，我们为这个入口路由指定 DNS 主机，并为我们之前创建的证书指定秘密名称。我们还传递入口将路由到的服务的名称。然后部署:

```
kubectl apply -f ingress.yaml
```

现在你知道了。总的来说，我们已经在集群中部署了 cert-manager 和 Issuer 资源。然后，我们为 cert-manager 创建了一个证书对象，通过发行者发出证书请求，并向 Kubernetes secrets 添加一个新证书。然后我们创建了一个 nginx 路由来使用证书。

如果你设法使用前一篇文章[中描述的外部域名(或其他方式)连接你的域，并使用 cert-manager 获得 TLS 证书，那么当你的用户在浏览器中访问你的域时，他们应该会获得安全连接。](/handling-dns-and-ssl-tls-for-your-aws-kubernetes-cluster-f3ecd0991e6a)