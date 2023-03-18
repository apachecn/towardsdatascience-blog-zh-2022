# 如何使用简单的 Web API 和 GPU 支持在 Docker 中运行稳定的扩散

> 原文：<https://towardsdatascience.com/launch-a-web-api-for-stable-diffusion-under-45-seconds-bbd88cfe41d8>

## 启动 web API，在 45 秒内实现稳定传播

![](img/dd2ecb424fa8ec55a67b574526c7fc9e.png)

稳定扩散产生的虚构黑山羊

[Stability Diffusion](https://github.com/CompVis/stable-diffusion)是一种潜在的文本到图像扩散模型，这要归功于 Stability AI 和 Runway 的合作。它具有最先进的文本到图像合成功能，内存需求相对较小(10 GB)。稳定扩散对其他扩散模型进行了一些改进，以实现这种效率，但这些创新超出了本文的范围，未来的文章将介绍如何在 TensorFlow 中训练扩散模型，并从技术上详细说明其内部工作原理。

Divam Gupta [将稳定扩散从原始权重移植到 TensorFlow / Keras](https://github.com/divamgupta/stable-diffusion-tensorflow) ，这篇文章重点介绍如何使用简单的 web API 和 GPU 支持在 Docker 映像中运行它。

**趣闻**:本帖的特色形象也是稳定扩散产生的。

## 它是如何工作的？

我决定在 TensorDock Marketplace 的 GPU 上运行它。，它应该可以在其他机器上工作，几乎不需要做任何改变，但这是我最近推出的 TensorDock Marketplace 的实验的直接结果。它目前处于公开测试阶段，但我已经喜欢上了他们的创新理念，让高性能计算的使用大众化。除了他们的[价格合理的核心云 GPU 服务](https://www.tensordock.com/)之外，Marketplace edition 还作为一个市场，将客户和 GPU 提供商聚集在一起。主机，即那些有备用 GPU 的主机，可以将它们租给客户，包括独立研究人员、初创公司、业余爱好者、修补匠等。价格极其便宜。[根据 TensorDock](https://www.tensordock.com/host) 的说法，这也让主机赚取 2 到 3 倍的采矿利润。为了比挖掘无意义的密码更好的目的。

服务器可根据所需的 RAM、vCPU 和分配的磁盘进行定制，启动时间太短，大约只有 45 秒。您可以选择从安装了 NVIDIA 驱动程序和 Docker 的最小 Ubuntu 映像开始，或者您也可以使用配置了 NVIDIA 驱动程序、Conda、TensorFlow、PyTorch 和 Jupyter 的成熟映像进行实验。我选择使用 Docker 而不是 Conda 或虚拟环境来隔离我的人工智能项目，我将从安装在 TensorDock Marketplace 上的 Docker 的最小图像开始。

首先，我将展示如何将 web API 封装成 Docker 映像中的稳定分发服务。然后，我将一步一步地描述如何在 TensorDock GPU 上提供它。如果你想在 45 秒内启动并运行，你可以直接跳到“Ok，演示给我看如何运行”部分，或者你可以选择查看 [GitHub repo](https://github.com/monatis/stable-diffusion-tf-docker) 。

## 让我们来整理一下！

TensorFlow 为每个版本提供官方[预建的 Docker 图像](https://www.tensorflow.org/install/docker)来启动您的 Docker 文件。如果您配置了 [NVIDIA Container Runtime](https://developer.nvidia.com/nvidia-container-runtime) 并且更喜欢 TensorFlow 提供的支持 GPU 的 Docker 映像，您可以在 GPU 支持下立即运行 TensorFlow 代码，而不会遇到 CUDA 或 CUDNN 的问题。幸运的是，TensorDock 的最小 Ubuntu 图像带有 NVIDIAContainerRuntime 支持。

为了稳定的扩散，我们以`tensorflow/tensorflow:2.10.0-gpu`开始我们的 Dockerfile。然后我们安装稳定的扩散需求和 FastAPI 来服务一个 web API。最后，我们复制包含 web API 的`app.py`,并将其配置为在容器启动时运行:

```
from tensorflow/tensorflow:2.10.0-gpu

RUN apt update && \
    apt install -y git && \
    pip install --no-cache-dir Pillow==9.2.0 tqdm==4.64.1 \
    ftfy==6.1.1 regex==2022.9.13 tensorflow-addons==0.17.1 \
    fastapi "uvicorn[standard]" git+https://github.com/divamgupta/stable-diffusion-tensorflow.git

WORKDIR /app

COPY ./app.py /app/app.py

CMD uvicorn --host 0.0.0.0 app:app
```

## 撰写以便于配置和启动

当你有一个`docker-compose.yml`文件时，Docker 就更有用了。因此，您可以简单地运行`docker compose up`，用一个命令就可以启动并运行一切。它在处理多个容器时尤其出色，但对于管理单个容器也非常有用。

下面的`docker-compose.yml`文件定义了几个环境变量，并将它们传递给容器。它还支持容器内部的 GPU 访问，并根据 TensorDock Marketplace 的要求正确配置 Docker 网络。在其他平台上，您可能需要删除有关网络的这一部分

```
version: "3.3"

services:
  app:
    image: myusufs/stable-diffusion-tf
    build:
      context: .
    environment:
      # configure env vars to your liking
      - HEIGHT=512
      - WIDTH=512
      - MIXED_PRECISION=no
    ports:
      - "${PUBLIC_PORT?Public port not set as an environment variable}:8000"
    volumes:
      - ./data:/app/data

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]

networks:
  default:
    driver: bridge
    driver_opts:
      com.docker.network.driver.mtu: 1442
```

## 编码时间到了

现在我们已经准备好编写我们的 webAPI 了。我们从所需的导入开始，然后用由`docker-compose.yml`传递的环境变量创建一个图像生成器和 FastAPI 应用程序。

```
import os
import time
import uuid

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field
from stable_diffusion_tf.stable_diffusion import Text2Image
from tensorflow import keras

height = int(os.environ.get("WIDTH", 512))
width = int(os.environ.get("WIDTH", 512))
mixed_precision = os.environ.get("MIXED_PRECISION", "no") == "yes"

if mixed_precision:
    keras.mixed_precision.set_global_policy("mixed_float16")

generator = Text2Image(img_height=height, img_width=width, jit_compile=False)

app = FastAPI(title="Stable Diffusion API")
```

然后，我们为我们的`/generate`端点定义请求和响应主体。值是不言自明的，所以这里不需要额外的文字。

```
class GenerationRequest(BaseModel):
    prompt: str = Field(..., title="Input prompt", description="Input prompt to be rendered")
    scale: float = Field(default=7.5, title="Scale", description="Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    steps: int = Field(default=50, title="Steps", description="Number of dim sampling steps")
    seed: int = Field(default=None, title="Seed", description="Optionally specify a seed for reproduceable results")

class GenerationResult(BaseModel):
    download_id: str = Field(..., title="Download ID", description="Identifier to download the generated image")
    time: float = Field(..., title="Time", description="Total duration of generating this image")
```

最后，是时候写我们的端点了。`/generate`端点将接受一个文本提示以及几个配置值来控制生成，并且它将用生成的图像的惟一 ID 来响应。然后，可以通过`/download`端点下载图像。生成的图像保存在`docker-compose.yml`中配置为 Docker 卷的目录中。

```
@app.post("/generate", response_model=GenerationResult)
def generate(req: GenerationRequest):
    start = time.time()
    id = str(uuid.uuid4())
    img = generator.generate(req.prompt, num_steps=req.steps, unconditional_guidance_scale=req.scale, temperature=1, batch_size=1, seed=req.seed)
    path = os.path.join("/app/data", f"{id}.png")
    Image.fromarray(img[0]).save(path)
    alapsed = time.time() - start

    return GenerationResult(download_id=id, time=alapsed)

@app.get("/download/{id}", responses={200: {"description": "Image with provided ID", "content": {"image/png" : {"example": "No example available."}}}, 404: {"description": "Image not found"}})
async def download(id: str):
    path = os.path.join("/app/data", f"{id}.png")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png", filename=path.split(os.path.sep)[-1])
    else:
        raise HTTPException(404, detail="No such file")
```

## 好的，告诉我怎么跑

不要让这些步骤吓到你——这只是从注册到提出请求的整个过程的一步一步的详细介绍。应该不会超过 10 分钟。

1.  [注册](https://marketplace.tensordock.com/register)并登录 TensorDock 市场。
2.  [转到订单页面](https://marketplace.tensordock.com/order_list)，选择一台提供至少 10 GB 内存的 GPU 的物理机。我建议买一个提供 RTX 3090 的。
3.  这将打开一个允许您配置服务器的模型。我的建议如下:

*   选择每个 GPU 型号的数量:1 个 GeForce RTX 3090 24 GB
*   选择内存容量(GB): 16
*   选择虚拟 CPU 的数量:2
*   选中复选框，最多可转发 15 个端口。您将能够通过这些端口访问您的服务器。
*   在“自定义您的安装”下，选择“Ubuntu 20.04 LTS”。

1.  为您的服务器选择一个密码，并为其命名，例如“stable-diffusion-api”
2.  点击“部署服务器”,瞧！你的服务器将在几秒钟内准备好。
3.  当您看到成功页面时，请单击“下一步”查看详细信息。
4.  查找您的服务器的 IPv4 地址。这可能是真实的 IP，也可能是类似`mass-a.tensordockmarketplace.com`的子域。
5.  找到映射到内部端口 22 的外部端口。您将使用它 SSH 到您的服务器。例如，可能是 20029 年。
6.  使用以下标准连接到您的服务器，例如:

*   `ssh -p 20029 user@mass-a@tensordockmarketplace.com`

Docker 已经配置了 GPU 访问，但是我们需要配置 Docker 网络来进行外部请求。

1.  将此存储库和 cd 克隆到其中:

*   `git clone https://github.com/monatis/stable-diffusion-tf-docker.git && cd stable-diffusion-tf-docker`

1.  将`daemon.json`复制到现有的`/etc/docker/daemon.json`上，并重启服务。不要担心——这只是为 MTU 值添加了一个设置。

*   `sudo cp ./daemon.json /etc/docker/daemon.json`
*   `sudo systemctl restart docker.service`

1.  为您想要使用的公共端口设置一个环境变量，运行 Docker Compose。我们的`docker-compose.yml`文件将从环境变量中提取它，它应该是您配置的端口转发之一，例如 20020。

*   `export PUBLIC_PORT=20020`
*   `docker compose up -d`

1.  一旦它启动并运行，进入`http://mass-a.tensordockmarketplace.com:20020/docs`获取 FastAPI 提供的 Swagger UI。
2.  使用`POST /generate`端点生成扩散稳定的图像。它会用一个下载 ID 来响应。
3.  点击`GET /download/<download_id>`端点下载您的图像。

## 结论

我们能够在 TensorDock Marketplace 的云 GPU 上运行稳定扩散，这是最先进的 tex-to-image 模型之一。它非常便宜，因此适合实验和副业。我将继续在我的一个副业项目中使用它，后续的帖子将提供 TensorDock Marketplace 上培训工作的一步一步的介绍。