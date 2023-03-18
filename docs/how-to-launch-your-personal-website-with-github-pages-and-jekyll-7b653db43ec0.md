# 如何用 GitHub Pages 和 Jekyll 启动你的个人网站

> 原文：<https://towardsdatascience.com/how-to-launch-your-personal-website-with-github-pages-and-jekyll-7b653db43ec0>

## 一个完整的一步一步的指南，自动化繁重的工作，让您专注于内容

![](img/057febbf5997cd57a5395b77fb472f19.png)

照片由[斯科特·格雷厄姆](https://unsplash.com/@homajob?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

建立自己的网站是脱颖而出的好方法，也是在网上建立个人品牌的好方法。在这篇文章中，我将分享一个逐步建立和启动一个看起来干净、专业且易于维护的网站的指南。您将学习如何:

*   使用静态站点生成器 Jekyll 在本地构建您的站点，
*   在网站托管服务 GitHub Pages 上托管您的网站，
*   将您的站点连接到您选择的自定义域，并
*   添加动态内容，例如来自您的 Twitter 订阅源的内容。

最重要的是，有了这个设置，你可以在简单的 Markdown 中完成所有的内容创建，而 Jekyll 则负责将你的文字转换成 html。(关于演示，请查看我自己的网站[这里](https://samflender.com)。)

让我们开始吧。

## 1.安装 Jekyll

Jekyll 是一个将 markdown 文件转换成 html 的 Ruby 应用程序。为了使用 Jekyll，你首先需要安装 Ruby。在苹果电脑上，你可以用自制软件做到这一点:

```
brew install chruby ruby-install xz
ruby-install ruby
```

从技术上来说，你刚才做的是先安装`chruby`，它只是一个简单的小 shell 脚本，然后用`chruby`安装最新版本的`ruby`。现在运行以下命令来配置您的环境(最好将这些行添加到您的。巴沙尔或者。zshrc):

```
source /opt/homebrew/opt/chruby/share/chruby/chruby.sh
source /opt/homebrew/opt/chruby/share/chruby/auto.sh
chruby ruby-3.1.2
```

现在你可以在你的 shell 中使用 Ruby 的包管理器`gem`。执行以下命令:

```
gem install Jekyll
```

去安装哲基尔。就是这样！

> 👉不是 Mac 用户？Jekyll 也可以安装在 Windows、Ubuntu 或其他 Linux 系统上。查看此处的说明[。](https://jekyllrb.com/docs/installation/)

## 2 .建立一个新的 git repo 并在本地构建您的站点

首先，你需要在 [GitHub](https://github.com) 上创建一个账户，如果你还没有的话。然后，您创建一个新的 repo，它可以命名为“site”，并将其克隆到您的本地计算机上:

```
git clone git@github.com:<your username>/site.git
```

在`site`目录中，你现在可以运行:

```
jekyll new --skip-bundle . --force
```

这初始化了一个新的 Jekyll 项目，并创建了一堆文件，您将使用这些文件来创建您的网站，最重要的是:

*   `Gemfile`包含执行 Jekyll 所需的 gem 依赖关系，
*   是 Jekyll 配置，它决定了你的站点的外观
*   `index.md`是你网站的首页。这是您添加第一个内容的地方。

接下来，您需要对代码做以下两个小的更改:

1.  在`Gemfile`里面，注释掉了这一行:

```
#gem "jekyll", "~> 4.2.2"
```

改为添加这一行:

```
gem "github-pages", "~> 227", group: :jekyll_plugins
```

> 👉这条线到底是做什么的？这告诉您的 Ruby 编译器，不要直接用 Jekyll 构建站点，而应该用 GitHub Pages Ruby gem 来构建，它拥有在 GitHub Pages 中运行 Jekyll 的所有依赖项。

2.在`_congig.yml`里面，注释掉这两行:

```
#baseurl: "" 
#url: ""
```

这是因为 Github 页面会在构建过程中自动设置这些 URL。

最后，跑

```
bundle install
```

这将在本地构建 html 网站，使用来自`Gemfile`的依赖项、`_config.yml`中的 Jekyll 配置和目录中 Markdown 文件的内容。

## 3.在本地测试你的站点

运行以下命令:

```
bundle exec jekyll serve
```

这将输出一个本地 URL。将此粘贴到您的浏览器中以预览网站。如果您得到错误消息“无法加载这样的文件— webrick”，只需运行`bundle add webrick; bundle install`。这将 webrick 包(一个 HTTP 服务器工具包)的依赖性添加到了 Gemfile 中。

每次您保存对 markdown 文件的更改时，Jekyll 都会自动更新网站，您只需刷新浏览器即可查看更改。

> 👉不想记住这个命令？只需运行`echo 'bundle exec jekyll serve' > dryrun.sh`。现在你可以简单地运行`bash dryrun.sh`来测试你的站点

## 4.部署站点

提交所有本地更改并将它们推送到 github:

```
git add .
git commit -m 'first commit'
git push origin master
```

在您的 GitHub repo 页面上，请访问:

```
Settings → Code and automation → Pages → Build and deployment → Deploy from branch
```

并选择要部署的分支(如主)。现在，GitHub 将运行自己版本的 Jekyll 来构建和部署您的站点。如果您使用`site`作为回购的名称，您的网站现在将托管在:

```
https://www.<your username>.github.io/site
```

每次您向存储库中推送新内容时，这些内容都会在几分钟内自动部署到生产环境中。

> 👉想要立即强制重建一个站点吗？单击“已部署”链接，然后单击“重新运行所有作业”。

## 5.添加自定义域

自定义域名更短，更容易让人记住，这可以为您的网站带来更多流量。以下是添加自定义域的方法:

*   去[https://domains.google.com](https://domains.google.com)查看你想要的域名是否可用。如果有，就购买。它应该花费你大约一个月一美元左右。
*   在 google 域上，在 DNS 设置下，您需要添加两条记录，一条`A`记录和一条`CNAME`记录。在 A 记录中，列出这 4 个 IP 地址(指向 GitHub 的服务器):

```
185.199.108.153
185.199.109.153
185.199.110.153
185.199.111.153
```

*   在《CNAME 记录》中，简单地写着`<your username>.github.io`。

现在，在 GitHub 上，进入设置→页面→自定义域，添加你的自定义域，点击“保存”。您的 Google 域名现在已连接到您的 GitHub 页面。

> 👉设置正确的 DNS 配置可能有点棘手。需要故障排除帮助吗？点击查看特伦特杨的帖子[。](https://dev.to/trentyang/how-to-setup-google-domain-for-github-pages-1p58)

## 6.内容，内容，内容

![](img/b8800571dfffbcc2a6d4a17fb6565c4a.png)

我自己网站上的 Twitter 小部件演示

一旦你的网站建立并运行，就有无限的可能性来添加更多的内容。以下是一些想法:

*   添加一个 Twitter 小部件，列出您最近的推文。只需将这段代码添加到您的`index.md`中:

```
<a class="twitter-timeline" href="https://twitter.com/elonmusk?ref_src=twsrc%5Etfw">Tweets by elonmusk</a> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
```

*(当然，你会想把* `*elonmusk*` *换成你自己的 Twitter 句柄。)*

*   从 Gumroad 添加您的电子书或课程。在 Gumroad 上，转到您网站上想要嵌入的产品，然后导航到*共享→嵌入→模态叠加*，简单地将代码复制粘贴到您的`index.md`。[下面是](https://samflender.com/book/)它在我的网站上的样子。
*   添加一个小部件，显示您的[最近的博客文章](https://github.com/jameshamann/jekyll-display-medium-posts)的提要。
*   使用 [Jekyll 帖子](https://jekyllrb.com/docs/posts/)把你的网站变成一个[个人博客](https://medium.com/@samuel.flender/6-simple-lessons-that-will-help-you-start-your-writing-side-gig-f3b9273fb1ca)。

最后，一个好的实践是使用分支来开发你的站点:当添加一个新的特性时，在一个单独的分支中完成。然后，在 GitHub 上，可以尝试从新的分支部署站点。如果新特性破坏了您的站点，您可以快速将部署分支回滚到`master`。

快乐创作！

*📫* [*订阅*](https://medium.com/subscribe/@samuel.flender) *把我的下一篇文章直接发到你的收件箱。
💪* [*成为中等会员*](https://medium.com/@samuel.flender/membership) *并解锁无限权限。
🐦关注我上* [*中*](https://medium.com/@samuel.flender) *、*[*LinkedIn*](https://www.linkedin.com/in/sflender/)*、*[*Twitter*](https://twitter.com/samflender)*。*