# 五月版:设计可用的仪表板

> 原文：<https://towardsdatascience.com/may-edition-designing-usable-dashboards-85bd73b75f97>

## 如何构建能激发良好决策的工具

![](img/26dc8c6ad828c5a8df82f894f2cabe21.png)

照片由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的[Darwin veger](https://unsplash.com/@darwiiiin?utm_source=medium&utm_medium=referral)拍摄

仪表板解决了——最好是防止——好数据的浪费。对于希望分享其辛勤工作成果的数据科学家，以及需要做出基于数据的业务和产品决策的其他利益相关者来说，它们已经成为一种至关重要的媒介。然而，在成功协作的道路上仍然存在许多障碍:如何决定共享哪些信息，以及如何组织这些信息？你如何阻止你的光滑仪表板从你的非 DS 同事的雷达上消失？

我们选择了几个最近的帖子，耐心地解释了开发人们*实际使用的仪表板的理论和实践，并解决了数据专业人员在创建仪表板时遇到的一些常见挑战。对于那些想探索其他话题的人，请继续阅读，发现我们上个月最受欢迎的帖子，以及一些我们非常自豪的原创功能。*

祝阅读愉快，感谢您对我们作者工作的支持。

[TDS 编辑](https://medium.com/u/7e12c71dfa81?source=post_page-----85bd73b75f97--------------------------------)

## TDS 编辑亮点

*   [**如何构建有效(有用)的仪表板**](/how-to-build-effective-and-useful-dashboards-711759534639)[Marie lefe vre](https://medium.com/u/2a04bf49928f?source=post_page-----85bd73b75f97--------------------------------)介绍了一种简化的四步法来构建仪表板，并以她自己的实际经验为基础。(2022 年 3 月 7 分钟)
*   [(2022 年 3 月 7 分钟)](https://medium.com/towards-data-science/the-dos-and-donts-of-dashboard-design-2beefd5cc575)
*   [**使用 Python 推进到专业仪表板，使用 Dash**](/advancing-to-professional-dashboard-with-python-using-dash-and-plotly-1e8e5aa4c668)如果你已经是一个经验丰富的仪表板创建者，考虑通过遵循 [Kay Jan Wong](https://medium.com/u/fee8693930fb?source=post_page-----85bd73b75f97--------------------------------) 的简明、循序渐进的教程来进一步提高你的技能。(2022 年 1 月，6 分钟)
*   [**用 Python、Dash 和 Plotly 创建一个更好的仪表板**](/creating-a-better-dashboard-with-python-dash-and-plotly-80dfb4269882)对于第一次使用仪表板的人来说，[布拉德·巴特莱姆](https://medium.com/u/400f79169eff?source=post_page-----85bd73b75f97--------------------------------)的指南特别全面——它将从头开始引导你完成整个过程。(2021 年 12 月 16 分钟)

## 原始特征

从作者问答到播客片段，我们的团队每周都会发布新文章，介绍我们蓬勃发展的社区的最新消息和想法。以下是最近的几个亮点，以防你错过:

*   [**在机器学习中，失败和不确定性有时是成功的必要成分**](/in-machine-learning-failure-and-uncertainty-are-sometimes-necessary-ingredients-for-success-62d969050866) ，[阿尼·马杜尔卡](https://medium.com/u/c9b0adccc01d?source=post_page-----85bd73b75f97--------------------------------)专访。
*   [**你写得越多，你就越擅长解释你的作品**](/the-more-you-write-the-better-you-are-at-explaining-your-work-708be316defc) :与[Varshita Sher](https://medium.com/u/f8ca36def59?source=post_page-----85bd73b75f97--------------------------------)博士关于她的职业道路和写作策略的对话。
*   [**与人工智能**](/generating-wikipedia-articles-with-ai-995436c9f95f) 生成维基百科文章，这是 TDS 播客上的一段精彩对话，由主持人 [Jeremie Harris](https://medium.com/u/59564831d1eb?source=post_page-----85bd73b75f97--------------------------------) 和人工智能研究员 Angela Fan 主讲。
*   [**发现数学背后的美**](/finding-beauty-behind-all-the-math-844c8e39472d) ，作者聚焦系列的最新作品，由[汉娜·鲁斯](https://medium.com/u/45a9e3b70a2?source=post_page-----85bd73b75f97--------------------------------)主演。

## 热门帖子

人们永远不应该低估群体的智慧——尤其是当讨论的群体是我们敏锐而忠实的 TDS 读者*你*时。以下是过去一个月阅读量最大的一些帖子。

*   [**如何构建可读性和透明度的数据科学项目**](/how-to-structure-a-data-science-project-for-readability-and-transparency-360c6716800) ，作者 [Khuyen Tran](https://medium.com/u/84a02493194a?source=post_page-----85bd73b75f97--------------------------------)
*   [**你应该用这个来可视化 SQL 连接，而不是文氏图**](/you-should-use-this-to-visualize-sql-joins-instead-of-venn-diagrams-ede15f9583fc) ，作者[安德烈亚斯·马丁森](https://medium.com/u/f6366993e3b5?source=post_page-----85bd73b75f97--------------------------------)
*   [**2022 年你应该读的 5 本最好的数据科学书籍**](/the-5-best-data-science-books-you-should-read-in-2022-9209616c203f) ，作者 [Terence Shin](https://medium.com/u/360a9d4d19ab?source=post_page-----85bd73b75f97--------------------------------)
*   [**如何自学数据科学所需的所有技术知识**](/how-to-self-study-all-the-technical-stuff-you-need-for-data-science-62e4b8b8152f) ，作者[弗兰克·安德拉德](https://medium.com/u/fb44e21903f3?source=post_page-----85bd73b75f97--------------------------------)
*   [**成为“真正的”数据分析师**](/becoming-a-real-data-analyst-dcaf5f48bc34) ，作者[凯西·科济尔科夫](https://medium.com/u/2fccb851bb5e?source=post_page-----85bd73b75f97--------------------------------)
*   [**我是一名自学成才的数据科学家。以下是我对新人**](/im-a-self-taught-data-scientist-here-are-my-3-suggestions-for-newcomers-5f5d54e597a8) 的三点建议，作者[Soner y ldr um](https://medium.com/u/2cf6b549448?source=post_page-----85bd73b75f97--------------------------------)

在我们结束之前，请和我们一起向我们在四月份欢迎的一些新作者表示热烈的欢迎，我们很高兴与你们分享他们的作品。(如果你想加入他们的行列，我们很乐意收到你的来信。)他们包括 [Adrienne Kline](https://medium.com/u/7cd59d41e4d7?source=post_page-----85bd73b75f97--------------------------------) 、 [Riccardo Andreoni](https://medium.com/u/76784541161c?source=post_page-----85bd73b75f97--------------------------------) 、 [Aine Fairbrother-Browne](https://medium.com/u/f1056428905d?source=post_page-----85bd73b75f97--------------------------------) 、[William foot](https://medium.com/u/805cf4020d74?source=post_page-----85bd73b75f97--------------------------------)、[Mario Nam Tao shian ti Larcher](https://medium.com/u/cd2b72f39ad4?source=post_page-----85bd73b75f97--------------------------------)、 [Boriharn K](https://medium.com/u/e20a7f1ba78f?source=post_page-----85bd73b75f97--------------------------------) 、 [John Willcox](https://medium.com/u/79bc72438a69?source=post_page-----85bd73b75f97--------------------------------) 、[Ali soleimani](https://medium.com/u/f3a5f77e2245?source=post_page-----85bd73b75f97--------------------------------)、 [Veronica Villa](https://medium.com/u/57c69c050f66?source=post_page-----85bd73b75f97--------------------------------) 、 [Carlo H 【T21AI](https://medium.com/u/52c87f77c5a1?source=post_page-----85bd73b75f97--------------------------------) 、 [Aydin Schwartz](https://medium.com/u/d94d711eed0a?source=post_page-----85bd73b75f97--------------------------------) 、 [Sankar Srinivasan](https://medium.com/u/9c1518365372?source=post_page-----85bd73b75f97--------------------------------) 、 [Kishan Manani](https://medium.com/u/2497691d4d1e?source=post_page-----85bd73b75f97--------------------------------) 、 [Hanzala Qureshi](https://medium.com/u/467270b83111?source=post_page-----85bd73b75f97--------------------------------) 、 [Tara Prole](https://medium.com/u/5ccc73dae224?source=post_page-----85bd73b75f97--------------------------------) 、 [Ariel Jiang](https://medium.com/u/92c0f3ada823?source=post_page-----85bd73b75f97--------------------------------) 、[bi GL Subhash](https://medium.com/u/3cff5f78d221?source=post_page-----85bd73b75f97--------------------------------)、 [Sascha Kirch](https://medium.com/u/5c38dace9d5e?source=post_page-----85bd73b75f97--------------------------------) 、 [Dan Pietrow](https://medium.com/u/f0636962dd83?source=post_page-----85bd73b75f97--------------------------------) 、【t5t [Charlotte Tu](https://medium.com/u/4196c2bfc3b?source=post_page-----85bd73b75f97--------------------------------) 、[Jack chi-Hsu Lin](https://medium.com/u/61758d129bdd?source=post_page-----85bd73b75f97--------------------------------)、 [Pieter Steyn](https://medium.com/u/41397dbd3d19?source=post_page-----85bd73b75f97--------------------------------) 、 [Joe Sasson](https://medium.com/u/32a49c3f499a?source=post_page-----85bd73b75f97--------------------------------) 、 [Anushka Gupta](https://medium.com/u/bd164c6cfac0?source=post_page-----85bd73b75f97--------------------------------) 、 [Louis Casanave](https://medium.com/u/a25bdaa6a5ad?source=post_page-----85bd73b75f97--------------------------------) 、 [Robin Thibaut](https://medium.com/u/7e56039e626c?source=post_page-----85bd73b75f97--------------------------------) 、 [Mario Hayashi](https://medium.com/u/f14e7507b476?source=post_page-----85bd73b75f97--------------------------------) 、[Maya mura](https://medium.com/u/d2082e3d715d?source=post_page-----85bd73b75f97--------------------------------)