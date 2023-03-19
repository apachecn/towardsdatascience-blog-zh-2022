# 六月版:走进 SHAP

> 原文：<https://towardsdatascience.com/june-edition-get-into-shap-b9e0c1e905b8>

## 一个强大的可解释人工智能方法的来龙去脉

![](img/89870e1ec344f7cc162a0c488fbc4e36.png)

赫克托·j·里瓦斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

近年来，机器学习模型的能力和规模已经发展到了新的高度。随着复杂性的增加，对构建这些模型的从业者和解释其结果的人来说，都需要更多的责任和透明度。

在可解释人工智能的广阔领域中，一种显示出巨大前景(并吸引了大量注意力)的方法是 SHAP(来自“沙普利加法解释”)；[正如其创造者所说](https://shap.readthedocs.io/en/latest/)，这是一种“解释任何机器学习模型输出的博弈论方法”最近几周，我们发表了一些关于 SHAP 的优秀作品，对于 6 月的月刊，我们决定分享几篇从多个角度涵盖它的优秀文章，从高度理论化到极端实践化。

祝阅读愉快，感谢您对我们作者工作的支持。

[TDS 编辑](https://medium.com/u/7e12c71dfa81?source=post_page-----b9e0c1e905b8--------------------------------)

## TDS 编辑亮点

*   [**可解释的人工智能:打开黑盒**](/explainable-ai-unfold-the-blackbox-5488253c01fd)要清晰透彻地介绍 XAI，尤其是 SHAP 和沙普利的价值观，只需看看 [Charu Makhijani](https://medium.com/u/a3b23858b69?source=post_page-----b9e0c1e905b8--------------------------------) 的新帖。(2022 年 5 月 10 分钟)
*   [**解释预言的艺术**](/the-art-of-explaining-predictions-22e3584ed7d8)
    [康纳·奥沙利文](https://medium.com/u/4ae48256fb37?source=post_page-----b9e0c1e905b8--------------------------------)强调了人性化解释的重要性，并展示了 SHAP 创造这些解释的能力。(2022 年 5 月 11 分钟)
*   [**SHAP 对语言模型的分区交代**](/shaps-partition-explainer-for-language-models-ec2e7a6c1b77)沙普利值、欧文值、分区交代是如何相互联系的？Lilo Wagner 在她的处女作《TDS 邮报》中，她来到了 SHAP 图书馆。(2022 年 5 月 9 分钟)
*   [**SHAP 价值观及其在机器学习中的应用简介**](/introduction-to-shap-values-and-their-application-in-machine-learning-8003718e6827)要全面、耐心地了解 SHAP 背后的数学，以及它在现实生活中是如何工作的，这里是 [Reza Bagheri](https://medium.com/u/da2d000eaa4d?source=post_page-----b9e0c1e905b8--------------------------------) 的权威指南。(2022 年 3 月，81 分钟)
*   [**SHAP:用 Python 解释任何机器学习模型**](/shap-explain-any-machine-learning-model-in-python-24207127cad7)要想更快地了解 SHAP，你可以随时重温 [Khuyen Tran](https://medium.com/u/84a02493194a?source=post_page-----b9e0c1e905b8--------------------------------) 的热门教程。(2021 年 9 月，9 分钟)
*   [**解释公平的衡量标准**](/explaining-measures-of-fairness-f0e419d4e0d7)最后，SHAP 的创造者 [Scott Lundberg](https://medium.com/u/3a739af9ef3a?source=post_page-----b9e0c1e905b8--------------------------------) 在 TDS 上写了大量关于这个库的文章。在这份来自我们档案的长期收藏中，斯科特提出了两个至关重要的概念:可解释性和公平性。(2020 年 3 月 11 分钟)

## 原始特征

从作者问答到播客剧集，我们的团队为您的阅读和聆听乐趣汇集了原创功能，以下是最近的几个亮点:

*   " [**Bryan R. Vallejo 利用地理空间科学和实时数据帮助生态保护**](/bryan-r-vallejo-leverages-geospatial-science-and-real-time-data-to-help-ecological-conservation-3ff7ca8fb007) **。**“不要错过我们与 [Bryan R. Vallejo](https://medium.com/u/cbd681aaa725?source=post_page-----b9e0c1e905b8--------------------------------) 关于新地理空间技术和研究方法的对话。
*   [**数据科学趋势**](/trends-in-data-science-633f092ffa52) 。数据科学下一步将走向何方？该领域的未来是这一期播客的中心，由 Jeremie Harris 和数据女性的创始人 Sadie St. Lawrence 客串。
*   **[**一个数据科学领导者最必备的技能是什么？**](/what-are-the-most-essential-skills-for-a-data-science-leader-683355c8915f)**对于[玫瑰日](https://medium.com/u/a7f2e8e50135?source=post_page-----b9e0c1e905b8--------------------------------)来说，良好的沟通和同理心(以及其他一些经常被低估的软技能)与技术掌握一样重要。****
*   ****[**“做好你的研究之后，是时候相信你的直觉了**](/after-doing-your-research-its-time-to-trust-your-instinct-252412610fc9) **”**我们与博士生 [Murtaza Ali](https://medium.com/u/607fa603b7ce?source=post_page-----b9e0c1e905b8--------------------------------) 的聊天涵盖了数据科学职业可以选择的不同道路，以及在选择这些道路时应该问自己的问题。****
*   ****[**合成数据可能改变一切**](/synthetic-data-could-change-everything-fde91c470a5b) 。最新一集的 TDS 播客由 Alex Watson 主讲，探讨了合成数据的隐私和性能优势。****

## ****热门帖子****

****如果你想深入了解上个月最热门的一些文章和对话，这里有一些 5 月份阅读量最大的帖子。****

*   ****[**3 个最有价值的数据科学技能让我的薪水增加了 60%**](/3-most-valuable-data-science-skills-that-increased-my-salary-by-60-89b4bbe0b34f) 作者[特伦斯·申](https://medium.com/u/360a9d4d19ab?source=post_page-----b9e0c1e905b8--------------------------------)****
*   ****[**rajini++:由**](/rajini-the-superstar-programming-language-db5187f2cc71) **[Aadhithya Sankar](https://medium.com/u/82053676fe58?source=post_page-----b9e0c1e905b8--------------------------------) 开发的超级编程语言******
*   ****[**没有计算机科学学位的数据科学初学者的最佳编程技巧**](/the-best-coding-tips-for-any-data-science-beginner-without-a-cs-degree-3881e6142668)Hennie de Harder****
*   ****[**PyScript —在您的浏览器中释放 Python 的威力**](/pyscript-unleash-the-power-of-python-in-your-browser-6e0123c6dc3f) 作者 [Eryk Lewinson](https://medium.com/u/44bc27317e6b?source=post_page-----b9e0c1e905b8--------------------------------)****
*   ****[**如何轻松学习新的编程语言**](/how-to-learn-new-programming-languages-easily-1e6e29d3898a) 作者 [Kay Jan Wong](https://medium.com/u/fee8693930fb?source=post_page-----b9e0c1e905b8--------------------------------)****
*   ****[**一种超快速的 Python 循环方式**](/a-super-fast-way-to-loop-in-python-6e58ba377a00) 作者[弗兰克·安德拉德](https://medium.com/u/fb44e21903f3?source=post_page-----b9e0c1e905b8--------------------------------)****

****在过去的一个月里，我们有幸分享了优秀的 TDS 新作者的作品。请和我们一起欢迎[玛戈·哈奇](https://medium.com/u/99a1fb5fa4e7?source=post_page-----b9e0c1e905b8--------------------------------)、[奥尔朱万·扎法兰尼](https://medium.com/u/155bd470ac4c?source=post_page-----b9e0c1e905b8--------------------------------)、[奥斯卡·涅梅诺贾](https://medium.com/u/21648547d748?source=post_page-----b9e0c1e905b8--------------------------------)、[米尔顿·辛巴·坎巴拉米](https://medium.com/u/34305067fe5c?source=post_page-----b9e0c1e905b8--------------------------------)、[马克西姆·库帕尼](https://medium.com/u/91d39083d3e9?source=post_page-----b9e0c1e905b8--------------------------------)、[玛丽-安妮·马文](https://medium.com/u/65e2d126b117?source=post_page-----b9e0c1e905b8--------------------------------)、[帕夫勒·马林科维奇](https://medium.com/u/e253e1c83d01?source=post_page-----b9e0c1e905b8--------------------------------)、[迪夫扬舒·拉杰](https://medium.com/u/5556e9ffcf23?source=post_page-----b9e0c1e905b8--------------------------------)、[德鲁夫·冈瓦尼](https://medium.com/u/6a64e46e3055?source=post_page-----b9e0c1e905b8--------------------------------)、 <https://medium.com/u/e253e1c83d01?source=post_page-----b9e0c1e905b8--------------------------------> [安娜·伊莎贝尔](https://medium.com/u/b2226422d029?source=post_page-----b9e0c1e905b8--------------------------------)，[阿维·舒拉](https://medium.com/u/5d33decdf4c4?source=post_page-----b9e0c1e905b8--------------------------------)，[埃里克·巴洛迪斯](https://medium.com/u/a09f1d009841?source=post_page-----b9e0c1e905b8--------------------------------)，[利洛·瓦格纳](https://medium.com/u/64bb3f6144f2?source=post_page-----b9e0c1e905b8--------------------------------)，[赛帕万·叶库拉](https://medium.com/u/3251c9fff04b?source=post_page-----b9e0c1e905b8--------------------------------)，[丹尼尔·里德斯通](https://medium.com/u/14d8d1f61342?source=post_page-----b9e0c1e905b8--------------------------------)，[雅各布·皮耶尼亚泽克](https://medium.com/u/6f0948d99b1c?source=post_page-----b9e0c1e905b8--------------------------------)，[玛丽·特朗](https://medium.com/u/4cfa1d0b321f?source=post_page-----b9e0c1e905b8--------------------------------)，[夏洛特·p .](https://medium.com/u/b5c229ea07f9?source=post_page-----b9e0c1e905b8--------------------------------)，[张子涵](https://medium.com/u/cdd192d40691?source=post_page-----b9e0c1e905b8--------------------------------)， [锡南·古尔特金](https://medium.com/u/db668a7751da?source=post_page-----b9e0c1e905b8--------------------------------)、[德弗什·拉贾迪亚克斯](https://medium.com/u/ff46afc9b9c2?source=post_page-----b9e0c1e905b8--------------------------------)、[伊森·克劳斯](https://medium.com/u/6dd1af751f3e?source=post_page-----b9e0c1e905b8--------------------------------)、[阿诺·卡皮坦](https://medium.com/u/95c6848a79aa?source=post_page-----b9e0c1e905b8--------------------------------)、[凯文·贝勒蒙特博士](https://medium.com/u/3dea771eb493?source=post_page-----b9e0c1e905b8--------------------------------)、[桑巴尔杰](https://medium.com/u/ff4c7293e945?source=post_page-----b9e0c1e905b8--------------------------------)、[埃拉·威尔逊](https://medium.com/u/88f10c2a3fea?source=post_page-----b9e0c1e905b8--------------------------------)、[萨迪克·巴丘](https://medium.com/u/a0654b43131a?source=post_page-----b9e0c1e905b8--------------------------------)、 如果你想在未来的月刊中看到你的名字，我们希望收到你的来信。****