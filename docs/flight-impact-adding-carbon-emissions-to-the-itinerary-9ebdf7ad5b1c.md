# 飞行影响:将碳排放加入旅程

> 原文：<https://towardsdatascience.com/flight-impact-adding-carbon-emissions-to-the-itinerary-9ebdf7ad5b1c>

# 飞行影响:将碳排放加入旅程

## 构建一个交互式应用程序，为关注气候的旅行者提供支持

![](img/a3defbc227b7e1b681913f17f7a84b54.png)

克里斯·莱佩尔特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

对许多美国人来说，航空旅行是生活中根深蒂固的一部分。无论是去看望家人度假，参加会议，还是快速休假，我们中的许多人在准备起飞时都没有考虑到我们的飞行可能会有非货币成本。我们的旅行计划中经常缺少的是对旅行碳排放的调查。虽然大多数旅行者没有考虑到环境影响，但欧洲运输和环境联合会警告说，随着航空排放到 2050 年增加一倍或两倍，该行业可能会消耗全球碳预算的 1/4。

关于为什么关注个人的“碳足迹”不是遏制气候危机的有效方法，有许多非常有见地的论点。然而，航空业确实提供了一个有趣的机会。2018 年，只有 [11%的全球人口](https://time.com/6048871/pandemic-airlines-carbon-emissions/)乘坐过飞机，而仅仅 [1%的人却承担了 50%](https://time.com/6048871/pandemic-airlines-carbon-emissions/) 的航空排放。这意味着这个小群体中的个体有一种独特的能力，可以通过相对较小的行为变化产生影响。

带着这个想法，我着手我的最后一个项目 [METIS](https://medium.com/u/d18bcb7f0383?source=post_page-----9ebdf7ad5b1c--------------------------------) 数据科学训练营，希望建立一些东西，让消费者能够自我教育，并在旅行中做出明智的决定。就这样，飞行冲击诞生了！(点击本文底部链接查看！)

## 这项任务

在这个项目中，我和我的同学们面临着建立一个完整的数据管道的挑战，从 API 到交互式 web 应用程序。Flight Impact 背后的管道如下:调用 [Climatiq API](https://climatiq.io/) (CC BY-SA 4.0)，清理 Google Colab 中的数据，存储在 MongoDB 中，最后在一个. py 文件中操纵它，该文件用于从 Github 启动我的 Streamlit 应用程序。

Climatiq API 提供了从客运到航运等各种活动的排放量计算。当给定两个机场的组合时，API 返回目的地之间单程的单个经济舱乘客的二氧化碳排放量(以千克为单位)。我能够指定我想要使用的计算标准，并且我选择了美国环保局对短程、中程和远程飞行的计算方法，以确保可重复性。使用世界大型机场(每年服务数百万乘客)的列表和 Python 请求库，我自动化了 180，000 对全球目的地的 API 请求。

接下来，我在 Google Colab 中清理了我的数据，添加了一些对前端用户体验有用的功能，如机场的完整国家名称(而不仅仅是一个代码)和航班排放量的计算(以吨为单位)。在 Google Colab 上，我使用 pymongo 将清理后的数据插入 MongoDB 数据库，该数据库托管在 [CleverCloud](https://www.clever-cloud.com/) 上。

## 在 Streamlit 中构建应用

管道的最后一步是在一个. py 文件中构建飞行影响的用户体验。 [Streamlit](https://streamlit.io/) ，一个将 Python 数据脚本转化为 web 应用的平台，允许你直接从 Github repo 中的文件启动应用。你只需要在代码中添加一些 Streamlit 命令来指定特性，比如下拉菜单和地图([这里有一个链接](https://github.com/ninaksweeney/flight_emissions/blob/main/flight_emissions_app.py)指向我的 Streamlit。py 文件，如果你正在寻找一个例子)。在这个过程中，我学会了不要低估 Streamlit 的缓存特性的重要性。每当用户改变页面上的某些内容时，Streamlit 都会重新运行脚本，但是在函数之前添加一个简单的@st.cache 命令会告诉应用程序*仅当输入或函数本身发生变化时才重新运行该函数*。这有助于应用程序运行更快，使用更少的内存。

## 简介:飞行冲击

当用户访问 Flight Impact 时，我们的目标是让他们能够探索自己的选择，并以更明智的消费者身份离开。用户可以:

*   查看所有从其所在城市出发的全球航班，以及每个航班的相关碳排放量
*   按出发国家或城市、目的地国家或城市或排放限制进行过滤，以比较各种路线
*   根据他们的路线是否已经确定，查看影响较小的替代方案或[使他们选择的航班购买更省油的方式](https://grist.org/guides/2021-holiday-makeover/6-habits-of-highly-effective-climate-conscious-travelers/)，例如经济舱、避免中途停留、只乘坐满员航班

当个人在寻找参与解决气候危机的方法时，我们的旅行习惯可以发挥作用。展望未来，让碳排放成为旅行计划过程中与获得靠窗座位同样重要的一部分。

自己试试这个应用[这里](https://share.streamlit.io/ninaksweeney/flight_emissions/main/flight_emissions_app.py)！注意:此链接使用 Streamlit 的免费版本，一段时间后可能会超过使用限制。如果是这样的话，下面的视频也展示了飞行撞击体验。

*要深入了解我对这个项目的代码或演示，请参见* [*项目回购*](https://github.com/ninaksweeney/flight_emissions) *。*