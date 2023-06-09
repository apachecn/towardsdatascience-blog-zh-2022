# 我会抽一张战斗 VIP 通行证吗？探索神奇宝贝交易卡游戏的可能性

> 原文：<https://towardsdatascience.com/will-i-draw-a-battle-vip-pass-exploring-the-probabilities-of-the-pok%C3%A9mon-trading-card-game-6b0b60b838c7>

许多纸牌游戏都涉及运气。不管一个玩家或一副牌有多好，拥有或不拥有优势都可能为轻松获胜或令人难忘的失败铺平道路。这条规则适用于神奇宝贝交易卡牌游戏(PTCG)。除了是我们有些人喜欢的生物牌，PTCG 是一个复杂和充满竞争的游戏，有丰富的机制，选项和机会，纯粹的技能不会让你走那么远。

其中一张卡叫做**战斗 VIP 通行证** (BVP)，我不确定我是爱它还是恨它。BVP 是竞争中的一张王牌。这是一张物品训练卡，可以让你在你的牌组中搜寻最多 2 个基本神奇宝贝，并将它们放在你的长凳上。然而，它有一个很大的缺点:你只能在第一个回合使用它。

![](img/eab3faf9cce0a0751b5775260b6b85a5.png)

战斗 VIP 通行证。插画师:(https://www.pokemon.com/us/pokemon-tcg/pokemon-cards/上田良？specialty artist = Ryo % 20 Ueda)。我拍的照片。

上周末玩的时候，我处在一个只有拿到这张牌才能赢的位置。所以我的第一次转变来了；我抽了一张牌，你猜怎么着？不是 BVP。如果这还不够的话，在第二场比赛中，我处于一种我会喜欢拥有那张牌的情况。

一副神奇宝贝牌有 60 张牌，我有四份 BVP 的拷贝。游戏开始时，每个玩家抽七张牌，每回合开始时抽一张。所以你用八张牌开始一场比赛，然而，在两场比赛中，我没有 BVP。在我的沮丧中，我想要答案；关于我为什么没有画 BVP 的答案。这个答案的问题是，“在第一轮抽中 BVP 的概率是多少？”

我使用**超几何分布**函数(图 1)来解决我计算的概率。这个分布回答了以下类型的问题:“如果我在一个有 *N* 个东西的桶中有 *K* 个东西，如果我从桶中取出 *n* 个东西，我得到这些东西的 *k* 个的机会有多大？”除了有助于确定神奇宝贝的概率之外，这种分布通常有助于确定没有替换的成功概率。它类似于数据科学中最常见的分布之一——二项式分布，在这种分布中，您使用替换进行采样。在我的例子中，考虑到我的一副 60 张牌中有 4 张，我想知道我第一手牌恰好抽到一张 BVP 的概率。用数学的方法来说，并解释维基百科的定义，超几何分布“描述了在 N 次抽奖中 K 次成功的概率，从大小为 N 的有限群体中，正好包含 K 个具有该特征的对象。”

![](img/ceca7c1f973f1c61c63004eeff8029fc.png)

图 1:超几何分布

# 拿到我第一手牌的概率

让我们从最简单的情况开始:我最初的手牌中至少有一张 BVP 的概率。在比赛开始时，每个玩家抽七张牌。你想在这手初始牌中拿到一张 BVP，因为你只能在第一回合使用它。让我们使用图 1 所示的超几何分布来计算概率。但我不想只给出数字，而是想一步一步地解释这个过程。

等式的右边是我们想要解决的问题。参数`K`代表群体中有多少感兴趣的对象。这里的值是 **4** ，因为这是我在我的牌组中运行的 BVP 数。接下来是`k`，我们希望获得的对象数量。在这种情况下，值是 **1** ，因为我想找到至少一个 BVP。`N`是人口的大小，或者说我的神奇宝贝套牌的大小，也就是 **60** 。最后，还有`n`，我要抽的牌数。在这个例子中，这个值是 **7** ，因为这是游戏开始时你抽的牌的数量。图 2 显示了插入了值的公式。我将跳过这个计算，因为它涉及几个步骤和巨大的数字。如果你很好奇，谷歌一下“二项式系数”，这是那些带数字的垂直括号的名字。同时，你可以在图 3 中看到我的涂鸦。

![](img/2ecca2a42e06970410a8d2e0e07dfabd.png)

图二。求解公式的第一步。

![](img/b926efc8f297899a3212127748f9de5f.png)

图 3。一次尝试。

我初始手牌中至少抽到一张 BVP 的概率是 **33.63** %。我认为由于这张牌对我不好，我的机会更低——但我可能会忘记所有我赢的次数，所以我不会抱怨太多。

我的牌组有四个 BVP，而不是一个，理想情况下，我更愿意抽**两个** BVP 来开始这场比赛，我的牌桌上至少有四个神奇宝贝。那么，这个事件发生的概率是多少？现在，我将使用 Python 编程语言编写一个小的计算机脚本来计算抽中两张以及零张、三张和四张战斗 VIP 通行证的概率，而不是使用计算器和我的大脑来解决这个问题。我将分享下面的脚本给那些可能觉得有用的人。

```
from scipy.stats import hypergeom
import numpy as np

# Supress scientific notation
np.set_printoptions(suppress=True)

def calculate(M, n, N):
    [M, n, N] = [M, n, N]
    rv = hypergeom(M, n, N)
    x = np.arange(0, n+1)
    return rv.pmf(x) * 100

calculate(60, 4, 7)
```

这个脚本输出`[60.05, 33.63, 5.93, 0.38, 0.01 ]`，分别是抽取零、一、二、三或四个 BVP 的概率。拥有所有值的好处是，我们可以将它们相加来计算累积概率，就是这样，获得至少或最多 *N* 张牌的概率。比如抽两个 BVP 的概率是 **5.93%** ，但是抽两个以上的概率是 **6.32%** (最后三个值之和)。下面的折线图(图 4)显示了概率。

![](img/6b7835aeb141a132f116b5fe96befab0.png)

图 4。可视化的概率。

# 画我的第一张卡片

在抽取了最初的七张牌并设置了六种奖品后，我们开始游戏。每个玩家从抽一张牌开始他们的回合，给我们另一次机会获得 BVP。不像上一个例子，我们从 60 个人中抽牌，现在我们从 47 个人中抽牌(60–7–6)。所以，假设我们在起手牌中没有得到任何 BVP，我们的新参数集是 *K* = 4， *k* = 1， *N* = 47， *n* = 1，这等于概率为 **8.51** %。

# 如果我第二个开始

先开始的缺点是你不能出支持者，一种你只能轮流出一次的牌，因为他们很强大。在这些支持者的卡片中，有一张是我打的，它叫伊里达。这张教练卡允许玩家在其牌组中搜寻任何水中神奇宝贝和物品卡。BVP 是一个物品，所以我可以从伊里达那里抓一个来增加我在第一回合玩的机会。

我想讲的最后一个场景是从第二局开始，在我最初的手牌中没有抽到 BVP 或伊里达，然后抽到了轮到我的牌。在这种情况下，我希望这张抽到的牌是 BVP 或伊里达，这样我就可以搜索 BVP 了。所以，我的新目标是拿到这八张牌中的一张( *K* =8)，而不是四张中的一张。这种情况发生的概率是 17.02%，给了我一个额外的机会在游戏早期看到这张广受好评的卡片。

![](img/8d3823def2230c5d1edec9e9191f8c04.png)

BVP 和艾瑞达。我拍的照片。

# 概述

运气影响神奇宝贝交易卡游戏。你可能在玩一场完美的游戏，知道你离胜利只有一步之遥，直到你的对手抽到了一张让比赛变得对自己有利的牌。这篇文章关注的是一张可以在第一回合就决定游戏结果的牌。问题中的卡片是战斗 VIP 通行证，它让你只能在你的第一个回合用神奇宝贝填满你的长椅。

利用超几何分布，我计算了在三种不同情况下抽中 BVP 的概率:在你的第一手牌中抽中，第一次抽中，或者第一回合抽中，假设你的牌组中有 Irida。在第一种情况下，你最初拿到 BVP 的概率是 33.63%。第二种情况，我称之为后备选择，有 8.51%的可能性。最后，还有一个涉及 Irida 的复杂场景，这是一张卡，它的效果让你可以搜索 BVP，并将使用它的几率增加到 17.02%，但前提是你愿意选择第二种。

还有成百上千的场景我没有涉及到。它们包括使用让你抽额外牌的牌，其他让你搜索牌的牌，以及让你用牌组顶上的牌替换你的牌的牌。这张卡会是什么？只有数字知道。现在，我将祈求好运，期待一个令人满意的结果。