# Python 强化学习实践介绍

> 原文：<https://towardsdatascience.com/hands-on-introduction-to-reinforcement-learning-in-python-da07f7aaca88>

## 通过教机器人走迷宫来理解奖励

![](img/a579c7076c9472a565b371bfa4af83a8.png)

照片由[布雷特·乔丹](https://unsplash.com/es/@brett_jordan?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

传统机器学习的最大障碍之一是，大多数有监督和无监督的机器学习算法需要大量数据才能在现实世界的用例中有用。即使这样，如果没有人类的监督和反馈，人工智能也无法学习。如果人工智能可以从头开始学习会怎样？

作为最著名的例子之一，谷歌的 DeepMind 构建了 [AlphaGo](https://www.deepmind.com/research/highlighted-research/alphago) ，它能够击败历史上最好的围棋选手 Lee Sedol。为了学习最佳策略，它使用了深度学习和强化学习的结合——就像在，通过与自己对弈成千上万的围棋游戏。李·塞多尔甚至说，

> 我认为 AlphaGo 是基于概率计算的，它只是一台机器。但是当我看到这个举动的时候，我改变了主意。毫无疑问，AlphaGo 很有创造力。

强化学习消除了对大量数据的需要，并且还优化了它可能在广泛的环境中接收的高度变化的数据。它密切模拟了人类的学习方式(甚至可以像人类一样找到非常令人惊讶的策略)。

更简单地说，强化学习算法由代理和环境组成。代理为环境的每个状态计算一些奖励或惩罚的概率。这个循环是这样工作的:给一个代理一个状态，代理向环境发送一个动作，环境发送一个状态和回报。

让我们试着编码一个机器人，它将试着在尽可能少的移动中通过一个 6×6 的迷宫。首先，让我们从创建代理和环境类开始。

# 环境和代理

我们希望我们的代理能够根据以前的一些经验来决定做什么。它需要能够基于一组给定的操作做出决策并执行一些操作。避免拟人化的定义代理是什么，以更严格地定义你的代理将有什么样的方法和功能——在强化学习中，代理不能控制的任何东西都是环境的一部分。

环境是代理可以与之交互的代理之外的任何事物，包括系统的状态。它不一定是您想象的完整环境，只要包括当代理做出选择时真正改变的东西。环境还包括你用来计算奖励的算法。

在名为`environment.py`的文件中，创建这个类:

```
import numpy as npACTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}class Maze(object):
    def __init__(self):
        # start with defining your maze
        self.maze = np.zeroes((6, 6))
        self.maze[0, 0] = 2
        self.maze[5, :5] = 1
        self.maze[:4, 5] = 1
        self.maze[2, 2:] = 1
        self.maze[3, 2] = 1 self.robot_position = (0, 0) # current robot position
        self.steps = 0 # contains num steps robot took self.allowed_states = None # for now, this is none
        self.construct_allowed_states() # not implemented yet
```

根据我们编写的代码，下面是我们的迷宫的样子(在我们的代码中，1 代表墙壁，2 代表机器人的位置):

```
R 0 0 0 0 X
0 0 0 0 0 X
0 0 X X X X
0 0 X 0 0 X
0 0 0 0 0 0 
X X X X X 0 <- here's the end
```

这是我们需要存储在环境中的核心信息。根据这些信息，我们可以在以后创建函数来更新给定动作的机器人位置，给予奖励，甚至打印迷宫的当前状态。

您可能还注意到，我们添加了一个`allowed_states`变量，并在其后调用了一个`construct_allowed_states()`函数。`allowed_states`将很快拥有一本字典，将机器人所处的每一个可能的位置映射到机器人可以从那个位置到达的可能位置的列表。`construct_allowed_states()`将构建此地图。

我们还创建了一个名为`ACTIONS`的全局变量，它实际上只是一个可能的移动及其相关翻译的列表(我们甚至可以省略方向标签，但它们是为了便于阅读和代码调试)。我们将在构建允许的州地图时使用它。为此，让我们添加以下方法:

```
def is_allowed_move(self, state, action):
    y, x = state
    y += ACTIONS[action][0]
    x += ACTIONS[action][1] # moving off the board
    if y < 0 or x < 0 or y > 5 or x > 5:
         return False # moving into start position or empty space
    if self.maze[y, x] == 0 or self.maze[y, x] == 2:
        return True
    else:
        return Falsedef construct_allowed_states(self):
    allowed_states = {}
    for y, row in enumerate(self.maze):
        for x, col in enumerate(row):
            # iterate through all valid spaces
            if self.maze[(y,x)] != 1:
                allowed_states[(y,x)] = []
                for action in ACTIONS:
                    if self.is_allowed_move((y, x), action):
                        allowed_states[(y,x)].append(action) self.allowed_states = allowed_statesdef update_maze(self, action):
    y, x = self.robot_position
    self.maze[y, x] = 0 # set the current position to empty
    y += ACTIONS[action][0]
    x += ACTIONS[action][1]
    self.robot_position = (y, x)
    self.maze[y, x] = 2
    self.steps += 1
```

这允许我们在迷宫实例化时快速生成状态到允许动作的地图，然后在机器人每次移动时更新状态。

我们还应该在环境中创建一种方法来检查机器人是否在迷宫的尽头:

```
def is_game_over(self):
    if self.robot_position == (5, 5):
        return True
    return False
```

现在我们准备为我们的代理开始上课。在名为`agent.py`的文件中，创建一个新类:

```
import numpy as npACTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}class Agent(object):
    def __init__(self, states, alpha=0.15, random_factor=0.2):
        self.state_history = [((0, 0), 0)] # state, reward
        self.alpha = alpha
        self.random_factor = random_factor

        # start the rewards table
        self.G = {}
        self.init_reward(states)
```

现在，很多内容看起来都不太熟悉，但这是一个介绍奖励算法的好时机，我们将使用该算法来培训我们的代理。

# 奖励

所有代理人的目标都是回报最大化。就像任何机器学习算法一样，奖励将采取数字的形式，根据某种算法而变化。代理将尝试估计其每个行动选择的损失，然后采取行动，然后从环境中获得行动的*真实*回报，然后调整其对该特定行动的未来预测。

我们将为我们的环境提出一个非常简单的奖励政策:机器人每走一步，我们罚-1 分(因为我们想要最快的解决方案，而不是任何解决方案)，然后当机器人到达终点时，奖励 0 分。因此，采取 20 个步骤的解决方案将奖励代理总共-20 分，而采取 10 个步骤的解决方案将奖励代理-10 分。我们奖励政策的关键是保持简单——我们不想过度监管我们的代理。

现在让我们将它编码到我们的环境中。将这些方法添加到您的`Maze`类中:

```
def give_reward(self):
    if self.robot_position == (5, 5):
        return 0
    else:
        return -1def get_state_and_reward(self):
    return self.robot_position, self.give_reward()
```

就是这样！

好吧，但是有一个问题——代理人怎么可能*预测*每一次行动会得到的回报？

## 情节剧

这里的目标是创建一个函数，在一集*中*(在我们的例子中，一集是一场游戏)为每个州的预期未来奖励建模。这些奖励随着代理经历更多的情节或游戏而不断调整，直到它收敛于环境给定的每个状态的“真实”奖励。例如，我们可能有这样一个状态表:

```
+--------+-----------------+
| State  | Expected Reward |
+--------+-----------------+
| (0, 0) | -9              |
| (1, 0) | -8              |
| ...    | ...             |
| (X, Y) | G               |
+--------+-----------------+
```

其中 G 是一个状态(X，Y)的给定期望报酬。但是我们的机器人将从一个随机化的状态表开始，因为它实际上还不知道任何给定状态的预期回报，并将试图收敛到每个状态的 G。

我们的学习公式是`*G_state = G_state +* α(target — *G_state*)` *。在实践中，在一集结束时，机器人已经记住了它所有的状态和相应的奖励。它也知道自己当前的 *G* 表。使用这个公式，机器人将根据这个简单的公式更新 *G* 表中的每一行。*

我们来分解一下。我们实际上是在给定状态下，增加了实际奖励(*目标*)和我们最初预期奖励之间的差额百分比。你可以把这个差额想象成*损失*。这个百分比被称为 alpha 或α，熟悉传统机器学习模型的人会将其视为学习率。百分比越大，它可能越快地向目标回报靠拢，但是它越有可能超过或高估真正的目标。对于我们的代理，我们将默认学习率设置为 0.15。

实现成功的奖励算法有多种方式，这都取决于环境及其复杂性。例如，AlphaGo 使用深度 q 学习，这种学习实现了神经网络，可以根据过去有益举动的随机样本来帮助预测预期回报。

让我们编码我们的算法。首先，我们需要一个函数来初始化我们的`Agent`类中的随机状态表，这个函数在代理初始化时被调用:

```
def init_reward(self, states):
    for i, row in enumerate(states):
        for j, col in enumerate(row):
            self.G[(j,i)] = np.random.uniform(high=1.0, low=0.1)
```

我们将 *G* 的随机值初始化为总是大于 0.1，因为我们不希望它将任何状态初始化为 0，因为这是我们的目标(如果一个状态最终从 0 开始，代理将永远不会从该状态中学习)。

其次，我们需要一种方法，允许代理在一集结束时“学习”G 的新值，给定该集的状态和奖励对(由环境给定)。将这个添加到`Agent`类中:

```
def update_state_history(self, state, reward):
    self.state_history.append((state, reward))def learn(self):
    target = 0 # we know the "ideal" reward
    a = self.alpha for state, reward in reversed(self.state_history):
        self.G[state] = self.G[state]+ a * (target - self.G[state]) self.state_history = [] # reset the state_history
    self.random_factor = -= 10e-5 # decrease random_factor
```

你会注意到我们在最后也把`random_factor`缩小了一点。我们来谈谈那是什么。

## 探索与利用

现在，代理人*可以*总是采取它认为会导致最大回报的行动。然而，如果一个行动，代理人估计会有最低的回报，最终却有最高的回报呢？如果一个特定行为的回报随着时间的推移会有回报呢？作为人类，我们能够估计长期回报(“如果我今天不买这部新手机，我就能在未来攒钱买车”)。我们如何为我们的代理复制这一点？

这就是通常所说的探索与利用的困境。一个总是利用(比如，总是采取它预测会有最高回报的行动)的代理人可能永远不会找到解决问题的更好的办法。然而，一个总是探索的代理(总是随机选择一个选项来看它通向哪里)将会花很长时间来优化自己。因此，大多数奖励算法将结合使用探索和利用。这是我们的`Agent`课程中的`random_factor`超参数——在学习过程的开始，代理将探索 20%的时间。随着时间的推移，我们可能会减少这个数字，因为随着代理的学习，它可以更好地优化利用，我们可以更快地收敛到一个解决方案。在更复杂的环境中，您可以选择保持相当高的探索率。

现在我们知道了我们的机器人会如何选择一个动作，让我们把它编码到我们的`Agent`类中。

```
def choose_action(self, state, allowed_moves):
    next_move = None n = np.random.random()
    if n < self.random_factor:
        next_move = np.random.choice(allowed_moves)
    else:
        maxG = -10e15 # some really small random number
        for action in allowed_moves:
            new_state = tuple)[sum(x) for x in zip(state, ACTIONS[action])])
            if self.G[new_state] >= maxG:
                next_move = action
                maxG = self.G[new_state] return next_move
```

首先，我们根据我们的`random_factor` 概率随机选择探索或利用。如果我们选择探索，我们从给定的允许移动列表中随机选择我们的下一步移动(传递给函数)。如果我们选择利用，我们循环遍历可能的状态(给定 allowed_moves 列表)，然后从 *G* 中找到期望值最大的一个。

完美！我们已经完成了代理和环境的代码，但是我们所做的只是创建类。我们还没有检查每一集的工作流程，也没有让代理人学习的方式和时间。

# 把所有的放在一起

在我们代码的开始，在创建了我们的代理和迷宫之后，我们需要随机初始化 *G* 。对于我们想玩的每一个游戏，我们的代理应该从环境中获得当前的状态奖励对(记住，除了最后一个方块，每个方块都是-1，应该返回 0)，然后用它选择的动作更新环境。它将从环境中接收一个新的状态-奖励对，并在选择下一个动作之前记住更新的状态和奖励。

在一集结束后(迷宫无论走了多少步都完成了)，代理应该查看该游戏的状态历史，并使用我们的学习算法更新其 *G* 表。让我们用伪代码描述一下我们需要做什么:

```
Initialize G randomly
Repeat for number of episodes
 While game is not over
  Get state and reward from env
  Select action
  Update env
  Get updated state and reward
  Store new state and reward in memory
 Replay memory of previous episode to update G
```

您可以在以下要点中看到这一点的完整实现:

在这段代码中，我们要求代理玩游戏 5000 次。我还添加了一些代码来绘制机器人在 5000 次游戏中每次完成迷宫所需的步数。尝试以不同的学习速率或随机因素运行代码几次，比较机器人收敛到 10 步解决迷宫需要多长时间。

另一个挑战是尝试打印出机器人完成迷宫的步骤。在`Maze`类中的方法`print_maze()`中有一些起始代码(下面显示了完整的类)，但是您需要添加代码，从代理接收 state_history 并在打印函数中格式化它，比方说，作为 R，用于每一步。这将允许您查看机器人最终决定的步骤-这可能很有趣，因为在我们的迷宫中有多条十步的路线。

环境和代理的完整代码如下。

*我要感谢菲尔·泰伯的精彩课程《动态强化学习》。*

**Neha Desaraju 是德克萨斯大学奥斯丁分校学习计算机科学的学生。你可以在网上**[**estau dere . github . io**](https://estaudere.github.io)**找到她。**