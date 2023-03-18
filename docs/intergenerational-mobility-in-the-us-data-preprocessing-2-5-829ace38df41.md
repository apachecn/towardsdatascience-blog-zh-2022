# 美国的代际流动——数据预处理(2/5)

> 原文：<https://towardsdatascience.com/intergenerational-mobility-in-the-us-data-preprocessing-2-5-829ace38df41>

> 本系列由 Ibukun Aribilola 和 Valdrin Jonuzi 共同撰写，是社会公益数据科学教程的一部分。

在跨越了数据收集的障碍之后，我们得到了一系列不同形式的数据，这些数据的原始形式不适合我们的预期分析。为了解决这个问题，我们将合并数据集并解决丢失数据的问题。因为我们所有的数据集都有县级数据，所以我们将根据县 id 进行合并。

# 合并单个数据集

这个项目的数据合并可以分为三个阶段。我们从合并单个机会图谱数据集开始，以创建一个大的机会图谱数据集。然后，我们将 Opportunity Atlas 数据集与 Data Commons 数据集合并，后者又与 Chetty & Hendren 数据集合并。

# 合并机会图谱数据集

在“收集数据”部分，我们描述了如何从机会地图集下载七个特征的县和区域级数据。正如在“数据预处理”一节的介绍中提到的，我们将只使用县级数据，因此我们将在这一节中介绍七个县级数据集的合并。

首先，我们需要导入所有必要的库，然后我们将各个数据集从 Google Drive 加载到 Python 笔记本中。

```
import pandas as pd
import functools*# County-level data*
*## Household income at age 35*
c_household_income = pd.read_csv('https://drive.google.com/uc?id=1dRLKhKqBjs2ARcynSUT99Y8CoJPoC5IZ')
*## Incerceration rate*
c_jail_rate = pd.read_csv('https://drive.google.com/uc?id=14K-QJ_bZi8Dtvod9J_s72tLE8TpWf25T')
*## Fraction married at age 35*
c_married = pd.read_csv('https://drive.google.com/uc?id=1MY_ulJnGFSBotgDtbf4dSYRR_SqW5rNq')
*## Employment rate at age 35*
c_working = pd.read_csv('https://drive.google.com/uc?id=1sZxD3OqA0IqbiDn1L22HXHfFE3J_Q6yX')
*## High school graduation rate*
c_hs = pd.read_csv('https://drive.google.com/uc?id=1cOFWFjPNyYn-dBdBH5s8PZ7V8B1ZNpC9')
*## College graduation rate*
c_coll = pd.read_csv('https://drive.google.com/uc?id=1CquuVRG_c-7wgW6P7yjXL6uUxt3VBb2v')
*## Poverty rate in 2012-16*
c_poor = pd.read_csv('https://drive.google.com/uc?id=166vtYSczyKAHMXP1qbjaMZBNIZlO3DjB')
```

当我们使用`df.shape`命令检查数据集的形状时，我们发现它们有 218 行，其中两行是县 ID 和名称。剩余的 216 个是 6 个父母收入百分位数水平、6 个种族亚组、3 个儿童性别亚组和 2 个儿童群组亚组的组合(6×6×3×2 = 216)。

我们还注意到,“c_poor”数据集中的“county name”列是以标题“name”的形式书写的，而其他数据集中的列名是以小写“Name”标记的。因为 Python 是一种区分大小写的语言，所以我们希望确保所有数据集中的 county name 列的标签都是相似的，所以我们只需更改列名。

```
c_poor = c_poor.rename(columns = {"Name": "name"})
```

现在，所有的数据集都可以使用 Pandas 的“合并”功能进行合并了。我们将合并县名和 ID 上的数据集，以便大型数据集将包含唯一的县名和 ID(没有重复)。

```
*# Make a list of all the datasets that need to be merged*
c_datasets = [c_household_income,
        c_jail_rate,
        c_married,
        c_working,
        c_hs,
        c_coll,
        c_poor]*# Batch merging the datasets into one*
oa_county_dataset = functools.reduce(lambda  left, right: 
                        pd.merge(left, right, on = ["cty", "name"], how = "outer"), c_datasets)
```

现在可以将合并的数据集导出到您选择的文件夹中。(点击[此处](https://share.streamlit.io/ibukunlola/dssg_final_project/main/finalised_notebooks/DataCleaning/data_preprocessing.py)查看完整的牌桌。)

合并后的数据集包含多达 1300 列，这意味着可能有太多我们不需要的变量。快速检查显示，大多数列是不同收入百分位数的居民的数据条目和码本中未描述的“晚期群体”的数据条目。因为我们想要删除的变量比保留的要多，所以我们可以列出想要保留的变量。对于这个项目，我们感兴趣的是整个样本、种族和性别分组的收入、结婚率、就业率、高中毕业率、大学毕业率和贫困率。

```
oa_data.info()
*# the output of the cell shows that there are 1300 columns*
```

我们使用`df.columns.difference`命令来选择 opportunity atlas 数据集的所有列，通过列出我们想要保留的列来删除这些列。因此，如果你想选择一组不同的变量来进行分析，在这一阶段必须将它们列出来。然后，我们使用`df.drop`命令删除那些不需要的变量。

# 数据共享空间+机会图谱

在为 Opportunity Atlas 数据创建单个数据集并对其进行修整以仅包含我们进行分析所需的变量之后，我们将把它与 Data Commons 数据合并。两个数据集都有一个专用于县 ID 的列，即 Opportunity Atlas 数据集中的`cty`列和 Data Commons 数据集中的“地点”列。我们希望将跨数据集的每个县的属性合并到一个数据集，因此我们将基于县 id 进行合并。为此，两个县 ID 列必须采用相同的格式。目前，县 Id 以不同的格式书写，因为它们以不同的字母为前缀，即“ctyxxxxx”和“geoId/xxxxx”。我们可以通过分割每个字符串并删除前缀来确保县 id 具有相同的格式。我们还将每个数据集的县 ID 变量重命名为“geo_id ”,以确保变量名称在数据集之间保持一致。

```
*# slice the elements of the 'cty' and 'place' columns in oa_data and dc_data, respectively*
*## the goal is to isolate the geo IDs of each county*
cty_new = []
place_new = []
for i in oa_data['cty']:
  cty_new.append(i[3:])for j in dc_data['place']:
  place_new.append(j[6:])*# replace the geo ID columns without surrounding text*
oa_data['cty'] = cty_new
dc_data['place'] = place_new
oa_data = oa_data.rename(columns={"cty":"geo_id"})
dc_data = dc_data.rename(columns={"place":"geo_id"})
```

有了一致的县 ID 格式，我们就可以使用`pd.merge`命令合并 Opportunity Atlas 和 Data commons 数据集了。我们将对`geo_id`列进行外部合并，合并两个数据集中的县 id，这样两个数据集中表示的每个县在合并的数据集中只出现一次。查看 [pd.merge 文档](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html)以了解合并数据集的不同方法。

```
*# merge the datasets*
merged_data = pd.merge(oa_data, dc_data, how="outer", on=["geo_id", "geo_id"])
merged_data = merged_data.drop(columns="name_y")
merged_data = merged_data.rename(columns = {"name_x": "name"})
merged_data.geo_id = merged_data.geo_id.astype(str)
merged_data
```

(点击[此处](https://share.streamlit.io/ibukunlola/dssg_final_project/main/finalised_notebooks/DataCleaning/data_preprocessing.py)查看完整表格。)

# Chetty & Hendren +数据共享+机会图谱

合并过程的最后一步是将 Chetty & Hendren 的数据集与之前合并的数据共享空间和机会地图集数据集合并。为了做到这一点，我们进一步削减了每个数据集中的变量列表。例如，我们放弃了所有与永久居民代际流动相关的变量，因为我们对永久居民和流动者之间的区别不感兴趣。

```
IGM = pd.read_stata("../data/raw/online_table4.dta")
df = pd.read_csv("../data/processed/merged_df.csv")*# Remove permanent resident intergenerational mobility from Chetty's 2014* 
*# because we aren't interested in the permanent residents vs movers distinction*
perm_res_IGM = list(filter(re.compile("perm_res_.*").match, IGM.columns))
IGM = IGM.drop(columns=["csa", "csa_name", "cbsa", "cbsa_name", "intersects_msa"]+ perm_res_IGM)
```

我们还从合并的(数据共享+机会地图集)数据集中删除了一些变量，如所有种族和性别子群的县数据。

最后一个预合并步骤是确保县 ID 列在数据集之间的命名一致。如果列名不一致，我们只需重命名其中一个以匹配另一个。

现在，数据集已准备好进行合并和导出。

```
df_subset.geo_id = df_subset.geo_id.astype(int)
df_subset.rename(columns={"geo_id":"cty2000"}, inplace=True)merged_data = pd.merge(IGM, df_subset, on=["cty2000", "cty2000"])
merged_data = merged_data.drop(columns=['name'])merged_data
```

(点击[此处](https://share.streamlit.io/ibukunlola/dssg_final_project/main/finalised_notebooks/DataCleaning/data_preprocessing.py)查看完整表格。)

# 电报密码本

这是一条漫长的道路，但我们最终获得了一个全面的数据集，并很高兴开始分析。然而，我们必须创建一个数据码本，定义数据集中的每个变量，并使浏览数据集的任何人都容易理解每个变量的含义。

当构造码本时，有不同的格式可供选择。例如， [Chetty](http://www.equality-of-opportunity.org/data/neighborhoods/online_table4.pdf) 混合使用了第 1-3 页的列表式变量描述和第 4-5 页的表格格式。[这里的](https://github.com/valdrinj/dssg_final_project/blob/main/data/processed/codebook.csv)是我们在这个分析中使用的数据集的码本。

# 因变量选择

当前的数据集包含一个结果变量的洗衣清单，所有这些我们都不一定感兴趣。因此，我们去掉了所有变量，只有一个除外——搬到一个县对 26 岁时父母收入排在第 75 百分位的孩子的收入排名的因果影响(“因果 _p75_cty_kr26”)。虽然这是我们分析感兴趣的结果变量，但还有其他有趣的结果变量值得探索，所以不要犹豫，进一步探索数据吧！

```
IGM_df = pd.read_pickle("../data/processed/IGM_merged_df.pkl")*# Drop irrelevant columns*
other_causal = list(filter(re.compile("causal_.*").match, IGM_df.columns))
perm_res = list(filter(re.compile("perm_.*").match, IGM_df.columns))
df = IGM_df.drop(columns=other_causal+perm_res+ ['cty1990', 'cz_name', 'cz_pop2000', 'csa', 'csa_name', 'stateabbrv',
       'cbsa', 'cbsa_name', 'intersects_msa'])*# Select the outcome variable*
pred = "causal_p75_cty_kr26"pred_df = df.copy() 
pred_df.insert(0, pred, IGM_df[pred])
```

另一个数据预处理最佳实践是指定丢失数据问题的处理方式。在该数据集中，我们以分层格式处理缺失值。如果缺少与某个县相关的特定值，我们将输入通勤区域中值。如果通勤区中间值不存在，我们插入州中间值，如果州中间值为空，我们输入国家中间值。

```
df_cz_median = df.groupby("cz").median()
df_state_median = df.groupby("state_id").median()
df_country_median = df.median()*# Impute by commuting zone median*
for col in df.columns[np.where(df.isna().sum()>0)]:
    df[col] = df.apply(
        lambda row: df_cz_median.loc[row['cz']][col] if np.isnan(row[col]) else row[col],
        axis=1
    )*# Impute by state median*
for col in df.columns[np.where(df.isna().sum()>0)]:
    df[col] = df.apply(
        lambda row: df_state_median.loc[row['state_id']][col] if np.isnan(row[col]) else row[col],
        axis=1
    )*# Impute with country median*
for col in df.columns[np.where(df.isna().sum()>0)]:
    df[col] = df.apply(
        lambda row: df_country_median[col] if np.isnan(row[col]) else row[col],
        axis=1
    )
```

替换 NA 值后，3221 行中仍有 767 行缺少结果变量值，因此我们将它们与其他不必要的列一起删除，并导出清理后的数据集。

# 结论

从变量选择到数据清理，许多必要的工作都是为了准备用于数据分析的数据集，每个步骤都必须仔细完成并记录在案，以便于与受众顺利沟通。在下一篇文章中，我们将解释如何执行探索性数据分析来理解数据，并将其用作对数据进行头脑风暴分析的工具。