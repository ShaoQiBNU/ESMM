[TOC]

# 阿里巴巴ESMM模型解读

## 背景

> 在推荐系统领域（尤其是电商），准确估计点击后的转化率(CVR)对items进行排名至关重要，CTR预估聚焦于点击率的预估（广告系统的目标），而电商领域user点击之后下单消费才是终极目标（用户的行为遵循一个序列模式：曝光->点击->转化）。请注意，这里的预测目标是假设商品被点击之后的转化率，即pCVR=p(conversion|click, impression)，而不是转化率。二者是有区别的：一个商品的点击率很低，但是一旦被点击之后，用户购买的概率可能非常高。常规的CVR模型采用深度学习方法进行建模，取得了非常好的效果。但是在实践中也遇到几个问题，这使CVR建模具有挑战性。主要问题如下：

- **Sample Selection Bias，SSB**

  > 样本选择偏差问题。传统CVR的预估模型是在曝光和click的数据上训练，在inference的时候使用了整个样本空间，如图所示。训练样本和实际数据不服从同一分布，不符合机器学习中训练数据和测试数据独立同分布的假设。直观的说，会产生转化的用户不一定都是进行了点击操作的用户，如果只使用点击后的样本来训练，会导致CVR学习产生偏置。如图所示：

  Img1

- **Data Sparsity，DS**

  > 数据稀疏问题。点击样本在整个样本空间中只占了很小一部分，而转化样本更少，高度稀疏的训练数据使得模型的学习变得相当困难。Table1显示了这个问题。

  Img2

## 解决方法

> 针对以上两个问题，目前也有了一些解决方案如下：

- Estimating conversion rate in display advertising from past erformance data.

  > 建立了基于不同特征的分层估计器，并将其与逻辑回归模型相结合来解决DS问题。 但是，它依靠先验知识来构造层次结构，这很难在具有数千万用户和项目的推荐系统中应用。 

- Oversampling method

  > 通过复制稀缺类的样本来缓解DS问题，但对采样率敏感。

- AMAN

  > 应用随机抽样策略来选择未点击的展现作为负样本，通过引入未知样本，可以在某种程度上消除SSB问题，但会导致模型低估。

- Unbiased method

  > 通过剔除抽样来拟合观测值的真实基础分布，但是，通过拒绝概率的除法对样本加权时，可能会遇到数值不稳定性。

**总之，CVR建模场景下存在的SSB问题或DS问题均没有得到有效解决，并且上述方法均没有用到序列行为信息。**

## ESMM模型

> 为了能够解决SSB和DS的问题，并且分利用用户行为的序列模式，阿里提出了ESMM模型。在ESMM中，引入CTR和CTCVR两个辅助任务。与直接在点击后的展现样本上训练CVR模型不同，ESMM把 pCVR作为一个中间变量，即 pCTCVR=pCVR*pCTR。pCTCVR和pCTR都在整个展现样本空间上进行预估，使得pCVR也在整个空间上进行预估。此时，消除了SSB问题。另外，CVR网络的特征表征参数与CTR网络共享。由于CTR网络使用更丰富的样本进行训练，这种参数迁移学习有助于明显缓解DS问题。具体如下：

- CTR、CVR和CTCVR的关系

  > CTR表示点击率、CVR表示假设商品被点击后的转化率、CTCVR表示商品被点击并且成功转化。三者的关系如下：

Img3

> 其中<a href="https://www.codecogs.com/eqnedit.php?latex=z" target="_blank"><img src="https://latex.codecogs.com/svg.latex?z" title="z" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=y" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y" title="y" /></a>分别表示conversion和click。
>
> 注意到，在全部样本空间中，CTR对应的label为click，而CTCVR对应的label为click & conversion，**这两个任务是都可以使用全部样本今夕预估**。**所以ESMM通过这学习两个任务，再根据上式隐式地学习CVR**，具体结构如下：

Img4

> ESMM模型分为两个子模型，pCTR和pCVR：两者共享特征的embedding层，从concatenate之后各自学习参数；pCVR仅仅是网络中的一个variable，没有监督信号，只有pCTR和pCTCVR才有监督信号进行学习。ESMM的loss函数定义如下：

img5

> 即采用CTR和CTCVR的监督信息来训练网络，隐地学习CVR。直观理解就是，只要ctr预估准确了，并且pCTCVR也预估准确了，因为pCTCVR = pCTR * pCVR，所以pCVR的预估也一定是准确的。

> 此外，ESMM的结构是基于“乘”的关系设计，是不是也可以通过“除”的关系得到pCVR，即 pCVR = pCTCVR / pCTR ？例如分别训练一个CTCVR和CTR模型，然后相除得到pCVR。其实也是可以的，但这有个明显的缺点：真实场景预测出来的pCTR、pCTCVR值都比较小，“除”的方式容易造成数值上的不稳定。具体的实验结果在下面介绍。

## 实验设置和比较

### 数据

> 由于没有公开的数据，作者从淘宝日志中抽取整理了一个数据集Product，并开源了从Product中随机抽样1%构造的数据集[Public](https://tianchi.aliyun.com/datalab/dataSet.html?dataId=408)（约38G）。

### 实验设置

#### 模型对比

- Base

  ESMM模型图左侧的CVR结构，训练集为点击

- AMAN

  采用负采样策略，采样比例设置为10%，20%，50%和100%

- OVERSAMPLING

  复制正样本来减少训练数据的稀疏性

- UNBIAS

  rejection sampleing，pCTR作为rejection probability

- DIVISION

  预测pCTR和pCTCVR，然后相除得到CVR

- ESMM-NS

  CVR与CTR部分不share embedding。

- ESMM

#### ESMM模型参数

> ESMM模型和Base模型采用相同的网络结构和超参数，1) ReLU激活函数；2) embedding维度设置为18；3) MLP网络结构设置为 360x200x80x2；4) 使用adam优化方法

#### 评价指标

- CVR预估

  在数据集的点击样本上，计算CVR的AUC；

- CTCVR预估

  每个模型训练预测得到pCVR，同时，单独训练一个和BASE一样结构的CTR模型。除了ESMM类模型，其他对比方法均以pCTR*pCVR计算pCTCVR，在全部样本上计算CTCVR的AUC。

#### 训练集和测试集

> 按时间分割，1/2数据训练，其余测试

### 实验结果

> ESMM模型表现出了最优的效果，其充分解决了SSB和DS的问题。在Product数据集上，各模型在不同抽样率上的AUC曲线如图所示，ESMM显示的稳定的优越性，曲线走势也说明了Data Sparsity的影响还是挺大的。

Img

img

## 讨论

- ESMM 根据用户行为序列，显示引入CTR和CTCVR作为辅助任务，“隐式” 学习CVR，从而在完整样本空间下进行模型的训练和预测，解决了CVR预估中的2个难题。
- 可以把 ESMM 看成一个**新颖的 MTL 框架**，其中子任务的网络结构是可替换的，当中有很大的想象空间。