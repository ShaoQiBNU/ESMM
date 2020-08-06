[TOC]

阿里巴巴ESMM模型系列解读
============

# ESMM模型

## 背景

> 在推荐系统领域（尤其是电商），准确估计点击后的转化率(CVR)对items进行排名至关重要，CTR预估聚焦于点击率的预估（广告系统的目标），而电商领域user点击之后下单消费才是终极目标（用户的行为遵循一个序列模式：曝光->点击->转化）。请注意，这里的预测目标是假设商品被点击之后的转化率，即pCVR=p(conversion|click, impression)，而不是转化率。二者是有区别的：一个商品的点击率很低，但是一旦被点击之后，用户购买的概率可能非常高。常规的CVR模型采用深度学习方法进行建模，取得了非常好的效果。但是在实践中也遇到几个问题，这使CVR建模具有挑战性。主要问题如下：

- **Sample Selection Bias，SSB**

  > 样本选择偏差问题。传统CVR的预估模型是在曝光和click的数据上训练，在inference的时候使用了整个样本空间，如图所示。训练样本和实际数据不服从同一分布，不符合机器学习中训练数据和测试数据独立同分布的假设。直观的说，会产生转化的用户不一定都是进行了点击操作的用户，如果只使用点击后的样本来训练，会导致CVR学习产生偏置。如图所示：

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/1.jpg)

- **Data Sparsity，DS**

  > 数据稀疏问题。点击样本在整个样本空间中只占了很小一部分，而转化样本更少，高度稀疏的训练数据使得模型的学习变得相当困难。Table1显示了这个问题。

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/2.jpg)

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

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/4.jpg)

> 其中<a href="https://www.codecogs.com/eqnedit.php?latex=z" target="_blank"><img src="https://latex.codecogs.com/svg.latex?z" title="z" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=y" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y" title="y" /></a>分别表示conversion和click。
>
> 注意到，在全部样本空间中，CTR对应的label为click，而CTCVR对应的label为click & conversion，**这两个任务是都可以使用全部样本进行预估**。**所以ESMM通过这学习两个任务，再根据上式隐式地学习CVR**，具体结构如下：

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/3.jpg)

> ESMM模型分为两个子模型，pCTR和pCVR：两者共享特征的embedding层，从concatenate之后各自学习参数；pCVR仅仅是网络中的一个variable，没有监督信号，只有pCTR和pCTCVR才有监督信号进行学习。ESMM的loss函数定义如下：

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/5.jpg)

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

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/6.jpg)

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/7.jpg)

## 讨论

- ESMM 根据用户行为序列，显示引入CTR和CTCVR作为辅助任务，“隐式” 学习CVR，从而在完整样本空间下进行模型的训练和预测，解决了CVR预估中的2个难题。
- 可以把 ESMM 看成一个**新颖的 MTL 框架**，其中子任务的网络结构是可替换的，当中有很大的想象空间。

## 代码

https://github.com/qiaoguan/deep-ctr-prediction

# ESM2模型

## 背景

> 虽然ESMM模型一定程度的消除了样本选择偏差，但对于CVR预估来说，ESMM模型仍面临一定的样本稀疏问题，因为click到buy的样本非常少。但其实一个用户在购买某个商品之前往往会有一些其他的行为，比如将商品加入购物车或者心愿单。如下所示：

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/8.jpg)

> 加入心愿单／购物车的数据相较购买数据还是比较多的，因此可以基于这部分数据，通过多任务学习模型来求解CVR模型。文中把加入购物车或者心愿单此类行为称作Deterministic Action (DAction) ，而其他对购买相关性不是很大的行为称作Other Action(OAction) 。此时原来的Impression→Click→Buy过程变成了更加丰富的Impression→Click→DAction/OAction→Buy过程。

## 模型

### 模型结构

> ESM2模型结构如图所示，共有3层，SEM、DPM、SCM。
>
> - SEM是embedding共享层，主要将user、item以及user和item交互的sparse ID features和dense features进行embedding；
> - DPM是全连接层，各个子任务分别训练
> - SCM是最终的loss函数输出，预测对应的概率

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/9.jpg)

> 图中共有4个任务Y1~Y4，意义如下：
>
> - Y1： 点击率
> - Y2： 点击到 DAction 的概率
> - Y3： DAction 到购买的概率
> - Y4： OAction 到购买的概率
>
> 由于DAction和OAction是对立事件，所以从点击到OAction的概率和点击到 DAction 的概率求和为1。

### 损失函数

> 模型有3个loss函数，采用logloss，分别定义如下：
>
> - **pCTR**
>
>   Impression→Click的概率是第一个网络的输出。
>
> - **pCTAVR**
>
>   Impression→Click→DAction的概率，pCTAVR = Y1 * Y2，由前两个网络的输出结果相乘得到。
>
> - **pCTCVR**：
>   Impression→Click→DAction/OAction→Buy的概率，
>
>   pCTCVR = CTR * CVR = Y1 * [(1 - Y2) * Y4 + Y2 * Y3]，由四个网络的输出共同得到。其中CVR=(1 - Y2) * Y4 + Y2 * Y3，因为从点击到DAction和点击到OAction是对立事件。

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/10.jpg)

> 最终的损失函数由3部分加权得到：

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/11.jpg)

> 在预测时，只需要经过后三个网络，便可以计算对应的CVR。

## 实验设置

> 文章对比了几个模型在CVR预估上的效果：
>
> - GBDT
>
> - DNN
>
>   使用Click→Buy的样本来训练CVR模型，使用Impression→Click的样本来训练CTR模型
>
> - DNN-OS
>
>   对Click→Buy的样本进行过采样，其他同DNN
>
> - ESMM
>
> - ESM2

> 评估指标包括AUC和GAUC，GAUC是对每个用户的AUC进行加权的结果。结果表明：ESM2的表现最好。

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/12.jpg)

## Ablation studies

> 此外，作者还在ESM2上做了Ablation studies，如下：

- Hyper-parameters of deep neural network

  主要包含dropout ratio，hidden layers数量，embeddings的dimension

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/13.jpg)

- Effectiveness of embedding dense numerical features

  对于numerical features，通常的做法是离散化成one-hot特征，然后embedding。作者尝试了另外一种方式，先将feature归一化，然后采用Tanh来做embedding，最终得到了0.004的AUC收益。

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/14.jpg)

- Effectiveness of decomposing post-click behaviors

  文中作者将post-click behaviors分为Scart和Wish，分别对比了only Scart、only Wish和both SCart and Wish的表现，the combination of both SCart and Wish achieves the best AUC scores. 

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/15.jpg)

## Performance analysis of user behaviors

为了对比ESM2模型和ESSM模型的表现差异，作者根据user的购买次数将测试集划分为4组，[0, 10], [11, 20], [21, 50], [50, +)。对比来看，购买行为丰富的user组里，ESM2的AUC提升较大，主要原因在于，购买行为丰富的user，其post-click behaviors（Scart，Wish）也更加丰富。

![image](https://github.com/ShaoQiBNU/ESMM/blob/master/img/16.jpg)
