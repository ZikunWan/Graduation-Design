# 数据集

| 数据来源 | 模态 | 类别 | 维度 | 样本数 |
| :--: | :--: | :--: | :--: | :--: |
| BraTS | `t1`、`t1c`、`t2w`、`t2f` | 转移瘤、胶质瘤、脑膜瘤 | 3D | `1296 + 1311 + 1000 = 3607` |
| Shanghai | `t1c`、`t2f` | 转移瘤、胶质瘤、正常 | 3D | `1054 + 1149 + 1218 = 3421` |
| Yale | `t1c`、`t2f` | 转移瘤 | 3D | `111` 个目录，其中 `100` 个完整双模态样本，`11` 个单模态样本 |
| Figshare | `t1c` | 胶质瘤、脑膜瘤、垂体瘤 | 2D | `1426 + 708 + 930 = 3064` |
| BRISC 2025 | `t1` | 胶质瘤、脑膜瘤、垂体瘤、正常 | 2D | `1401 + 1635 + 1757 + 1207 = 6000` |

# 预处理

## 总体原则
1. 预处理阶段不再进行多模态通道堆叠。
2. 每个模态单独保存为一个 `.npz` 文件，后续在 dataloader 中按病例目录聚合为一个病人的多模态输入。
3. 多模态客户端统一采用病例级目录结构保存，单模态客户端采用样本级目录结构保存。
4. `.npz` 中只保存单模态张量 `x` 和标签 `y`。
5. 3D 数据保存的数组顺序统一为 `[D, H, W]`。

## 输出结构

### 3D 客户端
`preprocessed/<dataset>/<split>/<label>/<case>/<modality>.npz`

例如：
- `preprocessed/BraTS/train/glioma/BraTS-GLI-00000-000/t1.npz`
- `preprocessed/BraTS/train/glioma/BraTS-GLI-00000-000/t1c.npz`
- `preprocessed/Shanghai/test/no_tumor/Clinical-NOR-0001/t1c.npz`
- `preprocessed/Yale/train/brain_metastases/YG_2GBL4CHD8WHX/t2f.npz`

### 2D 客户端
`preprocessed/<dataset>/<split>/<label>/<sample>/<modality>.npz`

例如：
- `preprocessed/Figshare/train/glioma/figshare_00001_xxx/t1c.npz`
- `preprocessed/Brisc2025/test/no_tumor/brisc2025_test_00001_xxx/t1.npz`

## BraTS
1. 原始读取四个模态：`t1n`、`t1c`、`t2w`、`t2f`。
2. 预处理后将 `t1n` 统一命名为 `t1` 保存。
3. 对所有模态统一方向到 `RAS`。
4. 以 `t1c` 为参考网格，将所有可用模态重采样到 `spacing = (1, 1, 1)`。
5. 对每个模态分别做非零区域的 `z-score` 强度归一化。
6. 对空间尺寸做中心 `crop/pad`：
   - `H, W -> 224 x 224`
   - `D -> 155`
7. 每个模态单独保存：
   - `t1.npz`
   - `t1c.npz`
   - `t2w.npz`
   - `t2f.npz`
8. 每个单模态文件最终满足 `x.shape = [155, 224, 224]`。

## Shanghai
1. 读取两个模态：`t1c`、`t2f`。
2. 当前版本不做方向统一，也不做 spacing 重采样，尽量保留原始几何关系。
3. 对每个模态分别做非零区域的 `z-score` 强度归一化。
4. 仅对 `H, W` 做中心 `crop/pad -> 224 x 224`。
5. 保持原始深度不变。按当前数据统计，预处理后 `D = 16`。
6. 每个模态单独保存：
   - `t1c.npz`
   - `t2f.npz`
7. 每个单模态文件最终满足 `x.shape = [16, 224, 224]`。

## Yale
1. 已先将 8 个包含两组 `t1ce/flair` 的目录按文件名前缀拆成独立样本目录。
2. 原始文件名中的 `t1ce`、`flair`，预处理后统一映射保存为 `t1c`、`t2f`。
3. 不再丢弃单模态目录，因为缺失模态本身就是研究对象。
4. 当前 `input/Yale` 中共有：
   - `100` 个完整双模态目录
   - `11` 个单模态目录
5. 对所有可用模态统一方向到 `RAS`。
6. 以 `t1c` 为优先参考模态；如果目录中只有一个模态，则以现有模态本身作为参考。
7. 将所有可用模态重采样到 `spacing = (1, 1, 1)`。
8. 对每个模态分别做非零区域的 `z-score` 强度归一化。
9. 对空间尺寸做中心 `crop/pad`：
   - `H, W -> 224 x 224`
   - `D -> 155`
10. 每个模态单独保存：
   - `t1c.npz`
   - `t2f.npz`
11. 每个单模态文件最终满足 `x.shape = [155, 224, 224]`。

## Figshare
1. 读取单张 `t1c` 图像。
2. 仅保留 `512 x 512` 的样本，不使用 `256 x 256` 的样本。
3. 将像素值转换为 `float32`，并归一化到 `[0, 1]`。
4. 不做堆叠，直接保存为单模态文件 `t1c.npz`。
5. 最终满足 `x.shape = [512, 512]`。

## Brisc2025
1. 读取单张 `t1` 图像。
2. 将像素值转换为 `float32`，并归一化到 `[0, 1]`。
3. 对图像做中心 `crop/pad -> 512 x 512`，不使用 resize。
4. 不做堆叠，直接保存为单模态文件 `t1.npz`。
5. 最终满足 `x.shape = [512, 512]`。

# 方法

## Baseline 方法速览（client/server 对应）
| 方法 | Client 实现 | Server 实现 | 核心 idea（只保留最核心） |
| :-- | :-- | :-- | :-- |
| `Local` | [client/clientlocal.py](./client/clientlocal.py) | [server/serverlocal.py](./server/serverlocal.py) | 不做联邦聚合，每个客户端独立训练与评估，作为本地下界基线。 |
| `LG-FedAvg` | [client/clientlg.py](./client/clientlg.py) | [server/serverlg.py](./server/serverlg.py) | 仅同步/聚合分类头，backbone 完全本地化，用最小共享缓解异构冲突。 |
| `FedProto` | [client/clientproto.py](./client/clientproto.py) | [server/serverproto.py](./server/serverproto.py) | 只通信“类原型”（特征中心），服务器按类平均，客户端用原型对齐约束本地特征。 |
| `FedGH` | [client/clientgh.py](./client/clientgh.py) | [server/servergh.py](./server/servergh.py) | 共享全局分类头；客户端上传类原型，服务器在原型-标签对上训练全局 head 再回传。 |
| `FedTGP` | [client/clienttgp.py](./client/clienttgp.py) | [server/servertgp.py](./server/servertgp.py) | 用可学习的原型生成器替代静态均值原型，并通过 margin 约束增强类间可分性。 |
| `FD` | [client/clientfd.py](./client/clientfd.py) | [server/serverfd.py](./server/serverfd.py) | 只通信类别级 logit，服务器做类 logit 聚合形成 teacher，客户端做蒸馏+监督联合训练。 |
| `FedAMM` | [client/clientamm.py](./client/clientamm.py) | [server/serveramm.py](./server/serveramm.py) | 以“模态组合-类别”原型为通信单元，同时做模态平衡与组合对齐，重点处理缺失模态场景。 |
| `FedMM` | [client/clientmm.py](./client/clientmm.py) | [server/servermm.py](./server/servermm.py) | 按模态拆分特征提取器，客户端只训练自己拥有的模态，并用全局原型正则化。 |

## 缺失模态 Baseline 规划
1. 当前任务关注“不同客户端模态不同”的脑肿瘤图像分类，并同时允许 2D / 3D 模型架构异构。
2. 因为 2D / 3D backbone 参数结构不同，缺失模态 baseline 不跨架构做 backbone 参数聚合。
3. 各方法统一使用本地私有模型提取特征，再通过 embedding / prototype 级通信比较缺失模态处理能力。
4. `FedMEMA` 暂不作为 baseline，因为当前设定下没有服务器端完整模态数据，无法公平构造原方法依赖的 multimodal anchors。
5. `PEPSY` 暂不作为 baseline，因为当前阶段不使用该方法。

### FedAMM
1. 保留 `Intra-client modality balance`：
   - 在客户端内部平衡不同模态对分类结果的贡献。
   - 对拥有多模态样本的客户端，可使用多模态融合特征作为 teacher，单模态特征作为 student。
2. 保留 `Prototype distillation`：
   - 将原分割任务中的像素级语义 prototype 改为分类任务中的类别级 prototype。
   - 单模态类别 prototype 对齐多模态类别 prototype，使单模态分支学习更接近多模态语义的信息。
3. 保留 `Inter-client modality-combination prototype`：
   - 服务器端按 `(modality_combination, class)` 维护全局 prototype。
   - 例如 `(t1c+t2f, glioma)`、`(t1, meningioma)`、`(t1+t1c+t2w+t2f, brain_metastases)`。
   - 客户端本地 prototype 与对应模态组合和类别的全局 prototype 对齐。
4. 原方法中的模态加权 encoder 聚合不跨 2D / 3D 使用；如需保留，只能在同架构子组内进行。

### FedMM
1. 保留 `Per-modality extractor decomposition`：
   - 每个模态拥有独立的特征提取分支或特征空间。
   - 客户端只实例化和训练自己拥有模态对应的部分。
2. 保留 `客户端按可用模态参与训练`：
   - 有 `t1` 的客户端参与 `t1` 相关表示学习。
   - 有 `t1c`、`t2w`、`t2f` 的客户端分别参与对应模态的表示学习。
   - 单模态客户端不需要构造不存在模态的输入。
3. 保留 `Local fusion/classifier`：
   - 多模态客户端在本地融合可用模态特征后分类。
   - 单模态客户端直接使用单模态特征分类。
   - 分类器可以本地保留，以适应标签空间和模型结构差异。
4. 保留 `Global prototype regularization`：
   - 服务器端维护 `(modality, class)` 级别的全局 prototype。
   - 客户端直接使用第 \(k\) 个模态 extractor 输出的 feature embedding 作为 \(h_i^{(k)}\)，不额外使用 prototype head。
   - 客户端本地同模态同类别 embedding 对齐全局 prototype。
   - 每个客户端先计算本地 `(class, modality)` prototype，服务器端再对各客户端上传的同一 `(class, modality)` prototype 做 client-level 平均。
5. 保留原文动态损失：
   - 用 $\lambda(t)=1/(1+\exp(-\alpha(t-t_0)))$ 在训练早期强调分类损失，后期逐步强调全局 prototype 的 L2 对齐损失。
   - prototype 正则项采用 $\beta\lambda(t)L_{proto}/D$，其中 $\beta$ 在分子上，$D$ 是 prototype 维度。
   - 当前分类项将原文的 `BCE` 替换为多分类任务的 `CE`。
6. 同一模态的 extractor 参数不跨 2D / 3D 聚合；跨架构只共享 prototype 级知识。

## 拟提出方法：模态感知融合与原型指导的分类头聚合

### 方法动机
本任务里的客户端差异很大。不同医院的数据模态不同，有的客户端是 3D MRI，有的客户端是 2D 图像，而且每个客户端拥有的类别也不完全一样。在这种情况下，如果强行把所有客户端的 backbone 参数平均，很容易把不一样的数据分布混在一起，导致负迁移。

因此，本文方法的基本选择是：**不聚合 backbone，只在更高层共享语义信息**。每个客户端保留自己的模态 encoder，用它适应本地数据；服务器只负责共享两类轻量信息：
1. prototype：表示某个类别在某种模态组合下的“特征中心”。
2. 分类头：表示每个类别的“分类边界”。

这个思路来自两个观察。第一，`FedMM` 和 `FedAMM` 说明模态专属 encoder 可以处理缺失模态问题。第二，`LG-FedAvg` 的结果很强，说明分类头共享在强异构场景中很有价值。因此，本文希望把二者结合起来：**用模态组合 prototype 描述缺失模态下的类别语义，再用这些 prototype 指导分类头聚合**。

本文方法主要改进三处：
1. 特征融合：不再简单拼接不同模态特征，而是让模型自动判断当前样本更应该依赖哪些模态。
2. Prototype 传递：单模态客户端不仅学习自己的单模态 prototype，也可以从更完整的多模态 prototype 中获得监督。
3. 分类头聚合：不再简单平均分类头，而是让样本更多、prototype 更可靠、模态更完整的客户端拥有更高权重。

### 主要符号
为避免后文公式过长，先约定几个符号：

| 符号 | 含义 |
| :-- | :-- |
| $k$ | 第 $k$ 个客户端 |
| $i$ | 第 $i$ 个样本 |
| $m$ | 某一个模态，例如 `t1c` |
| $s$ | 某一种模态组合，例如 `t1c+t2f` |
| $c$ | 某一个类别 |
| $z_i^k$ | 客户端 $k$ 上样本 $i$ 的融合特征 |
| $p_{k,s,c}$ | 客户端 $k$ 上模态组合 $s$、类别 $c$ 的本地 prototype |
| $G_{s,c}$ | 服务器维护的全局 prototype |
| $T_{s,c}$ | 服务器下发给客户端的 teacher prototype |
| $W_k,b_k$ | 客户端 $k$ 的线性分类头 |

### 模态感知特征融合
当前模型如果直接把多个模态的特征 concat 到一起，会有两个问题。第一，不同客户端拥有的模态数量不同，concat 后的特征维度不容易统一。第二，concat 默认所有模态都直接交给分类头处理，没有显式判断哪个模态对当前样本更重要。

因此，本文先把每个模态的特征投影到同一个维度。设样本 $i$ 拥有的模态集合为 $\mathcal{M}_i$，模态 $m$ 的 encoder 输出为：

$$
h_{i,m}^{k}=E_{k,m}(x_{i,m})
$$

再通过一个投影层得到统一维度的特征：

$$
u_{i,m}^{k}=P_{k,m}h_{i,m}^{k}\in \mathbb{R}^{D}
$$

这里 $D$ 是统一后的特征维度。这样无论客户端有一个模态还是多个模态，最后都可以得到相同维度的融合特征。

然后给每个可用模态分配一个权重：

$$
\alpha_{i,m}^{k}
=
\text{softmax}_{m\in\mathcal{M}_i}(g(u_{i,m}^{k}, m, \mathcal{M}_i))
$$

其中 $g(\cdot)$ 是一个很小的门控网络，用来根据模态特征、模态类型和当前样本缺失了哪些模态，判断该模态的重要性。缺失的模态不参与 softmax，因此不需要补零，也不需要生成缺失图像。

最终 fused feature 为：

$$
z_i^k=\sum_{m\in\mathcal{M}_i}\alpha_{i,m}^{k}u_{i,m}^{k}
$$

直观来说，这一步就是让模型学会“当前样本更相信哪个模态”。例如同样是 `t1c+t2f`，不同病例可能一个更依赖增强信息，另一个更依赖 FLAIR 信息，门控融合可以给它们不同的权重。

### 模态组合 Prototype Bank
Prototype 可以理解为一组样本的“平均特征”。本文不只按类别统计 prototype，而是按 `(模态组合, 类别)` 统计。例如：
- `(t1c, glioma)`
- `(t1c+t2f, glioma)`
- `(t1+t1c+t2w+t2f, meningioma)`

这样做的原因是：同一个类别在不同模态组合下看到的信息不同，直接把它们平均成一个类别 prototype 会丢失模态缺失信息。

客户端 $k$ 在本地统计：

$$
p_{k,s,c}
=
\frac{1}{n_{k,s,c}}
\sum_{i:y_i=c,\ \mathcal{M}_i=s} z_i^k
$$

这表示：把客户端 $k$ 中所有“模态组合为 $s$ 且类别为 $c$”的样本特征取平均，得到一个本地 prototype。

服务器聚合时，不简单平均所有客户端的 prototype，而是给更可靠的 prototype 更高权重：

$$
\omega_{k,s,c}
=
n_{k,s,c}\cdot
\exp(\cos(p_{k,s,c},G_{s,c}^{t})/\tau)
$$

这里的逻辑很简单：样本越多，prototype 越可信；同时，如果本地 prototype 和上一轮全局 prototype 方向更接近，说明它不像噪声点，也应该更可信。

服务器先得到新的候选 prototype：

$$
\bar{G}_{s,c}^{t+1}
=
\frac{\sum_k \omega_{k,s,c}p_{k,s,c}}
{\sum_k \omega_{k,s,c}}
$$

再用动量更新：

$$
G_{s,c}^{t+1}
=
(1-\mu)G_{s,c}^{t}
+
\mu\bar{G}_{s,c}^{t+1}
$$

动量更新的作用是让全局 prototype 不会因为某一轮客户端数据波动而突然变化太大。

### 缺失模态的 Prototype 下发
单模态客户端最大的问题是：它永远看不到完整多模态信息。本文不去生成缺失模态，而是让单模态客户端从更完整的多模态 prototype 中学习。

例如，一个只有 `t1c` 的客户端，除了接收 `(t1c, glioma)` 的 prototype，也可以参考 `(t1c+t2f, glioma)` 或 `(t1+t1c+t2w+t2f, glioma)` 的 prototype。服务器为模态组合 $s$ 构造 teacher prototype：

$$
T_{s,c}
=
\lambda G_{s,c}
+
(1-\lambda)
\sum_{s'\supset s}\psi(s,s')G_{s',c}
$$

这里 $T_{s,c}$ 是最终下发给客户端的目标 prototype。它由两部分组成：一部分来自客户端自己模态组合对应的 prototype，另一部分来自更完整的模态组合 prototype。$\lambda$ 控制这两部分的比例。

更完整模态组合的权重为：

$$
\psi(s,s')
=
\frac{\exp(|s'|/\tau)}
{\sum_{r\supset s}\exp(|r|/\tau)}
$$

其中 $|s'|$ 表示模态组合 $s'$ 中包含多少个模态。模态越完整，权重越高。这样单模态客户端虽然没有多模态输入，但仍然能通过 prototype 获得来自多模态客户端的语义指导。

### 原型指导的分类头聚合
分类头仍然是一个普通线性层：

$$
\hat{y}_i=W_k z_i^k+b_k
$$

由于前面的 fusion 已经把所有客户端的融合特征统一到维度 $D$，所以不同客户端的分类头形状一致，可以进行参数交换。

`LG-FedAvg` 是直接平均分类头。本文希望更细一些：不同客户端对不同类别的可靠性不一样，所以分类头按类别逐行聚合：

$$
W_c^{t+1}=\sum_{k\in\mathcal{K}_c}\pi_{k,c}W_{k,c}^{t+1}
$$

$$
b_c^{t+1}=\sum_{k\in\mathcal{K}_c}\pi_{k,c}b_{k,c}^{t+1}
$$

其中 $W_{k,c}$ 和 $b_{k,c}$ 是客户端 $k$ 分类头中对应类别 $c$ 的那一行参数，$\pi_{k,c}$ 是它在这个类别上的聚合权重。

权重主要看三个因素：
1. 这个客户端上该类别的样本多不多。
2. 这个客户端的 prototype 和全局 prototype 是否接近。
3. 这个客户端在该类别上是否拥有更完整的模态。

对应的权重可以写成：

$$
r_{k,c}
=
(n_{k,c}+\epsilon)^\gamma
\cdot
\exp(\rho_{k,c}/\tau_h)
\cdot
(1+\beta\eta_{k,c})
$$

$$
\pi_{k,c}
=
\frac{r_{k,c}}
{\sum_{j\in\mathcal{K}_c}r_{j,c}}
$$

这里 $n_{k,c}$ 是类别样本数，$\rho_{k,c}$ 表示 prototype 一致性，$\eta_{k,c}$ 表示模态完整度。$\epsilon,\gamma,\tau_h,\beta$ 都是超参数，用来控制各部分权重的强弱。

其中：

$$
\rho_{k,c}
=
\frac{
\sum_s n_{k,s,c}\cos(p_{k,s,c},G_{s,c})
}{
\sum_s n_{k,s,c}
}
$$

它表示客户端 $k$ 在类别 $c$ 上的 prototype 平均有多接近全局 prototype。

$$
\eta_{k,c}
=
\frac{
\sum_s n_{k,s,c}|s|/M
}{
\sum_s n_{k,s,c}
}
$$

它表示客户端 $k$ 在类别 $c$ 上平均拥有多少比例的模态。最终效果是：如果某个客户端在某类上样本更多、特征更稳定、模态更完整，那么它对该类分类头的贡献就更大。

### 训练目标
客户端本地训练只保留三类损失，避免方法过于复杂：

$$
\mathcal{L}
=
\mathcal{L}_{ce}
+
\lambda_p\mathcal{L}_{proto}
+
\lambda_h\mathcal{L}_{head}
$$

第一项是普通分类损失：

$$
\mathcal{L}_{ce}
=
\text{CE}(W_k z_i^k+b_k,y_i)
$$

它保证模型在本地数据上能正常分类。

第二项是 prototype 对齐损失：

$$
\mathcal{L}_{proto}
=
\|p_{k,s,c}-T_{s,c}\|_2^2
$$

它让客户端自己的 prototype 靠近服务器下发的 teacher prototype，也就是让本地特征向全局语义靠近。

第三项是分类头校准损失：

$$
\mathcal{L}_{head}
=
\text{CE}(W_k T_{s,c}+b_k,c)
$$

它把 teacher prototype 当作一类“标准样本”，要求分类头能把它判成正确类别。这样 prototype 不仅影响特征空间，也会直接影响分类边界。

### 联邦训练流程
每一轮通信包含以下步骤：
1. 服务器向客户端下发当前全局分类头和 prototype bank。
2. 客户端使用本地可用模态训练私有 encoder、gated fusion 模块和线性分类头。
3. 客户端统计本地 `(modality_combination, class)` prototype，并上传 prototype、样本计数和分类头参数。
4. 服务器先更新 prototype bank，再基于 prototype 可靠性与模态完整度聚合分类头。
5. 更新后的 prototype bank 和分类头进入下一轮训练。

总结来说，本文方法可以理解为：在 `FedAMM` 的模态组合 prototype 基础上，加入 `LG-FedAvg` 中被证明有效的分类头共享；但分类头不是简单平均，而是由 prototype 和模态完整度来决定每个客户端在每个类别上的贡献。



# 实验设计

## 联邦设定
1. 使用 5 个客户端，每个数据集对应一个客户端：
   - `BraTS`
   - `Shanghai`
   - `Yale`
   - `Figshare`
   - `Brisc2025`
2. 客户端之间同时存在：
   - 模态不一致
   - 标签空间不一致
   - 模型架构不一致

## 数据加载思路
1. dataloader 以“病例目录”为单位读取样本，而不是直接读取一个堆叠好的多通道张量。
2. 每个样本返回该病例下所有可用模态的单独张量，例如：

```python
{
    "modalities": {
        "t1c": tensor,
        "t2f": tensor,
    },
    "label": y,
    "available_modalities": ["t1c", "t2f"],
}
```

3. 单模态客户端只会返回一个模态；多模态客户端返回多个模态。

## 评估协议
1. 在异构模型架构设定下，不使用统一全局测试集。
2. 所有方法只在各客户端自己的本地测试集上评估。
3. 评估重点放在联邦学习方法本身，不把 Local-only 结果作为当前阶段主表。

## 主结果
1. 主结果表只报告以下 4 个客户端的本地测试表现：
   - `BraTS`
   - `Shanghai`
   - `Figshare`
   - `Brisc2025`
2. 这 4 个客户端都具有至少两个本地类别，因此本地测试指标有区分度。
3. 可以报告每个客户端的 `Accuracy`，并额外给出 `Macro-F1` 作为辅助指标。
4. 最终主表可对这 4 个客户端取简单平均，作为整体结果。
