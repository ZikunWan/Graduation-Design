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
当前任务同时存在三类异构性：不同客户端的 MRI 模态不一致、标签空间不完全一致，并且 2D / 3D 客户端使用的模型结构不同。在这种设定下，直接聚合 backbone 参数容易把不同成像协议、不同空间维度和不同模态组合下学习到的低层表征混合在一起，从而引入负迁移。因此，本文不追求构建一个所有客户端共享的统一图像编码器，而是保留客户端私有的模态 encoder，并将联邦协同限制在更高层的语义空间中。

已有 baseline 的结果也支持这一判断。一方面，`FedMM` 和 `FedAMM` 表明模态专属 encoder 与 prototype 级通信能够显式处理缺失模态；另一方面，`LG-FedAvg` 的强表现说明，在强异构场景中，分类头所表达的类别判别边界具有较高的跨客户端共享价值。基于这一观察，本文拟提出一种模态感知的联邦分类方法：客户端仍然训练本地模态 encoder，服务器不聚合 backbone，而是利用模态组合 prototype 指导分类头参数交换，使跨客户端知识共享同时作用于“类别语义中心”和“分类判别边界”。

该方法的核心思想可以概括为三点：
1. 用 mask-aware gated fusion 替代简单 concat，使模型根据当前样本实际拥有的模态自适应融合特征。
2. 用模态组合 prototype 描述不同缺失模态条件下的类别语义，并通过更完整模态组合向缺失模态组合提供语义监督。
3. 用 prototype 的可靠性和模态完整度指导分类头聚合，使分类头交换不再是简单平均，而是类别级、模态感知的参数共享。

### 模态感知特征融合
对客户端 $k$ 的样本 $i$，设其可用模态集合为 $\mathcal{M}_i$。每个模态 $m$ 由对应的本地 encoder 提取特征：

$$
h_{i,m}^{k}=E_{k,m}(x_{i,m})
$$

为避免不同客户端因可用模态数量不同而产生不同维度的 fused feature，先将每个模态特征投影到统一语义空间：

$$
u_{i,m}^{k}=P_{k,m}h_{i,m}^{k}\in \mathbb{R}^{D}
$$

随后根据模态特征、模态类型和当前样本的 modality mask 计算门控分数：

$$
s_{i,m}^{k}=a^\top \tanh(W_u u_{i,m}^{k}+W_e e_m+W_q q_i)
$$

其中 $e_m$ 表示模态 embedding，$q_i$ 表示由 modality mask 得到的缺失模态状态编码。只在可用模态集合内归一化：

$$
\alpha_{i,m}^{k}
=
\frac{\exp(s_{i,m}^{k})}
{\sum_{r\in\mathcal{M}_i}\exp(s_{i,r}^{k})}
$$

最终 fused feature 为：

$$
z_i^k=\sum_{m\in\mathcal{M}_i}\alpha_{i,m}^{k}u_{i,m}^{k}
$$

相比直接拼接不同模态特征，该融合方式具有两个优势：第一，所有客户端的分类头输入维度统一，便于后续进行分类头参数交换；第二，模型能够根据不同样本的模态缺失状态动态调整各模态贡献，而不是默认所有模态同等重要。

### 模态组合 Prototype Bank
服务器维护按 `(modality_combination, class)` 组织的全局 prototype bank。设 $s$ 表示一个模态组合，例如 `t1c`、`t1c+t2f` 或 `t1+t1c+t2w+t2f`。客户端 $k$ 在本地统计：

$$
p_{k,s,c}
=
\frac{1}{n_{k,s,c}}
\sum_{i:y_i=c,\ \mathcal{M}_i=s} z_i^k
$$

其中 $n_{k,s,c}$ 表示客户端 $k$ 中属于类别 $c$ 且模态组合为 $s$ 的样本数。服务器端不是仅做普通样本数加权平均，而是引入 prototype 可靠性：

$$
\omega_{k,s,c}
=
n_{k,s,c}\cdot
\exp(\cos(p_{k,s,c},G_{s,c}^{t})/\tau)
$$

并得到当前轮候选全局 prototype：

$$
\bar{G}_{s,c}^{t+1}
=
\frac{\sum_k \omega_{k,s,c}p_{k,s,c}}
{\sum_k \omega_{k,s,c}}
$$

最终采用动量更新：

$$
G_{s,c}^{t+1}
=
(1-\mu)G_{s,c}^{t}
+
\mu\bar{G}_{s,c}^{t+1}
$$

该设计的目的不是增加复杂通信，而是降低小样本客户端或噪声 prototype 对全局 prototype bank 的瞬时扰动，使服务器端语义中心随训练过程平滑演化。

### 缺失模态的 Prototype 下发
客户端接收 prototype 时，不只接收与自身模态组合完全一致的 $G_{s,c}$。对于缺失模态组合 $s$，服务器还可以利用包含 $s$ 的更完整模态组合 $s'$ 构造 teacher prototype：

$$
T_{s,c}
=
\lambda G_{s,c}
+
(1-\lambda)
\sum_{s'\supset s}\psi(s,s')G_{s',c}
$$

其中：

$$
\psi(s,s')
=
\frac{\exp(|s'|/\tau)}
{\sum_{r\supset s}\exp(|r|/\tau)}
$$

这样，单模态客户端不需要生成缺失图像，也不需要补齐缺失输入，而是通过更完整模态组合的 prototype 获得语义监督。该机制保留了缺失模态方法的效率优势，同时使单模态表示能够向多模态语义空间靠近。

### 原型指导的分类头聚合
分类头仍保持一个普通线性层，不拆分为额外模块：

$$
\hat{y}_i=W_k z_i^k+b_k
$$

其中 $W_k\in\mathbb{R}^{C\times D}$，$b_k\in\mathbb{R}^{C}$。服务器按类别聚合分类头的第 $c$ 行：

$$
W_c^{t+1}=\sum_{k\in\mathcal{K}_c}\pi_{k,c}W_{k,c}^{t+1}
$$

$$
b_c^{t+1}=\sum_{k\in\mathcal{K}_c}\pi_{k,c}b_{k,c}^{t+1}
$$

聚合权重由类别样本量、prototype 一致性和模态完整度共同决定：

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

其中 prototype 一致性为：

$$
\rho_{k,c}
=
\frac{
\sum_s n_{k,s,c}\cos(p_{k,s,c},G_{s,c})
}{
\sum_s n_{k,s,c}
}
$$

模态完整度为：

$$
\eta_{k,c}
=
\frac{
\sum_s n_{k,s,c}|s|/M
}{
\sum_s n_{k,s,c}
}
$$

这里 $M$ 是全局模态总数，$|s|$ 是模态组合 $s$ 中包含的模态数量。与 `LG-FedAvg` 的分类头简单平均相比，该策略仍然只交换轻量的线性分类头参数，但能够区分不同客户端在每个类别上的可靠性：样本更多、prototype 更接近全局语义中心、模态信息更完整的客户端，对该类别分类边界的贡献更大。

### 训练目标
为保持方法简洁，客户端本地训练只使用三类损失：

$$
\mathcal{L}
=
\mathcal{L}_{ce}
+
\lambda_p\mathcal{L}_{proto}
+
\lambda_h\mathcal{L}_{head}
$$

其中分类损失为：

$$
\mathcal{L}_{ce}
=
\text{CE}(W_k z_i^k+b_k,y_i)
$$

prototype 对齐损失为：

$$
\mathcal{L}_{proto}
=
\|p_{k,s,c}-T_{s,c}\|_2^2
$$

分类头语义校准损失为：

$$
\mathcal{L}_{head}
=
\text{CE}(W_k T_{s,c}+b_k,c)
$$

其中 $\mathcal{L}_{proto}$ 负责使本地融合特征靠近服务器端的模态组合语义中心，$\mathcal{L}_{head}$ 则进一步要求本地分类头能够正确识别这些全局语义 prototype。换言之，prototype 不仅作为表示层的对齐目标，也作为分类边界的校准样本。

### 联邦训练流程
每一轮通信包含以下步骤：
1. 服务器向客户端下发当前全局分类头和 prototype bank。
2. 客户端使用本地可用模态训练私有 encoder、gated fusion 模块和线性分类头。
3. 客户端统计本地 `(modality_combination, class)` prototype，并上传 prototype、样本计数和分类头参数。
4. 服务器先更新 prototype bank，再基于 prototype 可靠性与模态完整度聚合分类头。
5. 更新后的 prototype bank 和分类头进入下一轮训练。

因此，本文方法可以视为对 `FedAMM` 和 `LG-FedAvg` 的结合与扩展：`FedAMM` 提供模态组合级语义建模，`LG-FedAvg` 证明分类头共享在强异构场景中有效，而本文进一步用模态感知 prototype 指导分类头聚合，使共享的分类边界能够显式适应模态缺失结构。



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
