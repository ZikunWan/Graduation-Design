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

## 总体思路
1. 问题设定同时包含：
   - 模态缺失
   - 标签空间不一致
   - 2D / 3D 模型架构异构
2. 不共享模型参数，只共享 Prototype。
3. 服务器端维护 `(modality, class)` 级别的全局 Prototype 库。
4. 客户端保留自己的本地模型，通过分类损失和 Prototype 对齐损失联合训练。

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

## 数据加载
1. 使用 [dataset.py](./dataset.py) 自定义数据集与 `collate_fn`，不使用 `HtFLlib` 默认的 `npz -> x/y` 整包读取逻辑。
2. dataloader 按病例目录读取样本，每个样本返回其所有可用模态。
3. 全局模态顺序统一定义为：
   - `t1`
   - `t1c`
   - `t2w`
   - `t2f`
4. `collate_fn` 会输出：
   - `modalities`
   - `modality_mask`
   - `full_modality_order`
   - `sample_ids`
5. `modality_mask` 的 shape 为 `[B, 4]`，用于指示每个样本实际拥有的模态。

## 模型结构
1. 使用 [model.py](./model.py) 定义本地模型。
2. 2D 客户端与 3D 客户端分别共享各自的一套 `MONAI ResNet` 主体结构，但都遵循同一个多模态框架。
3. 模型采用“模态专属前端 + 共享后端”的设计：
   - 每个模态有自己的前端分支，前端到 `layer2`
   - `layer3 / layer4` 在同一个客户端内部共享
4. 缺失模态在 sample 级别直接跳过，不会送入对应分支。
5. 分类分支使用融合后的 backbone feature：
   - 先对所有存在模态的 feature 做 mask-aware 平均
   - 再经过分类头输出 `logits`
6. Prototype 分支不使用融合后的单一 prototype，而是为每个模态单独生成 prototype feature。

## Local Prototype 生成
1. 对一个 batch，模型输出：
   - `logits`
   - `modality_prototypes`
   - `modality_mask`
2. 其中 `modality_prototypes` 的 shape 为 `[B, 4, D_p]`，表示每个样本在每个模态上的 prototype feature。
3. 客户端端按 `(modality, class)` 分组。
4. 对每个 `(modality, class)` 组内部的样本 prototype feature 做 `K-Means`，得到多个 local prototypes。
5. 每个 local prototype 同时记录其对应的 cluster size，后续上传给服务器。
6. 使用 [utils.py](./utils.py) 完成：
   - 本地 `K-Means` prototype 计算
   - local prototype tensor 打包
   - server 端全局 prototype 聚合

## Server Prototype Bank
1. 服务器端的 Prototype 库按 `(modality, class)` 组织。
2. 每个 `(modality, class)` 组合最终只维护一个全局 Prototype。
3. 一个客户端在某个 `(modality, class)` 下可以上传多个 local prototypes。
4. 服务器端对这些 local prototypes 做加权均值聚合，得到唯一的全局 Prototype。
5. 权重由各 local prototype 对应的 cluster size 决定。

## 损失函数
1. 总损失由分类损失和 Prototype 对齐损失组成：

$$\mathcal{L} = \mathcal{L}_{cls} + \lambda_{proto}\mathcal{L}_{proto}$$


2. 分类损失为标准交叉熵：

$$\mathcal{L}_{cls} = \text{CE}(\text{logits}, y)$$


3. Prototype 对齐损失定义为：
   - 客户端 local prototype 对齐服务器对应 `(modality, class)` 的全局 Prototype
   - 若一个 `(modality, class)` 组内有多个 local prototypes，则它们都去对齐同一个全局 Prototype
4. 记：
   - $p_{m,c,k}$ 表示客户端在模态 $m$、类别 $c$ 下第 $k$ 个 local prototype
   - $g_{m,c}$ 表示服务器端对应的全局 Prototype
   - $\delta_{m,c,k} \in \{0,1\}$ 表示该 local/global prototype 对是否同时有效
   - $w_{m,c,k}$ 表示该 local prototype 的权重；若使用 cluster size 加权，则 $w_{m,c,k}=n_{m,c,k}$，否则 $w_{m,c,k}=1$
5. 则 Prototype 对齐损失可写为：

$$
\mathcal{L}_{proto}
=
\frac{
\sum\limits_{m}\sum\limits_{c}\sum\limits_{k}
\delta_{m,c,k}\, w_{m,c,k}
\left(1-\cos\left(p_{m,c,k}, g_{m,c}\right)\right)
}{
\sum\limits_{m}\sum\limits_{c}\sum\limits_{k}
\delta_{m,c,k}\, w_{m,c,k}
}
$$

6. 若不使用 cluster size 加权，则上式退化为对所有有效 prototype 对的 cosine distance 直接取平均。
7. 当前实现中对齐损失采用 cosine distance，由 [loss.py](./loss.py) 给出。

## 训练与通信流程
1. 客户端本地训练时，前向得到分类输出与模态级 prototype feature。
2. 使用分类损失更新本地模型参数。
3. 同时利用服务器下发的全局 Prototype 库计算 Prototype 对齐损失。
4. 每轮本地训练结束后，客户端重新统计自己的 `(modality, class)` local prototypes。
5. 客户端只上传 Prototype，不上传模型参数。
6. 服务器聚合后更新全局 Prototype 库，再下发给客户端进入下一轮训练。

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

## Yale 的处理方式
1. `Yale` 只有一个标签：`brain_metastases`。
2. 因此 `Yale` 本地测试集上的普通分类准确率没有区分度，不适合作为主结果。
3. `Yale` 仍然参与联邦训练，因为它提供了有价值的单类知识和域信息。
4. 但在主结果表中，`Yale` 的本地 `Accuracy` 不作为核心指标，可记为 `N/A` 或不纳入主表平均。

## Yale 贡献的评估方式
1. 通过消融实验评估 `Yale` 的价值：
   - `with Yale`
   - `without Yale`
2. 重点观察加入 `Yale` 后，其他客户端尤其是包含 `brain_metastases` 类的客户端是否受益。
3. 最值得关注的是：
   - `BraTS` 中 `brain_metastases` 相关结果
   - `Shanghai` 中 `brain_metastases` 相关结果
4. 这样可以把 `Yale` 定义为“单类知识源客户端”，而不是一个需要单独做可区分本地测试的普通客户端。
