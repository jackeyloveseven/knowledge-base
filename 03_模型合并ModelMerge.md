# 为什么25/26年图像生成&视频生成模型要做 Model Merge（模型合并）

> 日期：2026-06-23 | 触发人：李承燕

## 一句话总结

模型合并（Model Merging / Weight Interpolation）成为 2025-2026 年图像&视频生成模型后训练的标准操作，**核心原因是 SFT/D PO 阶段面临多能力维度的"不可能三角"——单一模型在高质量数据子集上微调后产生能力偏置，合并是低成本消除偏置、逼近帕累托最优的关键技术**。

---

## 核心概念定义

| 概念 | 定义 | 来源 |
|------|------|------|
| **Model Soup（模型汤）** | 对同一预训练基座、不同超参数微调的多个模型进行权重平均（Weight Averaging），推理时无额外开销 | Wortsman et al., ICML 2022, arxiv:2203.05482 |
| **Weight Interpolation（权重插值）** | 在参数空间对两个或多个 SFT 变体的权重做线性插值：θ_merged = α·θ_A + (1-α)·θ_B | Z-Image/Z-Image-Turbo 技术报告 |
| **Decoupled Training + Merge（解耦训练再融合）** | 将不同任务维度的数据子集用不同随机种子独立训练，最后在参数空间合并 | FIBO 模型训练策略 |
| **Divide-and-Conquer SFT（分而治之微调）** | 按视觉风格/运动/场景等维度切分数据集，独立微调子模型后合并为单一模型 | Seedance 1.0 (ByteDance) |
| **LoRA Merging（LoRA 合并）** | 多个偏好特定的 LoRA 专家并行训练，迭代合并到共享基础模型 | MapReduce LoRA, CVPR 2026, arxiv:2511.20629 |
| **帕累托最优（Pareto-optimal）** | 在不损害其他维度表现的前提下，无法再提升某个维度——模型合并的核心目标 | 多目标优化理论 |

---

## 为什么 SFT 阶段需要模型合并？

### 问题根源：能力偏置（Capability Bias）

在对特定高质量数据做 SFT 时，模型会产生微妙的能力偏置：

- **照片级真实感 ↑** → 风格灵活性 ↓
- **指令遵循 ↑** → 美学渲染 ↓  
- **运动质量 ↑** → 视觉保真度 ↓
- **文本对齐 ↑** → 图像多样性 ↓

这些偏置来自：
1. **数据分布差异**：不同数据子集的质量、风格、场景分布天然不同
2. **损失景观（Loss Landscape）的单峰约束**：SFT 把模型推向局部最优点，但不同的数据子集对应不同的局部最优
3. **多目标冲突**：likelihood、aesthetic、alignment、diversity 是互相制约的目标

### 为什么不直接混合训练？

| 方案 | 优点 | 缺点 | 为何不采用 |
|------|------|------|-----------|
| **全量混合 SFT** | 简单直接 | 数据冲突导致能力中庸化，训练不稳定 | 多维度目标天然冲突，混合优化会导致"全面平庸" |
| **多阶段课程学习** | 有序引入数据 | 灾难性遗忘，后阶段覆盖前阶段能力 | SFT 的 MSE 目标对冲突梯度不友好 |
| **MoE 路由** | 专家分工 | 推理时额外开销，路由不稳定 | 图像生成的 latency 预算极紧（≤10s 端到端） |
| **Model Merge** ✅ | 零推理开销，保留各自优势 | 需要基座在同一 loss basin 内 | 当前最优方案 |

---

## 三大典型方案对比

| 维度 | Seedance 1.0 | Z-Image / Z-Image Turbo | FIBO | MapReduce LoRA |
|------|-------------|------------------------|------|----------------|
| **模型类型** | 视频生成 DiT | 图像生成 DiT | 图像生成模型 | T2I/T2V/LM 通用 |
| **合并层级** | 全模型权重合并 | 全模型权重线性插值 | 全模型权重合并 | LoRA 权重迭代合并 |
| **拆分维度** | 视觉风格 × 运动类型 × 场景 | 不同 SFT 能力侧重 | 图像条件控制 vs 纯文本 | 多偏好奖励维度 |
| **训练策略** | 子集独立 SFT + 早停 | 同一骨干微调多个变体 | 不同种子解耦训练 | LoRA 并行训练 + MapReduce 迭代 |
| **合并方式** | 全量权重平均 | 参数空间线性插值 α·θ₁+(1-α)·θ₂ | 权重空间直接合并 | 多轮 Reduce：逐步融合 LoRA 到基座 |
| **推理开销** | 🟢 零额外延迟 | 🟢 零额外延迟 | 🟢 零额外延迟 | 🟢 零额外延迟（合并后 LoRA 内化） |
| **关键创新** | 用更小 lr + 少量 GPU + 早停防止过拟合 | 在损失景观的同一 basin 内找平衡点 | 任务解耦后避免梯度冲突 | MapReduce 式迭代融合 + RaTE token embedding |
| **论文/来源** | ByteDance Seedance 1.0 技术报告 | Tongyi (Alibaba) | 内部训练策略 | arxiv:2511.20629, CVPR 2026 |

---

## 技术深度分析

### 1. 为什么权重平均有效？——损失景观理论

Model Soup 论文的核心发现：**同一预训练模型在不同超参数下微调后，权重通常位于损失景观的同一低误差盆地（low-error basin）内**。

这意味着：
- θ_A 和 θ_B 之间的线性插值路径上的每个点，loss 都不会显著升高
- 平均后的模型 θ_avg = (θ_A + θ_B)/2 天然落在盆地中央，具有更好的泛化能力
- 不需要重新训练、不需要推理时 routing

**对于图像/视频生成模型**，同一 DiT 基座在不同数据子集上 SFT 后，权重仍然共享大量共性（UNet/Transformer backbone 的通用先验），差异集中在少数层，因此平均操作几乎不损失各子模型的核心能力。

### 2. Z-Image 的线性插值策略

Z-Image 团队直接对多个 SFT 变体的完整权重做线性插值：

```
θ_merged = Σᵢ wᵢ · θᵢ    (Σwᵢ = 1)
```

这种方法的精妙之处：
- **无需额外设计**：不需要定义哪些层合并、哪些不合并
- **可以连续调节**：通过调整权重系数 wᵢ 可以实现不同能力侧重的连续过渡（如 Z-Image Turbo 在速度和质量的权衡）
- **前提条件**：所有变体必须来自同一基座、同一架构，且微调幅度不能太大（否则不在同一 basin）

### 3. MapReduce LoRA 的迭代合并

CVPR 2026 的 MapReduce LoRA 是当前最系统化的模型合并工作：

- **Map 阶段**：并行训练多个偏好特定的 LoRA 专家（如 aesthetic LoRA、alignment LoRA、OCR LoRA）
- **Reduce 阶段**：迭代地将 LoRA 权重融合到共享基础模型中
- **关键指标**：在 SD3.5 Medium 上 GenEval +36.1%、PickScore +4.6%、OCR +55.7%；HunyuanVideo 上运动质量 +90.0%

结合 RaTE（Reward-aware Token Embedding）在推理时实现灵活偏好控制。

### 4. Seedance 的训练细节

Seedance 的子模型训练策略值得关注：
- **更小的学习率**：比预训练阶段小，防止过度偏离 basin
- **少量 GPU**：每个子任务用较少算力，但并行训练总吞吐更高
- **早停机制**：防止过拟合，保持对文本指令的控制力
- **数据划分**：按视觉风格 × 复杂运动 × 场景三维度交叉切分

---

## 实践建议

### 什么时候该用 Model Merge？

| 场景 | 建议 |
|------|------|
| 多个高质量数据子集，混合训练效果不佳 | ✅ 独立 SFT + 合并 |
| 需要同时优化多个冲突维度（如美感 + 文本对齐） | ✅ 分维度 SFT + 合并（参考 MapReduce LoRA） |
| LoRA 微调场景，不想增加推理延迟 | ✅ 多个 LoRA 合并后再部署 |
| 只有一个数据源、单一优化目标 | ❌ 不需要合并 |
| 子模型差异太大（不同架构/基座） | ❌ 无法合并，不在同一 basin |

### 常见踩坑

1. **微调幅度过大**：如果某个子模型 SFT 太狠，它的权重会偏离共享 basin，合并后效果退化。解法：控制 lr 和训练步数（Seedance 的做法）
2. **盲目平均**：不是所有层都应该等权平均。实践中通常对浅层（通用特征）用平均，深层（任务特定）需要调权重或使用更复杂的合并策略（如 TIES-Merging、DARE）
3. **合并时序**：先 SFT 再合并 vs 边训练边合并（MapReduce 方式）效果差异大。MapReduce 的迭代合并通常优于一次性合并
4. **评估盲区**：合并后可能在综合指标上更好，但某些极端 case 退化。需要细粒度 per-dimension 评估

### 对于咱们的方向

做 Qwen Image Edit / Klein / FireRed Image Edit / HiDream O1 训练时：
- DPO/GRPO 阶段如果发现不同 reward model 之间冲突，考虑 MapReduce LoRA 式的多偏好合并
- SFT 阶段数据来源多样（编辑质量数据 vs 美学数据 vs 指令遵循数据），值得尝试分而治之 + 权重平均
- 图像编辑场景特有挑战：编辑一致性（Edit Consistency）vs 编辑自由度（Edit Freedom）是天然冲突维度，模型合并可能是解法

---

## 参考资料

- **Model Soups (ICML 2022)**：权重平均的理论基础。Wortsman et al., "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time", arxiv:2203.05482
- **MapReduce LoRA (CVPR 2026)**：当前最系统的多偏好模型合并方案，覆盖 T2I/T2V。Chen et al., "MapReduce LoRA: Advancing the Pareto Front in Multi-Preference Optimization for Generative Models", arxiv:2511.20629
- **Breaking Likelihood-Quality Trade-off (ICLR 2025 DeLTa)**：扩散模型中合并预训练专家的去噪轨迹切换方案。Esfandiari et al., arxiv:2511.19434
- **Seedance 1.0**：ByteDance 视频生成模型，分而治之 SFT + 模型合并的代表案例
- **Z-Image / Z-Image Turbo**：Tongyi (Alibaba)，参数空间线性插值方案
- **FIBO**：解耦训练 + 合并的代表案例
- **GitHub: OrthoStudio**：开源工具，支持 Flux/Chroma/Z-Image 等扩散模型的正交 delta 合并。lexterslab/orthostudio

> 注：Seedance 1.0、Z-Image、FIBO 的具体技术细节来自官方技术报告/博客，arXiv 上可能尚无完整论文。知乎/小红书未搜索到额外中文内容。


---

## 🔍 多模型交叉审核（2026-06-27）

以下为三模型并行审核的补充意见：

### 来自 GLM-5.2 — 工程落地 & 国产生态

                                  
     Merge（仅合并非冲突维度）——正是为了解决这个。DARE（Yu et al., 2023,        
     arxiv:2310.03075 实际是 arxiv:2311.03011）则通过随机丢弃大幅 delta         
     来稀疏化，减少干扰面。对于文档重点关注的图像生成场景，这些方法的适用性     
     差异（全模型合并 vs LoRA 合并）值得展开，而不是一笔带过。                  
                                                                                
     5. 缺少"如何验证子模型在同一 basin 内"的实操方法 + 工具链覆盖不全          
                                                                                
     文档反复说"前提是子模型在同一 loss basin                                   
     内"，但没说怎么验证。实操方法：在 θ_A 和 θ_B 之间做等间距线性插值（如      
     11 个点：0.0, 0.1, ..., 1.0），对每个插值点在验证集上算 loss，画 loss      
     interpolation curve。如果曲线是凸的（U 形或平坦），说明在同一              
     basin，可以安全合并；如果出现峰值，说明 basin                              
     边界被穿越，合并会出问题。这个方法来自 Model Soup                          
     原文的实验设计。工具链方面，文档只提了 OrthoStudio，漏了                   
     mergekit——这是目前最主流的开源合并工具（GitHub 8k+ stars），原生支持       
     Linear/SLERP/TIES/DARE/NearSwap/Task Arithmetic 等多种合并策略，且支持     
     diffusers 格式的扩散模型权重。对团队实际操作来说，mergekit                 
     的可及性远高于自己写插值脚本。                                             
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184951_31ab6b

Session:        20260627_184951_31ab6b
Duration:       50s
Messages:       2 (1 user, 0 tool calls)


### 来自 Gemini 3.1 Pro — 多模态交叉 & Google 生态

事实：SD/Flux 社区早就标配了 MBW (Merge Block Weighted)。在        
     DiT 中，浅层（Early Blocks）捕捉高频结构/Condition 控制，深层（Late        
     Blocks）负责低频色彩/美学，中间层（Middle Blocks）负责语义。               
     *   跟咱们的关联：对 Qwen Image Edit 这类模型，编辑保真度 和 生成美学      
     的冲突极大概率是分层的。你可以让模型                                       
     A（重编辑）只出浅层和中间层权重，模型 B（重美学）只出深层权重。实操中      
     $\alpha$ 必须是一个与层数对应的 Vector，而不是全局 Scalar。                
     4. DPO/GRPO 阶段的“防过拟合”合并（Model Ratcheting）                       
     你在实践建议里提到了 DPO，少了一个极关键的对齐期 trick：防 Reward          
     Hacking（在图像里就是饱和度溢出、油腻感、崩坏）。                          
     *   补充事实：在 RLHF/DPO 阶段，合并不仅是为了多目标，更是一种             
     强大的正则化手段。Google 团队在多篇对齐论文中用到类似 EMA / Spherical      
     Interpolation 的思路：把 DPO                                               
     训练到后期的模型（指令跟随极强但画质开始炸）与原始 SFT                     
     模型（画质好）按例如 0.2 : 0.8 的比例合并。                                
     *   跟咱们的关联：在搞 FireRed 或 Klein 的 DPO/OPD 时，不用非得硬调 KL     
     Penalty 系数。放手让它训，训完把 Checkpoint 往 Reference Model             
     方向拉一拉（插值），往往能在“听指令”和“高美感”之间切中完美的帕累托前沿     
     。                                                                         
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184951_f4f5d9

Session:        20260627_184951_f4f5d9
Duration:       44s
Messages:       2 (1 user, 0 tool calls)


### 来自 Claude Opus 4.6 — 论证严密性 & 结构完备性

                    
                                                                                
     Seedance 的策略是"更小 lr + 早停"，但逻辑上存在张力：                      
                                                                                
     - 早停通常用于防止泛化性能下降，但 SFT                                     
     的目标就是让模型"专精"于某个数据子集。早停会不会导致子模型本身都没训练     
     充分，合并后只是"几个半成品的平均"？                                       
     -                                                                          
     文中说"保持对文本指令的控制力"，这暗示早停是为了防止子模型丢失通用能力     
     。但如果每个子模型都要保留通用能力，那分而治之的意义何在？合并后的增益     
     是否只是"避免了灾难性遗忘"而非"能力叠加"？                                 
                                                                                
     建议补充：明确 Seedance 的早停标准（是 validation loss                     
     不再下降？还是固定步数？），以及早停后子模型的能力边界测试结果。           
                                                                                
     总结：草稿的核心论点（Model Merge                                          
     解决多维度不可能三角）成立，但在假设边界量化、横向对比数据、安全关键场     
     景、指标溯源四个方面需要补强，否则在面对"为什么不用                        
     MoE"或"合并后安全性如何"等追问时缺少防御深度。                             
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184951_1fe4dd

Session:        20260627_184951_1fe4dd
Duration:       43s
Messages:       2 (1 user, 0 tool calls)
