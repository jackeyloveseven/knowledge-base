# 为什么 Glance LoRA 会有效？

> 记录日期：2026-06-24
> 触发人：李承燕

## 一句话总结

Glance 不是蒸馏流水线，而是**按时间步分段训练的单图 LoRA**——在同一张图片上分别训练一个"Slow" LoRA（早期步，负责结构/布局）和一个"Fast" LoRA（后期步，负责细节/纹理），推理时串联切换。有效的原因是**解耦了不同去噪阶段的学习目标，避免了单 LoRA 的梯度冲突**。

---

## 核心概念定义

### Glance LoRA 是什么

一种**单图微调**技术，输入一张图片 + 一条 caption，输出**两个 LoRA 权重**：

| LoRA | 训练时间步范围 | 学习内容 | 推理时作用阶段 |
|------|---------------|----------|---------------|
| Slow LoRA | 早期步（高噪声，t ≈ 1.0 → 0.5） | 全局结构、构图、物体位置 | 推理前半段 |
| Fast LoRA | 后期步（低噪声，t ≈ 0.5 → 0） | 局部细节、纹理、边缘锐化 | 推理后半段 |

**与普通单图 LoRA 的区别**：普通 LoRA 对所有 timestep 一视同仁，一个权重学所有——Glance 把它拆成两个专门化的小 LoRA。

### 关键标志参数：`--flow_custom_timesteps`

Flow-matching 模型默认用 CDF（累积分布函数）采样 timestep，但 Glance 用自定义 timestep 范围（`--flow_custom_timesteps`），直接指定 Slow/Fast 各自训练的时间步区间。这在 Flow-matching 框架（Flux、SD3、Qwen-Image）里特别自然，因为 t ∈ [0,1] 有明确的语义映射。

### 为什么不是"蒸馏"

| 维度 | 真正蒸馏（如 SD3-Turbo） | Glance LoRA |
|------|------------------------|-------------|
| 训练信号 | Teacher 模型输出（output matching） | 原始图像 loss（同标准 fine-tune） |
| 数据需求 | 需要 Teacher 推理 N 步 | 单张图片即可 |
| 目标 | 减少推理步数 | 提升单图微调质量 |
| 关键机制 | 步蒸馏（step distillation） | 时间步分段训练（timestep splitting） |

Glance 更像"带分时调度的 LoRA"，不是蒸馏流水线。

---

## 为什么有效：三层原理

### 第 1 层：去噪过程的阶段性（扩散模型的固有特性）

扩散/flow-matching 去噪过程天然分两个阶段：

```
t=1.0 ──────────────────────────────────────────→ t=0
 │                                                   │
 ├─ 早期步 (high noise) ──┤── 后期步 (low noise) ──┤
 │                         │                        │
 └─ 学结构/布局/构图       └─ 学细节/纹理/边缘     └─
```

**具体到代码层面**（以 DiT/flow-matching 为例）：

- 早期步（t ∈ [0.7, 1.0]）：latent 接近纯噪声，模型预测的是**速度场的大方向**——决定物体在哪、形状大概什么样。这一步的 loss 主要来自**大尺度结构对齐**。
- 后期步（t ∈ [0, 0.3]）：latent 已经很接近目标，模型在**微调高频细节**——纹理、边缘、光照。这一步的 loss 主要来自**像素级/感知级匹配**。

一个**单一 LoRA** 要同时学好两个阶段，梯度方向天然存在张力——结构约束（全局一致性）和细节约束（局部保真度）会互相掣肘。

### 第 2 层：梯度冲突解耦

训练时，不同 timestep 的梯度方向经常冲突。举个例子：

- **t=0.9 时**：模型需要把 latent 往"有一个人在左边"的方向推 → 梯度方向 A
- **t=0.1 时**：模型需要把 latent 往"这个人的眉毛长这样"的方向调 → 梯度方向 B

在同一个 LoRA 里，A 和 B 求平均后可能既没学好结构也没学好细节。Glance 的做法是：

```
Slow LoRA: 只看 t ∈ [0.5, 1.0] 的梯度 → 专心学结构
Fast LoRA: 只看 t ∈ [0, 0.5] 的梯度 → 专心学细节
```

每个 LoRA 的优化目标更纯粹 → 同样的训练步数下收敛更快、效果更好。

### 第 3 层：单图场景下的过拟合缓解

单图训练的固有困境：只有一个样本，模型要么欠拟合（没学会），要么过拟合（学会了但崩溃了基座先验）。

Glance 通过**拆分任务**缓解这个问题：

- 每个 LoRA 只需要学一半的工作量 → 可以用更小的 rank
- Smaller rank = 更少参数改动 = 更不容易破坏基座模型
- 两个小 LoRA 叠加的效果 > 一个大 LoRA（在单图场景下）

类比：让一个人同时学画画的结构课和细节课 vs 两个分别专攻结构和细节的人协作——后者在单样本场景下效果更好。

---

## 技术实现要点

### 训练流程

```
输入: 单张图片 + caption
        │
        ├─→ Slow LoRA: 只在 t ∈ [T_split, 1.0] 上训练
        │     • loss = flow_matching_loss(pred, target, t ~ U(T_split, 1.0))
        │     • rank 较小（如 r=4~8）
        │
        └─→ Fast LoRA: 只在 t ∈ [0, T_split] 上训练
              • loss = flow_matching_loss(pred, target, t ~ U(0, T_split))
              • rank 较小（如 r=4~8）
```

### 推理流程

```
噪声 z_T
  │
  ├─ Step 1 ~ K/2: 加载 Slow LoRA
  │     • 负责 t ∈ [1.0 → ~0.5] 的去噪
  │
  ├─ Step K/2 ~ K: 切换到 Fast LoRA
  │     • 负责 t ∈ [~0.5 → 0] 的去噪
  │
  └─→ 输出图片
```

### `--flow_custom_timesteps` 的作用

Flow-matching 默认用 **CDF 采样** timestep：`t = CDF_inverse(u)`，u ~ U(0,1)。这导致 t 的分布偏向中间区域。

Glance 用 `--flow_custom_timesteps` 覆盖这个默认行为：
- Slow 训练：`--flow_custom_timesteps "0.5,1.0"` → 只在 t∈[0.5,1.0] 范围内均匀采样
- Fast 训练：`--flow_custom_timesteps "0.0,0.5"` → 只在 t∈[0.0,0.5] 范围内均匀采样

**为什么不用 CDF 采样？** CDF 采样会导致 Slow LoRA 的 t 分布偏中间（靠近 0.5），实际的"高噪声 regime"训练不够充分。自定义均匀采样确保每个 regime 都被充分覆盖。

---

## 技术深度分析

### 与相关方法的对比

| 方法 | 核心思路 | 推理开销 | 单图适用 | LoRA 数量 |
|------|---------|---------|---------|----------|
| 普通单图 LoRA | 全 timestep 训练一个 LoRA | 1x LoRA | ✅ | 1 |
| Glance LoRA | 分段训练两个 LoRA | 1x LoRA（串联切换） | ✅ | 2 |
| Lightning LoRA | 步蒸馏 + LoRA | 1x LoRA（更少步数） | 需要特定基座 | 1 |
| Step-aware LoRA (概念) | Timestep embedding 注入 LoRA | 1x LoRA | ❓ | 1（内部分段） |

**关键区别**：Glance 的推理延迟和普通 LoRA 一样——因为每次只加载一个 LoRA，只是中间切换一次权重。不需要额外的前向分支或适配器。

### Glance 为什么特别适合 Flow-Matching

Flow-matching 模型的 t ∈ [0,1] 映射是**线性的**：t=1 对应纯噪声，t=0 对应干净图像。这意味着：

1. **Timestep 有明确的物理意义**——不像 DDPM 的噪声 schedule 需要换算
2. **分段阈值 T_split 的选择更直观**——直接对应"结构→细节"的过渡点
3. **自定义 timestep 采样更干净**——不需要处理 β schedule 的非线性

这就是用户说的"适用于 flow-matching 模型（Flux、SD3 系、Qwen-Image 等）"的根本原因——Flow-matching 的 t-space 天然适合这种分段训练。

### 为什么"不是蒸馏"

蒸馏的本质是**知识迁移**：用 Teacher 的输出作为 Student 的训练目标。Glance 的训练目标和标准 fine-tune 完全一样（原图 loss），没有 Teacher 参与。它只是**通过分段训练来优化单 LoRA 做不到的事**——与其说蒸馏，不如说是"分治策略"（divide and conquer）。

---

## 实践建议

### 什么场景用 Glance

- ✅ 单图/极少样本（1-5 张）微调，追求极致质量
- ✅ Flow-matching 系列模型（Flux、SD3、Qwen-Image）
- ✅ 对推理延迟零容忍（不需要额外模块）
- ✅ 需要兼顾结构保真度和细节还原

### 什么场景不用

- ❌ 多图训练（几十张以上）——数据足够，单 LoRA 就够
- ❌ 非 flow-matching 模型（DDPM/DDIM 系）——timestep 语义不直接
- ❌ 追求推理速度的场景——Glance 不减少推理步数，只提升质量

### 关键超参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| T_split | 0.5（默认） | 结构/细节分界点；可调，0.3~0.7 |
| Slow rank | r=4~8 | 早期步参数少也够 |
| Fast rank | r=4~8 | 后期步同样 |
| 切换步 | 总步数的 1/2 | 和 T_split 对应 |

### 踩坑点

1. **T_split 选不好**：太大（0.7）→ Slow 学太多细节、Fast 无事可做；太小（0.3）→ Slow 没学到足够结构
2. **切换时的不连续性**：中间步切换 LoRA 可能导致 latent 出现轻微跳变——可以通过 overlap 区（如 t∈[0.45,0.55] 两边都参与）平滑
3. **两个 LoRA 的 rank 不一定要相等**：通常 Slow 可以更小（结构信息维度低）

---

## 参考资料

- ⚠️ 未搜索到 Glance 的公开 arXiv 论文
- ⚠️ 未搜索到 Glance 的公开 GitHub 仓库
- ⚠️ 未搜索到 HuggingFace 上的 Glance 模型
- 中文社区：未搜索到知乎/小红书相关讨论（搜索超时或被墙）

> 注：Glance 目前似乎是未正式发表的技术——可能存在于特定社区的实践经验或内部工具中。以上原理分析基于用户描述的技术特征（分段 LoRA + 自定义 flow timesteps + 单图训练）和扩散模型/flow-matching 的基础理论推导。

### 相关理论背景

- **Flow Matching for Generative Modeling** (Lipman et al., 2023) — flow-matching 理论基础，arXiv:2210.02747
- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021) — LoRA 原始论文，arXiv:2106.09685
- **SD3 / Flux** — flow-matching 在实际图像生成模型中的应用


---

## 🔍 多模型交叉审核（2026-06-27）

以下为三模型并行审核的补充意见：

### 来自 GLM-5.2 — 工程落地 & 国产生态

都要在 GPU 上，虽然小但 serving                 
     多用户时每个 user request 可能加载不同的 LoRA pair，累积起来不可忽略。     
     - 如果走硬切换方案（前半段只加载 Slow，后半段切换 Fast），则涉及 LoRA      
     热加载/卸载的延迟。diffusers 的 load_lora_weights 涉及磁盘 IO +            
     权重合并，在 API serving 场景下这个延迟会直接体现在首 token 延迟中。       
     - vLLM / TGI 等推理框架目前对图像生成模型的 LoRA                           
     多适配器动态切换支持有限（主要面向 LLM），Glance 这类 mid-inference        
     adapter swap 在现有 serving 框架里没有原生支持，需要自研调度逻辑。         
                                                                                
     5. 缺一个关键消融：2×(r=4) vs 1×(r=8) 的公平对比                           
                                                                                
     文中第 3 层原理说"两个小 LoRA 叠加 > 一个大                                
     LoRA"，但这个结论缺少对照实验支撑。公平对比应该是：总参数量相等的条件      
     下，2×(r=4) Glance vs 1×(r=8) 普通 LoRA。如果这个对比不显著，那 Glance     
     的增益可能纯粹来自"分段训练"这个正则化效果而非"双 LoRA                     
     结构"本身。文中把"拆分任务"和"更小                                         
     rank"两个因素混在一起说，没有区分哪个是主要贡献。这对判断 Glance           
     是否值得工程复杂度很关键——如果只是分段采样 timestep 就够了（单 LoRA        
     但只训特定 t 范围），就不需要双 LoRA 的推理切换逻辑。                      
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184958_df085a

Session:        20260627_184958_df085a
Duration:       1m 22s
Messages:       2 (1 user, 0 tool calls)


### 来自 Gemini 3.1 Pro — 多模态交叉 & Google 生态

nce 策略配合：避免“烤焦”的隐藏 Buff                         
     单图微调经常遇到的过拟合是“画面烤焦”（饱和度溢出、高频伪影）。             
     在低噪声阶段（Fast 阶段），网络本身就不太需要 Text                         
     Condition，如果继续强行给很大 Guidance 就会烤焦。Glance 用 Fast LoRA       
     专攻低噪后，实际上可以作为一种局部约束，让模型在后期减少对 Prompt          
     的依赖。                                                                   
     建议补充：实操中，如果把 Glance 和 Timestep-aware Guidance                 
     衰减（比如在 Flux 里让 guidance_scale 随 t 衰减）配合使用，Fast LoRA       
     阶段的泛化性和画质纯净度会发生质的飞跃。                                   
     4. 纠正工程误区：权重切换的显存与延迟代价                                  
     草稿里提到“每次只加载一个，推理延迟一样”，这在工程落地时过于乐观了。       
     具体技术事实：在 Diffusers 或 ComfyUI                                      
     的实际计算图中，去噪循环是一个连续的 GPU 任务。如果在 Step K/2             
     强行卸载 Slow 权重再加载 Fast 权重，会打断 CUDA Stream，引发 CPU-GPU       
     同步阻塞，导致严重的 latency                                               
     spike。如果要真正做到“零延迟切换”，就必须在初始化时把 Slow 和 Fast         
     两个 LoRA 的权重同时挂载在 VRAM 里，通过传入的 timestep 做 mask            
     动态路由（Dynamic Weight Injection）。所以，要提醒开发者：它的 VRAM        
     峰值占用实际上是普通单图 LoRA 的 2 倍。                                    
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184958_f39d8b

Session:        20260627_184958_f39d8b
Duration:       35s
Messages:       2 (1 user, 0 tool calls)


### 来自 Claude Opus 4.6 — 论证严密性 & 结构完备性

             
     -                                                                          
     结构-细节失配场景：若图片本身结构极简单但细节极复杂（如微距纹理照片）      
     ，Slow LoRA 可能欠训练（早期步没啥可学）→ 推理时 Slow LoRA                 
     输出接近噪声 → Fast LoRA 要从更混乱的起点开始细化 → 效果可能不如单         
     LoRA。需要给出识别这种场景的 heuristic（如 caption                         
     不含空间关系词、边缘检测后低频能量占比 < 阈值）。                          
     - T_split 选错的最坏后果：若 T_split=0.9，Slow LoRA 几乎学了全部，Fast     
     LoRA 只是微调 → 推理时切换后 Fast LoRA 的贡献可能被 Slow 的惯性淹没 →      
     出现"后半段生成停滞"的现象。这种情况下是否需要动态调整切换步数（如不是     
     总步数/2，而是根据 latent 变化率决定）？                                   
     - 显存不足时的 fallback：若推理环境无法同时存储两个 LoRA 的参数（虽然      
     rank 小，但在边缘设备上仍可能紧张），Glance                                
     是否可以只加载一个（牺牲质量换可用性）？需要说明在资源受限场景下的降级     
     方案。                                                                     
                                                                                
     总结：当前草稿的原理阐述清晰，但缺少量化证据支撑关键论断（梯度冲突、参     
     数效率）、关键超参（T_split）的理论依据不足、边界条件和失效模式未明确      
     。建议补充实验数据或理论推导，使"为什么有效"从"合理解释"升级为"可验证      
     的因果链"。                                                                
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184958_44ad42

Session:        20260627_184958_44ad42
Duration:       48s
Messages:       2 (1 user, 0 tool calls)
