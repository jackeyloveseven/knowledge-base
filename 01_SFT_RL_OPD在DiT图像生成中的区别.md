# SFT、RL 与 OPD 在 DiT 图像生成/编辑中的区别

> 📅 记录日期：2026-06-21 | 触发人：李承燕
> 问题：SFT 和 RL 与 OPD 的区别在哪，在 DiT 图像生成/编辑中又有什么不同

---

## 一句话总结

**SFT 教模型「模仿」、RL 教模型「探索并优化」、OPD 让模型「从更强的专家那里学习如何探索」——三者在 DiT 图像生成/编辑中构成从数据拟合到策略蒸馏的能力递增链条。**

---

## 核心概念定义

### SFT（Supervised Fine-Tuning）

通过对齐数据集做最大似然估计（MLE），最小化预测与 ground-truth 的 Cross-Entropy Loss。在图像生成中，SFT 让模型在给定 prompt + source image 的条件下预测目标图像。

- **数学本质**：$\min_\theta \mathbb{E}_{(x,y)\sim\mathcal{D}}[-\log p_\theta(y|x)]$，即模仿训练数据中的「正确答案」
- **信息来源**：静态数据集，固定分布
- **来源**：基础监督学习范式，在 LLM post-training 和 diffusion model fine-tuning 中广泛使用

### RL（Reinforcement Learning）

通过奖励信号指导模型探索解空间，优化策略以最大化累积奖励。在图像生成中，RL（包括 GRPO、DPO、RLHF 变体）让模型生成多样化的图像，并由 reward model 打分后梯度更新。

- **RL in Diffusion 的核心论文**：
  - **DDPO** (Black et al., 2023)：首个将策略梯度用于 diffusion model 的工作，用 PPO-style update 优化 denoising trajectory
  - **DPOK** (Fan et al., 2024)：将 RL 形式化为 diffusion 的 KL-constrained policy optimization
  - **GRPO for Diffusion**：Flow-GRPO 等将 Group Relative Policy Optimization 扩展到 flow matching，消除 critic 网络需求
  - **DPO for Diffusion**：Diffusion-DPO 将偏好优化直接用于 diffusion 输出对比较

- **信息来源**：reward model（可学习的打分模型）或 rule-based reward（如 OCR accuracy、aesthetic score）

### OPD（On-Policy Distillation）

On-Policy Distillation 是一种知识蒸馏范式：学生模型沿着**自己当前策略 roll-out 的轨迹**（on-policy）向教师模型学习，而非在教师预先生成的静态轨迹上蒸馏（off-policy）。

- **在 LLM 中的起源**：OPD 最早在 LLM 社区提出，用于将多个专项能力 teacher 蒸馏到一个通用 student。核心思路：学生的生成轨迹被 teacher 评分/修正，学生沿着自己探索的路径学习。
- **在 Diffusion 中的扩展**（DiffusionOPD, arXiv:2605.15055）：
  - 将 OPD 从离散 token 空间扩展到**连续扩散 Markov 过程**
  - 推导了 **closed-form per-step KL 散度目标**，针对 denoising transition

- **信息来源**：教师模型（已训练好的专项 expert），沿学生的 rollout 轨迹提供监督

---

## 关键区别/对比

| 维度 | SFT | RL（GRPO/DPO） | OPD |
|------|-----|----------------|-----|
| **优化目标** | MLE / Cross-Entropy Loss | 最大化 reward 期望（+ KL 约束） | 最小化学生-教师在学生轨迹上的 KL 散度 |
| **数据来源** | 静态数据集（固定分布） | reward model 实时打分 | 教师模型沿学生 rollout 轨迹输出 |
| **探索能力** | 无（模仿训练数据） | 强（reward 引导探索） | 强（但探索由学生主导，教师提供轨道修正） |
| **泛化性** | 差（记忆训练分布，OOD 退化） | 好（泛化到未见场景） | 更好（多教师知识融合，超越单一 reward） |
| **多任务能力** | 需要多任务混合数据集 | 联合优化有 reward conflict；级联 RL 笨重 | 天然解耦：教师各训各的，学生统一蒸馏 |
| **训练稳定性** | 稳定（MLE 梯度良好） | 中等（reward hacking、方差大） | 较好（closed-form KL 避免 score function 噪声） |
| **灾难性遗忘** | 严重（fine-tune 覆盖原知识） | 可控（KL 约束锚定原分布） | 更可控（教师多样性 + manifold anchor regularization） |
| **在 DiT 中的实现** | 标准 next-token/denoising prediction | PPO-style policy gradient 或 DPO preference pairing | per-step denoising KL，兼容 SDE/ODE sampler |
| **适用阶段** | 预热/初始化（必须的基础） | 单一能力精调 | 多能力融合、统一模型构建 |

---

## 技术深度分析

### 1. SFT 的本质困境：Mode Collapse & 泛化陷阱

来自 **SFT Memorizes, RL Generalizes** (Chu et al., 2025, arXiv:2501.17161) 的核心发现：

> SFT 倾向于记忆训练数据中的模式，在 OOD 场景下性能急剧下降。RL 通过 reward 引导的探索，学会的是**可泛化的能力**而非表面模式。

在 DiT 图像生成中的体现：
- SFT 后的模型在训练集 prompt 上效果好，但换个描述方式就崩
- 例如：SFT 训练「把猫变成狗」→ 只学会了训练集中的特定猫→狗转换风格，换一种毛色的猫就失配

**SED-SFT** (Chen et al., 2026, arXiv:2602.07464) 进一步分析：
> 传统 Cross-Entropy Loss 会导致 mode collapse——模型过度集中在某些输出模式上，限制了后续 RL 的探索效率。

**Quagmires in SFT-RL** (Kang et al., 2025, arXiv:2510.01624) 的警告：
> 高 SFT score ≠ 好的 RL 起点。SFT 分数可能偏向简单/同质数据，对后续 RL 增益不具预测性。某些情况下，SFT 后的模型做 RL 反而比 base model 直接 RL 效果更差。

### 2. RL 在 Diffusion 中的独特挑战

在 DiT 架构下做 RL 与 LLM 有本质不同：

**a) 连续状态空间 vs 离散 token 空间**
- LLM RL：离散 token 采样，概率分布直接可微分
- Diffusion RL：连续噪声空间，denoising trajectory 是多步 Markov chain
- 解决方案：DDPO 将 denoising 过程视为 MDP，每步 denoising 是一个 action

**b) Reward Sparsity（奖励稀疏）**
- Flow-OPD 论文指出（arXiv:2605.08063）：
  > Scalar-valued rewards 导致 reward sparsity，联合优化多个异构目标还会产生 gradient interference 和「跷跷板效应」

**c) 主流 RL for Diffusion 框架**

| 框架 | 方法 | 论文 |
|------|------|------|
| **DDPO** | PPO-style denoising policy gradient | Black et al., 2023 |
| **DPOK** | KL-constrained policy optimization | Fan et al., 2024 |
| **Flow-GRPO** | Group Relative Policy Optimization for flow matching | — |
| **Diffusion-DPO** | Direct Preference Optimization on image pairs | Wallace et al., 2024 |
| **DRaFT** | Differentiable reward through full denoising | Clark et al., 2024 |

### 3. OPD 在 Diffusion 中的核心创新

**DiffusionOPD** (Quanhao et al., 2026, arXiv:2605.15055, ali-vilab) 是 OPD 在扩散模型中最重要的系统性工作：

#### 核心公式

将 OPD 从离散 token KL 推广到连续 denoising KL：

在 step $t$，给定当前噪声状态 $\mathbf{x}_t$（学生自己 denoise 出来的），OPD 优化：

$$\min_\theta \mathbb{E}_{t, \mathbf{x}_t \sim \pi_\theta} \left[ D_{KL}\left( \pi_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) \,\|\, \pi_{\text{teacher}}(\mathbf{x}_{t-1}|\mathbf{x}_t) \right) \right]$$

对于 Gaussian denoising transition，这退化为**每步均值和方差匹配**：

$$\mathcal{L}_{\text{OPD}} = \mathbb{E}_{t,\mathbf{x}_t\sim\pi_\theta}\left[ \|\mu_\theta(\mathbf{x}_t, t) - \mu_{\text{teacher}}(\mathbf{x}_t, t)\|^2 \right]$$

#### 关键优势

1. **Closed-form 避免 score function 噪声**：PPO-style 需要估计 score function 导致高方差，OPD 用 analytic KL 避免了这个问题
2. **Sampler 兼容**：天然支持 stochastic SDE 和 deterministic ODE sampler
3. **多教师解耦**：每个教师专注单一任务达到性能天花板，学生统一蒸馏

#### 训练流程（DiffusionOPD）

```
Phase 1: 训练专项教师（RL，单 reward）
  ├── Aesthetics Teacher: GRPO + aesthetic reward
  ├── OCR Teacher: GRPO + OCR accuracy reward
  └── GenEval Teacher: GRPO + GenEval reward

Phase 2: OPD 多教师蒸馏
  for each round:
    ├── 采样多任务 prompt
    ├── 学生 roll-out denoising trajectory（on-policy）
    ├── 对应教师对每个 step 的 denoising transition 提供监督
    └── 累积所有 task 的 OPD loss，更新学生
```

#### 关键结果（SD3.5 Medium 上）

| 指标 | Baseline | GRPO (multi-reward) | **DiffusionOPD** |
|------|----------|---------------------|-------------------|
| GenEval | 63 | ~82 | **92** |
| OCR Accuracy | 59 | ~84 | **94** |
| Aesthetic | — | 提升 | 更好 |

### 4. OPD 的两个变体：Flow-OPD 与 D-OPSD

**Flow-OPD** (Fang et al., 2026, arXiv:2605.08063)
- 基于 Flow Matching 框架（非 DDPM）
- 引入 **Manifold Anchor Regularization (MAR)** 防止纯 RL 驱动的 aesthetic degradation
- 在 SD3.5 Medium 上 GenEval 63→92，OCR 59→94

**D-OPSD** (Jiang et al., 2026, arXiv:2605.05204)
- **不需要外部 reward model**：利用 VLM encoder 的 in-context 能力，同一模型扮演两个角色（教师/学生）
- 专为 step-distilled 模型（如 Z-Image-Turbo、FLUX.2-klein）设计
- 可以持续 fine-tune 而不丢失 few-step inference 能力
- 已验证场景：anime domain adaptation（全量微调）、LoRA customization（少量图像）

### 5. OPD vs DMD

容易混淆：OPD 和 DMD 都叫 distillation，但本质不同。

| | DMD | OPD |
|------|-----|-----|
| **目标** | 把多步模型蒸馏成少步（加速推理） | 把多教师能力蒸馏到一个学生（能力融合） |
| **蒸馏对象** | 原模型自身的分布 | 外部教师模型的知识 |
| **是否 on-policy** | 否（off-policy distribution matching） | 是（学生自己的 rollout 轨迹） |
| **核心论文** | DMD (Yin et al., 2023, arXiv:2311.18828) | DiffusionOPD (Li et al., 2026) |
| **应用场景** | 1-step / few-step 推理加速 | 多任务 unified model 构建 |

值得注意的是，**DMD meets RL** (arXiv:2511.13649) 表明 DMD 和 RL 可以互补——RL 提升生成质量，DMD 加速推理，两者可以级联使用。

---

## 在 DiT 图像生成 vs 图像编辑中的差异

### 图像生成（Text-to-Image）

| 方法 | 应用方式 | 典型场景 |
|------|----------|----------|
| **SFT** | 在 (prompt, image) pair 上做 denoising prediction | 基础生成能力、风格适应 |
| **RL** | reward model 评估生成图的美学/质量/alignment | 美学对齐、OCR 精度提升 |
| **OPD** | 美学老师 + OCR 老师 + GenEval 老师 → 统一生成模型 | 构建通用高质量 T2I 模型 |

### 图像编辑（Image-to-Image Editing）

| 方法 | 应用方式 | 难点 |
|------|----------|------|
| **SFT** | (source img, instruction, target img) triplet 上 MLE | mode collapse：只学会训练集中的编辑模式 |
| **RL** | reward 评估编辑准确性 + 原图保留度 | reward 设计复杂：需要同时衡量编辑成功度和 identity preservation |
| **OPD** | D-OPSD 的 self-distillation 模式天然适合编辑 | FLUX2-klein 上的编辑实验已验证可行 |

**图像编辑的特殊性：**
- 图像编辑需要在「改得对」和「改后还是原图」之间平衡
- SFT 改得不对就偏了；RL 的 reward 设计需要多维度；OPD 的多教师可以提供不同的评判视角
- D-OPSD 的 self-distillation 在 FLUX2-klein 编辑场景下已验证：不需要外部 reward，用 teacher-context 和 student-context 的对比来指导训练

---

## 实践建议

### 什么场景用什么

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| 刚拿到 base model，需要基本能力 | **SFT** | 稳定、可预期，打好基础 |
| 单一能力不够好（如 OCR 差） | **RL (GRPO)** | reward 明确，单任务优化高效 |
| 需要同时提升多个能力 | **OPD** | 避免 reward conflict，Multi-teacher 解耦 |
| 蒸馏过的少步模型需要持续 tuning | **D-OPSD** | 不需要外部 reward，保持 few-step 能力 |
| 美学和通用质量都差 | **SFT → RL → OPD** | 完整管线：SFT 打底 → RL 专项提升 → OPD 融合 |
| 只需要加速推理 | **DMD** | 1-step 生成，与上述方法正交 |

### 常见坑

1. **SFT 过度训练 → RL 起不来**：SFT 做太多会导致 plasticity loss，RL 阶段难以提升（When RL Fails after SFT, arXiv:2606.09932）
   - 解法：控制 SFT epoch 数，或使用 Rejuvenation（base-anchored model fusion）

2. **只看 SFT score 决定何时切 RL**：SFT score 高 ≠ RL 会好，反而可能过拟合（Quagmires paper）
   - 解法：用 generalization loss 和 Pass@large k 做 proxy

3. **多 reward 联合 RL → 「跷跷板」**：美学上去、OCR 下来
   - 解法：用 OPD 解耦，各教师独立优化到天花板再蒸馏

4. **OPD teacher 不够好 → student 上限被 cap**：OPD 蒸馏的质量取决于 teacher 质量
   - 解法：确保每个 teacher 在自己的单任务 RL 上跑到收敛

### 你的 DiT 项目参考

针对你目前用的 Qwen Image Edit 2511、Klein、FireRed Image Edit 1.1、HiDream O1 等模型：

- **SFT 阶段**：打基础，在编辑数据 triplet 上做标准 diffusers SFT
- **RL 阶段**：用 GRPO → 单 reward（如 CLIP-I + DINO structure loss 的加权组合）
- **如果要多能力**：例如同时要编辑精度 + 美学质量 + identity 保留 → OPD 多教师

---

## 参考资料

### 核心论文

1. **DiffusionOPD: A Unified Perspective of On-Policy Distillation in Diffusion Models** — Li et al., 2026, arXiv:2605.15055, ali-vilab
2. **Flow-OPD: On-Policy Distillation for Flow Matching Models** — Fang et al., 2026, arXiv:2605.08063
3. **D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models** — Jiang et al., 2026, arXiv:2605.05204
4. **SFT Memorizes, RL Generalizes** — Chu et al., 2025, arXiv:2501.17161
5. **Quagmires in SFT-RL Post-Training** — Kang et al., 2025, arXiv:2510.01624
6. **SED-SFT: Selectively Encouraging Diversity in SFT** — Chen et al., 2026, arXiv:2602.07464
7. **When RL Fails after SFT: Rejuvenating Model Plasticity** — Liu et al., 2026, arXiv:2606.09932
8. **Distribution Matching Distillation (DMD)** — Yin et al., 2023, arXiv:2311.18828
9. **DMD Meets Reinforcement Learning** — arXiv:2511.13649
10. **RL Fine-Tuning Heals OOD Forgetting in SFT** — Jin et al., 2025, arXiv:2509.12235

### 代码仓库

- **DiffusionOPD**: https://github.com/ali-vilab/DiffusionOPD （阿里，⭐108）
- **D-OPSD**: https://github.com/vvvvvjdy/D-OPSD （⭐255）
- **Flow-OPD**: https://github.com/CostaliyA/Flow-OPD
- **OpenDMD**: https://github.com/Zeqiang-Lai/OpenDMD （⭐184）
- **Awesome Diffusion RL**: https://github.com/YuanaHao/Awesome-Diffusion-RL

### 中文搜索

- 知乎/小红书：未搜索到相关高质量中文内容（搜索超时/被墙，已跳过）

---

> 📎 此文件位于 `items/knowledge-base/01_SFT_RL_OPD在DiT图像生成中的区别.md`
>
> ---
>
> ## 🔍 多模型交叉审核（2026-06-27）
>
> 以下为 DeepSeek V4 Pro 完成草稿后，由 GLM-5.2、Gemini 3.1 Pro、Claude Opus 4.6 并行审核的补充意见。已去重合并。
>
> ### 来自 Gemini 3.1 Pro — 多模态交叉 & Google 生态视角
>
> **1. RL 阶段的 Reward Model 正在被 VLM-as-a-Judge (RLAIF) 降维打击**
>
> 文档中 RL 部分还在用 Aesthetic 或 OCR accuracy 当 reward，但 Google Imagen 3 的对齐已全面转向用强 VLM（如 Gemini-1.5-Pro）提取 Dense Reward 或做细粒度偏好打分。VLM 能精准判断图文对齐度、空间关系（左边是猫右边是狗）、甚至物理常识。
>
> - 关联论文：**RichHF-18: Aligning Text-to-Image Models with Rich Human Feedback** (Google Research, CVPR 2024)。基于 VLM 预测的局部 heatmap reward（哪里画崩了打哪里）比全局标量打分在 RL 阶段效率高得多。
>
> **2. OPD 的工程落地硬伤：显存爆炸与 Semi-On-Policy 退让**
>
> DiT 的 OPD 是 On-Policy 的，意味着显存里要同时塞下一个 Student 走 forward/backward，外加 N 个 Teacher 走 inference。像 Klein 或 Qwen 这种大参数 DiT，多跑两个 Teacher 显存直接 OOM。
>
> - 实践解法：真实集群里通常退让为 Semi-On-Policy——把 Student 昨天的 checkpoint roll-out 轨迹存盘，今天拿 Teacher 离线打分并算 KL，明天再喂给 Student 优化。引入 lag 但救了命。
>
> **3. 图像编辑独有的 Reward 困境：Attention 泄漏**
>
> DiT 编辑模型做 RL 时，如果只给最终 RGB 图像算 reward，模型容易学到直接把 Source Image 的特征硬 copy 过去。高级 RL/OPD 做法需引入 Cross-Attention 层面的对齐约束。
>
> - 关联论文：**MagicBrush** (arXiv:2306.10012) 及后续编辑 RL 框架，显式对 Attention map 算 regularization，强迫模型知道哪个 token 对应图中哪块像素。
>
> **4. SFT 阶段的降维打法：Data-Centric Recaptioning**
>
> 文档写「SFT 容易 Mode Collapse」，但在切 RL 之前，工业界更常用的解法是用强 VLM 把粗糙的 triplet/pair 数据重新打上极其详尽的 Dense Caption。高质量 SFT 往往能直接越过部分 RL 需求。
>
> - 关联论文：**PixArt-α: Fast Training of Diffusion Transformer Models** (arXiv:2310.00426)——用 LLaVA 洗过的数据做 SFT，收敛速度和泛化性直接爆杀原始 LAION 数据。
>
> ### 来自 Claude Opus 4.6 — 论证严密性 & 结构完备性视角
>
> **5. OPD "超越单一 reward" 的机制推导缺失**
>
> 文档写 OPD 泛化性"更好（多教师知识融合，超越单一 reward）"，但没解释为什么。遗漏事实：
>
> - Reward conflict 的数学根源：多 reward 联合优化时梯度方向存在 cosine similarity < 0 的情况，导致 **gradient interference**（PCGrad, Yu et al. 2020）。OPD 绕过这个问题因为教师各自独立优化到 Pareto frontier，学生蒸馏时只需拟合教师分布，不参与梯度博弈。
> - 理论保证缺失：DiffusionOPD 只给了 empirical 结果（GenEval 63→92），未给出 PAC-learning 式泛化界。OPD 的泛化优势目前是实验观察，不是理论定理。
>
> **6. On-policy vs Off-policy 对比没讲透**
>
> OPD 定义里说了 on-policy vs off-policy，但对比表格里没单独拉出这个维度。这是 OPD 区别于传统蒸馏（如 **Progressive Distillation**, Salimans & Ho 2022）的核心特征：后者教师提前生成静态数据集，学生分布会 drift，静态数据不再覆盖学生的 OOD 区域。OPD 的核心优势是每轮沿学生当前策略 roll-out，教师在实际轨迹上提供监督，形成 closed feedback loop。
>
> **7. SFT→RL→OPD 管线的失效阈值**
>
> 文档推荐"完整管线：SFT 打底 → RL 专项提升 → OPD 融合"，但没提这个管线在什么条件下会挂。当 SFT 把 plasticity 耗光后，RL 提不动，后续 OPD 蒸馏的 teacher 质量也受影响，整条管线从入口就失败了（When RL Fails after SFT, arXiv:2606.09932）。
>
> **8. RL 探索的边界与代价：noise vs KL 约束 tradeoff**
>
> RL 在 Diffusion 中的探索能力被 KL 约束锚定——KL 太紧探索不足（退化成 SFT-like），KL 太松画面崩坏。文档没讨论这个 tradeoff。Griffin 等 (2024) 的 KL-adaptive 调度方案值得补充。
>
> **9. DMD（自蒸馏）vs OPD（跨蒸馏）的锐利区分**
>
> 文档 DMD vs OPD 对比表正确但不够锐利。DMD 的核心是 **self-distillation**：学生学的是原模型自身的分布，目标是压缩推理步数。OPD 是 **cross-distillation**：学生学的是外部教师的分布，目标是能力融合。两者的 loss 形式看似相似（都最小化分布差异），但角色完全不同。
>
> ### 来自 GLM-5.2 — 工程落地 & 国产生态视角
>
> **10. diffusers 框架对 RL/OPD 几乎零支持**
>
> - RL (DDPO/DPOK)：diffusers 有 `examples/research_projects/ddpo` 目录，但 Readme 标注"experimental"，不支持最新版 diffusers
> - OPD (DiffusionOPD/Flow-OPD)：完全没有 diffusers 官方支持，需要 fork 原论文仓库 + 手写 trainer
> - D-OPSD：官方用 PyTorch Lightning + LoRA，有自己的 trainer 但和 diffusers Trainer 不兼容
> - 这意味着你如果要在 Qwen Image Edit 上跑 RL/OPD，不是加个 training_args 就行——得重新搭训练管线
>
> **11. FlowGRPO 代码仓库实际为空**
>
> 文档引用 FlowGRPO 但 GitHub 上无公开实现（只搜到 1 个 0-star 的空模板仓库）。如果要写进去，必须标注"无可用代码仓库"，否则误导读者。
>
> **12. D-OPSD 的工程友好度明显高一档**
>
> - D-OPSD 支持 **LoRA custom**（28 张图 + 1k steps，3090 可跑）
> - 对比 DDPO/GRPO 需要 8×A100 跑完整 denoising trajectory
> - 你在用的 FLUX2-klein 如果走 D-OPSD 的 `flux2-klein_self-distill-edit` LoRA 路线，是成本最低的 OPD 实验入口
>
> **13. 训练成本表缺失**
>
> 「实践建议」给了方法论推荐但没给资源需求：
>
> | 方法 | 最低 GPU | 论文配置 | 备注 |
> |------|---------|---------|------|
> | DDPO | 4×A100 80G | 8×A100 | 完整 denoising trajectory 回传 |
> | Flow-GRPO | 4×A100 80G | 8×A100 | 无公开代码 |
> | DiffusionOPD | 4×A100 80G | 8×A100 | 多教师 round-robin 需频繁 swap |
> | D-OPSD (LoRA) | 1×24G | 1×3090 | 唯一单卡可跑路径 |
>
> **14. 补充代码仓库**
>
> - Flow-OPD 官方仓库：https://github.com/THU-Kingmin/Flow-OPD（代码质量高、结构清晰，但 star 为零——2026.5 才公开，不是代码质量问题）
> - D-OPSD 的 FLUX2-klein 编辑 LoRA：https://huggingface.co/vvvvvjdy/dopsd_flux2_kl_edit_lora
