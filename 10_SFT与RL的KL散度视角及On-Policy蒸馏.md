# SFT、RL 与 On-Policy 蒸馏：KL 散度视角的统一框架

> 记录日期：2026-06-23 / 更新：2026-07-28（+RL’s Razor、负样本假说检验、R1-Zero/R1-Distill 边界） | 触发人：李承燕
> 触发问题：「SFT 为什么常被理解成 Forward KL？RL/RLHF 为什么又更接近 Reverse KL？on-policy distillation 为什么和传统蒸馏不一样？」

---

## 一句话总结

**SFT = 最小化 Forward KL(p_data || p_model)，是 mode-covering（均值化）；RL/RLHF = 最小化 Reverse KL(p_model || p_optimal)，是 mode-seeking（聚焦高 reward 区域）；On-Policy Distillation 的核心差异在于学生用自己的 rollout 采样训练信号，解决传统蒸馏的 exposure bias — 传统 KD 是「老师做题、学生抄答案」，OPD 是「学生做题、老师当场批改」。**

---

## 1. 核心概念定义

### 1.1 Forward KL vs Reverse KL

给定两个分布 P 和 Q：

| | Forward KL: KL(P || Q) | Reverse KL: KL(Q || P) |
|---|---|---|
| **公式** | E_{x~P}[log P(x)/Q(x)] | E_{x~Q}[log Q(x)/P(x)] |
| **采样分布** | 从 P 采样 | 从 Q 采样 |
| **行为特征** | Mean-seeking / Mode-covering | Mode-seeking |
| **惩罚模式** | Q 在 P 有概率的地方概率低 → 重罚 | Q 在 P 概率极低的地方概率高 → 重罚 |
| **直观理解** | 「不能漏掉 P 的任何模式」 | 「不能生成 P 不认可的东西」 |

**来源：** MiniLLM (Gu et al., 2023) — arXiv:2306.08543，首次将 Forward/Reverse KL 框架引入 LLM 蒸馏

### 1.2 SFT 为什么是 Forward KL

SFT 的目标函数：

```
L_SFT = -E_{(x,y)~D}[log p_θ(y|x)]
```

等价于最小化：

```
KL(p_data || p_θ) = E_{x~D}[KL(p_data(·|x) || p_θ(·|x))]
```

- 采样分布是 **p_data**（训练数据），所以是 **Forward KL**
- 行为：模型试图覆盖训练数据中所有可能的输出模式
- 后果：当数据中有多个 valid 输出时，MLE 学出来的是它们的「平均」→ 可能生成模糊/平庸的结果
- **对应图像生成场景**：用 mixed-quality 数据做 SFT，模型学出来的是「所有质量级别的平均」→ 画面不够 sharp

**来源：** MiniLLM Section 3.1 明确推导了 SFT ↔ Forward KL 的等价性；OPD Survey (2604.00626) Section 2 再次确认

### 1.3 RL/RLHF 为什么是 Reverse KL

RLHF PPO 的目标：

```
max E_{x~D, y~p_θ}[r(x,y)] - β·KL(p_θ || p_ref)
```

- KL 惩罚项是 **KL(p_θ || p_ref)** — 采样分布是 p_θ（当前 policy），所以是 **Reverse KL**
- Reward 项推动 p_θ 向高 reward 方向移动
- Reverse KL 的 mode-seeking 特性：模型会聚焦于高 reward 的模式，可以主动放弃低 reward 区域
- KL 到 p_ref 的约束防止 policy 偏离太远导致 reward hacking

**关键直觉（来自 OPD Survey 2604.00626）：**

> "OPD as f-divergence minimization over student-sampled trajectories" — OPD 的 loss 本质上也是对学生自己采样轨迹的 f-divergence 最小化

RL 和 OPD 共享同一个骨架：**在学生/策略自己生成的样本上，用 teacher/reward 提供 dense 信号进行优化。**

**来源：** InstructGPT (Ouyang et al., 2022) — arXiv:2203.02155 PPO 目标；ExOPD (2602.12125) 显式将 OPD 建模为 KL-constrained RL

---

## 2. Forward KL vs Reverse KL：一张表说清

| 维度 | Forward KL (SFT) | Reverse KL (RL/RLHF/OPD) |
|------|-----------------|------------------------|
| **最小化目标** | KL(p_data \|\| p_model) | KL(p_model \|\| p_optimal) |
| **谁采样** | 数据分布（固定） | 模型自己（在线） |
| **分布行为** | Mode-covering（广撒网） | Mode-seeking（聚焦好的） |
| **信息论解释** | 最大化数据似然 = 最小化交叉熵 | 最小化 policy 与 optimal policy 的散度 |
| **训练稳定性** | 稳定（固定数据集） | 不稳定（分布随训练漂移） |
| **质量上限** | 受限于数据质量 | 理论上可超越数据 |
| **典型失败** | 「平均化」— 多峰数据学出模糊输出 | 「模式坍塌」— 丢掉合法但低 reward 的模式 |
| **在 LLM 中** | Pretrain SFT, Instruction SFT | RLHF PPO, GRPO, DPO |
| **在图像生成中** | 常规 SFT 微调 | GRPO/DPO 对齐训练 |

> 💡 补充：**DPO 本质也是 Reverse KL 视角下的优化**，只是把 reward 隐式参数化为 policy ratio，不需要显式 reward model，但优化方向同样是 mode-seeking。

---

## 3. On-Policy Distillation vs 传统蒸馏

### 3.1 传统蒸馏（Off-Policy KD）

```
流程：
1. Teacher 生成大量数据（offline）
2. Student 在这些 teacher 数据上做 SFT/KD
3. 推理时 Student 自己生成

问题（Exposure Bias）：
  - 训练时 Student 看到的都是「老师做对的 prefix」
  - 推理时 Student 自己生成 → 会犯错 → prefix 偏离训练分布
  - 错误累积 ≈ O(L²)，L 是序列长度
```

**来源：** OPD Survey (2604.00626) Abstract 直接指出：「exposure bias scales roughly with the square of sequence length」

### 3.2 On-Policy Distillation

```
流程：
1. Student 自己生成 rollout（on-policy）
2. Teacher 对这些 student 生成的 prefix 提供 token-level logit/score
3. 用 teacher 的分布作为 target 做 KL 最小化
4. Student 更新后，下一轮 rollout 分布已变化 → 真正 on-policy 闭环
```

核心差异一句话：

> **传统 KD 是「老师做题、学生抄答案」→ 学生从没见过自己做错的题。OPD 是「学生做题、老师当场批改」→ 学生在自己的错误上学习。**

**来源：** OPD Survey (2604.00626) Section 1 正式定义；GKD (Agarwal et al., 2023) — arXiv:2306.13649 提出统一框架

### 3.3 为什么 OPD 有效：分布论视角

来自「Post-Training is About States, Not Tokens」(2605.22731) 的核心论点：

```
SFT 训练状态 = p_data（固定的教师数据分布）
RL 训练状态 = p_policy（策略自己采样）
OPD 训练状态 = p_student（学生自己采样 + 教师当场监督）

OPD 的关键优势：训练状态来源是学生自己的 rollout，
但监督信号来自教师 → 既有 on-policy 的分布匹配，
又有教师的 denser supervision（比 RL 的 sparse reward 信息量大）
```

### 3.4 OPD 的 KL 散度设计演变

| 方法 | KL 形式 | 年份 | 关键贡献 |
|------|---------|------|---------|
| MiniLLM | Reverse KL(student \|\| teacher) | 2023 | 命名 OPD 领域，首次用 Reverse KL 做生成式蒸馏 |
| GKD | 灵活（forward/reverse/skew 可选） | 2023 | 统一 on-policy/off-policy mixture |
| DistiLLM | Skew-KL | 2024 | ICML 2024，自适应偏斜 + student-generated data |
| Veto | 中间目标分布 | 2026 | 在 logit 空间构造 stable 中间 target |
| Entropy-Aware OPD | Forward KL on high-entropy tokens | 2026 | 对高熵 token 用 forward KL 保持多样性 |
| ExOPD | Dense KL-constrained RL | 2026 | 将 OPD 显式建模为 KL 约束 RL，支持超越 teacher |
| Revisiting OPD | Truncated Reverse KL | 2026 | 截断 + teacher top-K support matching 解决不稳定 |

**来源：** awesome-on-policy-distillation (chrisliu298, GitHub 405 stars)，OPD Survey (2604.00626) Table 1

---

## 4. 技术深度分析

### 4.1 SFT = Forward KL 的严格推导

```
SFT loss: L = -E_{(x,y)~D}[log p_θ(y|x)]

展开：L = -E_{x~D}[ Σ_y p_data(y|x) log p_θ(y|x) ]
     = E_{x~D}[ H(p_data(·|x)) + KL(p_data(·|x) || p_θ(·|x)) ]
     = const + E_x[KL(p_data || p_θ)]

所以最小化 SFT loss ⇔ 最小化 Forward KL(p_data || p_θ)
```

### 4.2 RLHF PPO 中的 Reverse KL 约束

```
PPO objective:
max E[ r(x,y) - β·KL(π_θ(·|x) || π_ref(·|x)) ]

其中 KL(π_θ || π_ref) = E_{y~π_θ}[ log π_θ(y|x) - log π_ref(y|x) ]

这是 Reverse KL：采样方是 π_θ，目标方是 π_ref
```

如果把 reward 理解为 -log π_optimal（隐式 optimal policy），那么：

```
max E_{y~π_θ}[ -log π_optimal(y) - β·KL(π_θ || π_ref) ]
= min E_{y~π_θ}[ log(π_θ/π_optimal) + (β-1)·log(π_θ/π_ref) + const ]
= min KL(π_θ || π_optimal) + (β-1)·KL(π_θ || π_ref)  【当 β→1】

本质是 Reverse KL(π_θ || π_optimal) + 正则项
```

### 4.3 OPD 的 f-divergence 统一视角

OPD Survey (2604.00626) 的形式化：

```
OPD = min_θ E_{x~D, y~p_θ(·|x)}[ D_f(p_teacher(·|x,y_<t) || p_θ(·|x,y_<t)) ]

其中：
- 采样方 = p_θ（学生自己的 rollout） → on-policy
- 目标方 = p_teacher → teacher 提供 dense token-level 监督
- D_f = 任意 f-divergence（forward KL, reverse KL, skew-KL, JS...）
```

**与 RL 的关键区别：**
- RL：sparse reward（只在序列末尾给 reward）
- OPD：dense supervision（每个 token 位置都有 teacher logits）
- 结果：OPD 训练更稳定、样本效率更高，但受限于 teacher 能力上限

---

## 5. 实践建议

### 什么时候用 SFT（Forward KL）？

- 数据质量高且覆盖全面
- 不需要超越数据质量上限
- 追求训练稳定性
- 图像生成中的基础 SFT 微调

### 什么时候用 RL/GRPO（Reverse KL）？

- 需要超越 SFT 数据质量
- 有可量化的 reward 信号（如 CLIP score、aesthetic score）
- 能接受训练不稳定性和 reward hacking 风险
- Qwen Image Edit 的 GRPO 阶段

### 什么时候用 OPD？

- 有强 teacher 但 student 需要更小的推理成本
- 需要超越传统 KD 的质量上限
- Teacher 有 logits 输出（白盒）→ GKD/Veto；Black-box → GAD/OVD
- 图像生成领域目前较少用 OPD，因为 teacher logits 在连续 latent 空间不好定义。**但对离散 token 路线（如 VQ-based 模型）OPD 可以直接套用**

### OPD 的坑

| 坑 | 表现 | 解法 |
|----|------|------|
| **Exposure bias 反转** | 学生 rollout 太烂，teacher 无法有效纠偏 | 混合 on/off-policy（GKD）、speculative KD |
| **Diversity collapse** | Reverse KL 的 mode-seeking 导致输出单一化 | Entropy-Aware OPD（高熵 token 用 forward KL） |
| **Tokenizer mismatch** | Teacher/student 不同词表，logit 不能直接对齐 | ULD、GOLD、SimCT |
| **Length inflation** | 迭代 OPD 导致输出越来越长 | Stable-OPD divergence 约束 |

---

## 6. 跟咱相关的

### Qwen Image Edit / Klein 训练管线中

```
当前流程：SFT → GRPO（Reverse KL 模式）
可能的增强：
  SFT → OPD（用更大的 teacher 做 on-policy 蒸馏）→ GRPO
  或
  SFT → GRPO → OPD（用 GRPO 后的模型做 self-distillation 巩固）
```

**注意：图像生成的连续 latent 空间做 OPD 目前没有成熟方案**。但以下场景可以考虑：

- VQ-VAE tokenizer 路线：token 是离散的，可以直接套用 LLM 的 OPD 框架
- 如果用 DiT + 连续 latent：teacher/student 都在 latent 空间用 MSE 做 dense supervision（类似 OPD 的理念但非标准 KL 形式）

### 如果未来要做模型压缩

```
大模型（Qwen Image Edit 7B+）→ 小模型（1-2B）
传统 KD 的 exposure bias 在长序列图像生成中会更严重（latent patch 序列长）
→ 考虑 on-policy 方案（虽然连续空间需要适配）
```

---

## 7. RL’s Razor：RL 遗忘更少的关键不是负样本，而是 On-Policy 的低 KL 偏置

### 7.1 核心结论

MIT Improbable AI Lab 的 **RL’s Razor: Why Online Reinforcement Learning Forgets Less**（arXiv:2509.04259）比较了 SFT 与在线 RL。实验表明：在新任务性能匹配时，RL 对旧能力的破坏显著更少；真正预测遗忘的变量不是优化器名称、权重改变量、梯度秩或更新稀疏性，而是**在新任务输入分布上，微调模型相对基座模型的输出分布 KL**。

论文将这一原则称为 **RL’s Razor**：一个任务通常存在多个同样高奖励的解，on-policy RL 从模型自己的当前分布出发重加权高奖励输出，因此倾向于选择“能解题、同时离初始策略 KL 最近”的解；固定外部标签驱动的 SFT 则可能把模型拉向任意远端分布。

实验中，forward KL 对遗忘的二次拟合在 ParityMNIST 上达到 **R²=0.96**，在 LLM 实验中达到 **R²=0.71**。论文主体覆盖 Qwen2.5-3B-Instruct 的数学、科学问答、工具调用任务，OpenVLA-7B 机器人任务，以及三层 MLP 的受控实验。

### 7.2 “负样本让 RL 更强”为什么不是主解释

论文使用 2×2 消融拆开“是否 on-policy”和“是否使用显式负梯度”：

| 方法 | 采样方式 | 显式负梯度 | 结果 |
|---|---|---|---|
| GRPO | On-policy | 有 | 遗忘少、KL 小 |
| 1–0 Reinforce | On-policy | 无；正确样本权重 1，错误样本权重 0 | 表现接近 GRPO |
| SFT | Offline | 无 | 遗忘多、KL 大 |
| SimPO | Offline | 有 | 表现接近 SFT |

如果负样本是决定因素，GRPO 应与 SimPO 聚类，1–0 Reinforce 应与 SFT 聚类；实际结果按 on-policy/offline 分组。因此，论文强力反驳了“负样本是 RL 低遗忘优势的主因”。

但严谨结论不是“负样本已被彻底证伪”。论文证明的是：**负样本不是低遗忘优势的必要条件，也不是该实验中的主导变量**。负样本对样本效率、奖励塑形和偏好边界仍可能有效；论文没有覆盖所有 DPO/SimPO 变体、frontier-scale 模型和全部任务域。

### 7.3 Oracle SFT 与 RL Teacher 蒸馏：目标分布比优化器更重要

作者在 ParityMNIST 上构造了一个 oracle SFT 标注分布：它在达到 100% 新任务准确率的条件下，与基座分布的 KL 最小。用该分布训练的 SFT 模型，旧能力保持甚至优于 RL。

作者还让 RL 训练后的模型生成数据，再从基座初始化训练 SFT student。该 student 在“新任务准确率—旧任务保持率”上，在噪声范围内匹配 RL teacher。这说明 RL 的优势可以被 SFT 复制：关键是 RL 先找到了一个高奖励、低 KL 的目标分布。

这里必须区分概念：**RL teacher 生成固定数据后再训练 student，仍然是 off-policy 蒸馏，不是 on-policy SFT**。数据来自固定 teacher，而不是 student 当前策略；student 更新后，两者会进一步错位。准确表述应是：

> 先用 on-policy RL 搜到 KL-minimal 的高奖励分布，再用 off-policy SFT/RFT 复制和扩散该分布。

### 7.4 与 DeepSeek-R1-Zero / R1-Distill 的关系

RL’s Razor 可以作为 R1-Zero 路线的**事后机制解释**，但不能被表述为 DeepSeek 当时的直接动机。DeepSeek-R1 技术报告明确写道：绕过 RL 前的 SFT，是因为团队假设人类定义的推理轨迹会限制探索；直接从 DeepSeek-V3-Base 做 GRPO，是为了让验证、反思和替代路径等推理行为自然涌现。官方主因是探索空间与能力涌现，而不是防止灾难性遗忘。

RL’s Razor 补充的解释是：从 Base 直接做 on-policy GRPO，可能在探索新推理策略的同时，因低 KL 偏置而较少破坏基座能力。二者路线相容，但不能倒置因果；RL’s Razor 发表于 2025 年 9 月，DeepSeek-R1 报告发表于 2025 年 1 月。

DeepSeek 随后使用 R1 生成并筛选约 **80 万条**样本，仅用 SFT 训练 Qwen/Llama 基座。DeepSeek-R1-Distill-Qwen-32B 在 AIME 2024 pass@1 达到 **72.6**，而对 Qwen2.5-32B-Base 进行超过一万步大规模 RL 的 Qwen2.5-32B-Zero 为 **47.0**。工程含义是：强 RL teacher 负责昂贵的搜索，小模型用 SFT 复制已发现的高质量分布，通常比让小模型自己重新做大规模 RL 更划算。

不过，这不是 RL’s Razor 同基座实验的严格复现：DeepSeek 的 teacher/student 容量与架构不同，数据还经过 rejection sampling 和规则过滤。它只能算更广义的工程印证。

### 7.5 对图像生成/编辑训练的启发

对 Qwen Image Edit、Klein、FireRed 一类模型，结论不是简单地“少做 SFT、多做 GRPO”，而是把**新能力增益—基座分布漂移—旧能力保持**作为共同评价轴。

建议做四组同预算实验：offline SFT、offline preference、on-policy 正样本过滤、on-policy GRPO。横轴使用新任务质量/编辑成功率，纵轴使用旧 benchmark retention，并用 base-to-finetuned 的输出或特征分布距离着色。这样能区分收益究竟来自 reward objective、负样本，还是 on-policy 采样。

Teacher 数据筛选也不应只看 reward。若 teacher 输出位于 student/base 很难到达的远端模式，SFT 会产生更大分布漂移；更理想的数据同时满足高奖励、可达性高、相对基座漂移小。工程范式可概括为：**RL Search → 低 KL 轨迹筛选 → SFT/RFT Distill**。

### 7.6 论文边界

- 主体 LLM 规模为 Qwen2.5-3B-Instruct，尚未验证 frontier-scale 模型。
- 作者未研究 online-but-off-policy RL，也未覆盖更多生成域。
- KL 与遗忘之间的表征干扰机制尚未完全解释；目前最强证据是跨方法预测关系、受控消融和简化条件下的理论推导。
- “On-policy 通常低 KL”是优化偏置，不是保证；reward hacking、探索不足和过强更新仍可能导致失败。

---

## 8. 参考资料

### 核心论文
- **RL’s Razor**: Shenfeld, Pari & Agrawal, "RL’s Razor: Why Online Reinforcement Learning Forgets Less" — arXiv:2509.04259 (2025)
- **SFT Memorizes, RL Generalizes**: Chu et al. — arXiv:2501.17161 (2025)
- **DeepSeek-R1**: "Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" — arXiv:2501.12948 (2025)
- **OPD Survey**: Song & Zheng, "A Survey of On-Policy Distillation for Large Language Models" — arXiv:2604.00626 (2026.04, v4 2026.06)
- **MiniLLM**: Gu et al., "MiniLLM: Knowledge Distillation of Large Language Models" — arXiv:2306.08543 (2023)
- **GKD**: Agarwal et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes" — arXiv:2306.13649 (2023)
- **InstructGPT**: Ouyang et al., "Training Language Models to Follow Instructions with Human Feedback" — arXiv:2203.02155 (2022)
- **ExOPD**: "Learning beyond Teacher via Generalized On-Policy Distillation with Reward Extrapolation" — arXiv:2602.12125 (2026)
- **Post-Training is About States**: "A State Distribution View of SFT, RL, and On-Policy Distillation" — arXiv:2605.22731 (2026)
- **On the Geometry of OPD**: arXiv:2606.07082 (2026)
- **DistiLLM**: Ko et al., "Towards Streamlined Distillation for LLMs" — arXiv:2402.03898 (ICML 2024)

### 代码仓库
- **awesome-on-policy-distillation**: chrisliu298/awesome-on-policy-distillation — GitHub, 405 stars, 374 entries (2026)
- **Awesome-LLM-On-Policy-Distillation**: nick7nlp/Awesome-LLM-On-Policy-Distillation — GitHub, 363 stars
- **MiniLLM 代码**: microsoft/LMOps/tree/main/minillm
- **GKD 参考实现**: google-research (speculative_kd)

### 博客 / 深度文章
- **Thinking Machines**: "On-Policy Distillation" — thinkingmachines.ai/blog/on-policy-distillation (2025)
- **OPD 深度解析 (中文)**: 知乎专栏 — zhuanlan.zhihu.com/p/2033212181823608430 (2026)
- **重温 On-Policy Distillation (中文)**: yph22.github.io/files/notes/OPD.pdf (2026)

### 中文社区（已搜索，未能获取内容）
- 知乎搜索「SFT RLHF forward KL reverse KL 蒸馏」：请求超时，未获取到内容
- 小红书搜索「SFT RLHF KL 散度 蒸馏」：请求超时，未获取到内容


---

## 🔍 多模型交叉审核（2026-06-27）

以下为三模型并行审核的补充意见：

### 来自 GLM-5.2 — 工程落地 & 国产生态

Initializing agent...
────────────────────────────────────────


┌─ Reasoning ──────────────────────────────────────────────────────────────────┐
用户让我以技术审核专家身份，从「中文技术社区认知、国产模型生态、模型部署工程实践」三个维度，对这篇知识库草稿提出3-5条补充意见。只说遗漏或不够深入的部分，不复述已有内容，每条以具体技术事实支撑。

让我仔细审阅这篇草稿，找出在我擅长的三个维度上的遗漏点。

维度1：中文技术社区认知
维度2：国产模型生态
维度3：模型部署工程实践

让我逐条思考遗漏：

  ... (150 more lines — /reasoning full to show)
└──────────────────────────────────────────────────────────────────────────────┘
 ─  ⚕ Hermes  ───────────────────────────────────────────────────────────────── 
                                                                                
     审核意见（从中文技术社区认知、国产模型生态、模型部署工程实践三个维度）     
     ：                                                                         
                                                                                
     1. DeepSeek-R1 蒸馏路线是 OPD 的关键反例，草稿未讨论工程成本权衡           
                                                                                
     DeepSeek-R1 技术报告明确写了：用 R1 生成约 80 万条 CoT 数据，直接对        
     Qwen2.5-7B/14B/32B 和 Llama-8B 做 SFT，全程 off-policy，没用 on-policy     
     蒸馏。结果这些蒸馏模型在 AIME 等推理 benchmark                             
     上表现优秀，开源后被社区广泛使用。这是当前最成功的国产模型蒸馏案例，却     
     恰恰走了 OPD 文献批评的 off-policy                                         
     路线。原因是工程层面的核心权衡：on-policy 蒸馏要求 teacher 对 student      
     的每个 rollout 做实时前向传播提供 token-level logits，teacher 和           
     student 必须同时在线，GPU 占用翻倍且无法离线批量；而 off-policy SFT        
     可先批量生成数据再训练，工程上简单一个数量级。草稿在"什么时候用            
     OPD"只讲了理论条件，没有这个工程成本决策。建议补一节 OPD vs off-policy     
     KD 的工程成本对比。                                                        
                                                                                
     2. 缺梯度视角：Forward/Reverse KL                                          
     的梯度采样方不同，直接解释训练稳定性差异                                   
                                                                                
     草稿的 KL                                                                  
     分析全在信息论层面（谁采样、mode-covering/seeking），缺一个关键的优化      
     层面视角：Forward KL 的梯度 ∇KL(p‖q) 的蒙特卡洛估计在 p                    
     分布下采样，梯度方差由 teacher/data 控制；Reverse KL 的梯度在              
     q（student/policy）分布下采样，方差由 student                              
     自己的采样质量决定。这意味着 RL/OPD                                        
     训练不稳定不只是"分布漂移"导致，更根本的是梯度估计天然方差更高——studen     
     t                                                                          
     早期采样质量差时梯度噪声爆炸。这个视角来自苏剑林（科技苑）的《KL散度与     
     生成模型》系列博文，是中文社区对这个话题最清晰的原创贡献之一。建议在       
     §4 补一个梯度视角小节。                                                    
                                                                                
     3. 图像生成领域 OPD 已有成熟实践，草稿"没有成熟方案"的说法不准确           
                                                                                
     草稿 §6 说"图像生成的连续 latent 空间做 OPD                                
     目前没有成熟方案"。实际上连续扩散空间里的 on-policy                        
     蒸馏是成熟且广泛使用的：Consistency Distillation（Song et al., 2023,       
     arXiv:2303.01419）中学生用自己的采样点沿 ODE 轨迹匹配 teacher              
     的去噪目标，这就是标准 on-policy distillation                              
     在连续扩散空间的形式；Flow Matching 的 reflow/rectification（Liu et        
     al., 2023, arXiv:2210.02747；SD3 论文 §5.2）中 student                     
     在自己采样的点上对 teacher 的 velocity 场做回归逐步拉直轨迹，也是          
     on-policy 的。这两类方法在知乎中文社区（关于 Flow Matching 和              
     Consistency Model 的专栏分析）有大量讨论。建议修正 §6                      
     表述，补充这两类作为图像生成 OPD                                           
     的已有实践。对做图像生成训练来说，这比"VQ-based 模型才能套用 LLM 的        
     OPD 框架"更直接相关。                                                      
                                                                                
     4. GRPO 相对 PPO 的工程优势未提及                                          
                                                                                
     草稿多次提到 GRPO 但没讲它相对 PPO 在工程上的关键差异：GRPO 去掉了         
     value network（critic），用 group baseline 代替 GAE 的 advantage           
     估计。训练时少一个与 policy 同等规模的模型占显存，对 7B+                   
     的图像生成模型（如 Qwen Image Edit）省的显存非常可观。这个改进由           
     DeepSeek 在 DeepSeekMath（arXiv:2402.03300）中提出，是 Reverse KL          
     训练框架在国产模型生态中的标志性工程优化。在图像生成领域用 GRPO            
     时，省掉 critic 还附带好处：不用训一个难以收敛的图像 value model——图像     
     reward 稀疏加空间维度高，critic 比 LLM 更难训。建议在 §2 表格或 §5         
     补一句 GRPO 的工程差异。                                                   
                                                                                
     5. OPD 模型部署阶段的 OOD 行为差异完全没讨论                               
                                                                                
     草稿聚焦训练阶段，但 on-policy 蒸馏的模型在推理部署时和 SFT                
     模型行为不同：SFT 的 forward KL 强制覆盖所有数据模式，天然有一定 OOD       
     覆盖；OPD 的 reverse KL 聚焦高概率区域，OOD                                
     处可能概率塌陷。国产模型生态中的典型案例：DeepSeek-R1-Distill              
     系列在非数学/代码任务上相比 R1                                             
     有明显掉点，部分原因就是蒸馏数据分布集中。模型服务化时这个差异需要考虑     
     ——OPD 模型对推理 prompt/条件的分布偏移更敏感。建议补一节 OPD               
     模型的部署/推理特性。                                                      
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184720_65a862

Session:        20260627_184720_65a862
Duration:       1m 19s
Messages:       2 (1 user, 0 tool calls)

### 来自 Gemini 3.1 Pro — 多模态交叉 & Google 生态

Initializing agent...
────────────────────────────────────────


┌─ Reasoning ──────────────────────────────────────────────────────────────────┐
**Reviewing SFT, RL, and OPD**

I'm currently analyzing the draft knowledge base entry, focusing on the intricacies of Supervised Fine-Tuning (SFT), Reinforcement Learning (RL) through the lens of KL divergence, and On-Policy Distillation (OPD). My next step is to formulate 3-5 targeted supplementary points to enhance its technical depth and clarity.

**Identifying Gaps and Gaps**

Bro, I'm digging into the draft, mostly looking for what's missing or could be deeper, focusing on cross-modal tech, Google ecosystem stuff, and knowledge breadth. Need to find some solid technical facts to flesh out these omissions.

**Expanding Google Ecosystem Knowledge**

  ... (29 more lines — /reasoning full to show)
└──────────────────────────────────────────────────────────────────────────────┘
 ─  ⚕ Hermes  ───────────────────────────────────────────────────────────────── 
                                                                                
     兄弟，这篇笔记把文本端 SFT/RL/OPD 的 KL                                    
     散度视角理得非常清，骨架很漂亮。但结合你平时搞的图像生成（尤其是           
     DMD/OPD）以及最新的理论进展，有几个维度还可以往深挖，我给你补 3            
     条硬核的：                                                                 
     1. 多模态交叉补全：DMD 与 LCM 就是视觉版的双向 KL 与 OPD（修正第 6         
     节）                                                                       
     你在 6.1 说「图像生成的连续 latent 空间做 OPD                              
     目前没有成熟方案」，这句稍微保守了点。实际上你日常关注的                   
     DMD（Distribution Matching Distillation） 和 LCM（Latent Consistency       
     Models） 完美对应了这套理论：                                              
     *   DMD 是双向 KL 散度的极致应用：DMD 的核心目标就是 $D_{KL}(p_{data}      
     || p_{student}) + D_{KL}(p_{student} || p_{teacher})$。第一项是            
     Forward KL（用 Fake-Score 算，确保                                         
     mode-covering，不糊不丢样），第二项是 Reverse KL（用 Real-Score 算，做     
     mode-seeking，拉近和老师分布的距离）。DMD                                  
     证明了在连续扩散空间里，Forward 和 Reverse KL 组合起来能实现 One-step      
     生成。                                                                     
     *   LCM 本质就是视觉的 On-Policy Distillation：LCM                         
     训练时，学生模型自己在 ODE 轨迹上采样（Student                             
     Rollout），然后老师模型（预训练 Diffusion）计算一步到多步的                
     Consistency 作为监督。这完全契合你总结的「学生做题，老师当场批改」的       
     OPD 范式。                                                                 
     2. Google 生态理论补充：IPO 对 Reverse KL 坍塌的降维打击（补全第 4.2 /     
     5 节）                                                                     
     在分析 RLHF / DPO（Reverse KL                                              
     视角）导致的模式坍塌和长度膨胀（Diversity Collapse / Length                
     Inflation）时，漏了 Google DeepMind 23 年的核心论文 IPO（Identity          
     Preference Optimization, Azar et al.）。                                   
     *   技术事实：DPO 严格依赖 Reverse KL 约束，DeepMind 在 IPO                
     论文里从数学上证明了，只要目标分布是确定性的，DPO 这种 Reverse KL          
     公式必然导致策略极度过拟合（Mode                                           
     Collapse），遇到矛盾的偏好数据时会直接发散。                               
     *   解法差异：相比你提到的用 Entropy-Aware 打补丁，IPO 放弃了 KL           
     散度正则化，把偏好对齐改写成了                                             
     Root-finding（求根）问题。加这笔可以拉高文章的理论天花板：解决 Reverse     
     KL 缺陷的终极方案可能是不去算 KL。                                         
     3. 知识广度延展：f-DPO 视角的散度泛化（补全第 2 节的 DPO 备注）            
     你在第 2 节末尾提了「DPO 本质是 Reverse                                    
     KL」，但目前的理论前沿已经突破了这点，建议补充 f-DPO (Wang et al.,         
     2023) 的结论。                                                             
     *   技术事实：DPO 的公式已经被推倒并泛化到了所有                           
     f-divergence。如果你把 f-DPO 的底层散度从 Reverse KL 换成 Forward KL       
     或者 Jensen-Shannon Divergence (JSD)，DPO 的行为就会从 Mode-seeking        
     突变成 Mode-covering。                                                     
     *   实践意义：这意味着你可以用 Forward KL 的性质来跑                       
     RLHF/DPO。如果你的 Qwen Image Edit 训练在 GRPO                             
     阶段发现画风太单一（模式坍塌），你不需要退回 SFT，直接用 Forward KL        
     形式的 f-DPO 目标函数，就能在保持对齐的同时强行撑开多样性。这比调          
     Temperature 管用得多。                                                     
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184720_aefbc2

Session:        20260627_184720_aefbc2
Duration:       39s
Messages:       2 (1 user, 0 tool calls)

### 来自 Claude Opus 4.6 — 论证严密性 & 结构完备性

Initializing agent...
────────────────────────────────────────

 ─  ⚕ Hermes  ───────────────────────────────────────────────────────────────── 
                                                                                
     审核完毕。以下是我从论证严密性、结构化对比完备性、安全边界、逻辑链条检     
     验四个维度提出的补充意见：                                                 
     1. Forward KL 的 mode-covering 行为在图像生成中的失效边界未说明            
                                                                                
     条目中提到"用 mixed-quality 数据做                                         
     SFT，模型学出来的是「所有质量级别的平均」→ 画面不够                        
     sharp"，但未说明为什么连续 latent 空间的 mode-covering 行为和离散          
     token 空间不同。                                                           
                                                                                
     技术事实：                                                                 
     - 在 LLM 的离散 token 空间，Forward KL 的 mode-covering                    
     表现为"多峰数据导致词汇选择平庸化"（如多个合法续写被平均成低置信度的混     
     合分布）                                                                   
     - 在图像生成的连续 latent 空间（DiT/U-Net），mode-covering                 
     的表现是空间域的模糊（spatial blurring）而非分布多峰性——因为 L2 loss       
     本质就是 Gaussian 假设下的 Forward KL，多个 sharp                          
     图像的平均在像素/latent 空间直接表现为模糊                                 
     -                                                                          
     关键差异：离散空间的"平均"是概率分布的熵增，连续空间的"平均"是几何平均     
     导致的细节丢失                                                             
                                                                                
     建议补充： 在 Section 5 实践建议中增加一条脚注，说明图像生成领域           
     Forward KL 的失效模式和 LLM 不同，以及为什么 GRPO                          
     在图像生成中能有效解决（因为 reward 可以直接惩罚模糊）。                   
     2. Reverse KL 的 mode collapse 风险在 GRPO 中的实际表现和缓解机制缺失      
                                                                                
     条目中提到"模式坍塌 — 丢掉合法但低 reward 的模式"，但未说明 GRPO           
     在图像生成中是否真的观察到了严重的 diversity collapse，以及 β（KL          
     惩罚系数）如何调节这个权衡。                                               
                                                                                
     技术事实：                                                                 
     - InstructGPT/PPO 的 β 通常在 0.01-0.1，β 越小越容易 mode collapse         
     - GRPO 在图像生成中的 β 调参规律和 LLM 不同：图像的                        
     reward（CLIP/aesthetic）比 LLM 的 human preference 更加 dense 和           
     smooth，理论上更不容易 reward hacking，但实际 β                            
     如何设置缺乏公开的消融研究                                                 
     - Qwen Image Edit 2511 的技术报告中未披露 GRPO 的 β 值和 diversity         
     监控指标（如生成图像的 LPIPS 多样性）                                      
                                                                                
     建议补充： 在 Section 4.2 或 5 中增加一段"GRPO 中 β 的调参经验和           
     diversity 监控"，说明：                                                    
     1. 图像生成中 β 的典型范围（如果 Qwen/Klein 有内部数据可以参考）           
     2. 如何通过 LPIPS diversity 或 cluster-level reward 监控 mode collapse     
     3. 如果没有公开数据，明确标注"缺乏公开消融研究"作为未来补充方向            
     3. OPD 在连续 latent 空间的适配方案未给出可行性验证的技术路径              
                                                                                
     条目在多处提到"图像生成的连续 latent 空间做 OPD                            
     目前没有成熟方案"，但同时又在 Section 5 和 6 中建议"DiT + 连续 latent      
     用 MSE 做 dense supervision（类似 OPD 的理念但非标准 KL                    
     形式）"——这个类比缺乏严格的数学等价性验证。                                
                                                                                
     技术事实：                                                                 
     - LLM 的 OPD 本质是对离散分布的 KL 最小化：KL(p_teacher(y_t|x, y_<t)       
     || p_student(y_t|x, y_<t))，其中 y_t 是离散 token                          
     - 图像生成的 MSE loss（如 ||z_teacher - z_student||^2）在连续 latent       
     空间等价于Gaussian 假设下的负对数似然，即 log N(z_student | z_teacher,     
     σ^2·I)                                                                     
     - 问题：这不是 KL，而是 cross-entropy 的连续版本。如果 teacher 的          
     latent 分布不是单峰 Gaussian（如 stochastic VAE），MSE 会强行做            
     mode-covering（回到 Forward KL 的坑）                                      
     - 更严重的问题： 条目中未说明"学生自己的 rollout"在连续 latent             
     空间如何定义——DiT 的去噪过程是确定性的（给定 noise                         
     schedule），没有类似 LLM 的 sampling-based rollout                         
                                                                                
     建议补充： 在 Section 6"跟咱相关的"中增加一段技术澄清：                    
     1. 明确说明"MSE-based dense supervision 不是严格的                         
     OPD"，只是借用了"on-policy + dense signal"的理念                           
     2. 指出 DiT 的确定性去噪过程不需要 on-policy 采样（因为没有 exposure       
     bias——每个 timestep 的输入是 noise schedule 决定的，不依赖之前的输出）     
     3. 给出真正可行的 OPD 场景：VQ-VAE tokenizer + autoregressive 生成（如     
     DALL-E 1/2 的 discrete codebook 路线），这时 token                         
     是离散的，可以直接套用 LLM 的 GKD/MiniLLM                                  
     4. Exposure bias 在图像生成中的严重性被高估，缺乏定量证据                  
                                                                                
     条目在 Section 3.1 中引用"错误累积 ≈                                       
     O(L²)"，但未说明这个结论是否适用于图像生成的 latent diffusion 模型。       
                                                                                
     技术事实：                                                                 
     - O(L²) 的 exposure bias 推导来自自回归生成（如 GPT），因为每个 token      
     的生成依赖之前所有 token 的 prefix                                         
     - Latent diffusion（DiT/U-Net）不是自回归的：去噪过程的每个 timestep       
     只依赖当前的 noisy latent 和 timestep                                      
     embedding，不依赖之前的去噪输出（除非是 DDIM 的 deterministic              
     sampling，但也是 Markov 的）                                               
     - 自回归图像生成（如 VQ-VAE + GPT）确实有 exposure bias，但 Stable         
     Diffusion/DiT 这类 non-autoregressive 模型理论上没有这个问题               
                                                                                
     建议补充： 在 Section 3.1 或 6 中增加一段澄清：                            
     1. Exposure bias 主要影响自回归生成（LLM、VQ-VAE + GPT），对 latent        
     diffusion 的影响有限                                                       
     2. 如果未来做离散 token 的自回归图像生成（如 Qwen 如果切换到 VQ-based      
     路线），exposure bias 才会成为实际问题，这时 OPD 才有明确价值              
     3. 当前的 DiT/U-Net 路线，OPD 的主要价值不是解决 exposure                  
     bias，而是提供比 SFT 更强的 dense supervision（类似 RL                     
     但样本效率更高）                                                           
     5. DPO 的 Reverse KL 解释缺乏推导，与 PPO 的等价性未验证                   
                                                                                
     条目在 Section 2 表格中断言"DPO 本质也是 Reverse KL                        
     视角下的优化"，但未给出推导，且这个结论在学术界有争议。                    
                                                                                
     技术事实：                                                                 
     - DPO 的 loss 是 E_{(x,y_w,y_l)~D}[ -log                                   
     σ(β·log(π_θ(y_w|x)/π_ref(y_w|x)) - β·log(π_θ(y_l|x)/π_ref(y_l|x))) ]       
     - 这可以理解为隐式地最小化 KL(π_θ || π_optimal)，其中 π_optimal ∝          
     π_ref·exp(r/β)（来自 DPO 论文的 Section 3）                                
     - 问题：DPO 的优化是在 preference pair 上的 offline 优化，不涉及           
     on-policy 采样，所以不是严格的 Reverse KL（Reverse KL 要求从 π_θ           
     采样）                                                                     
     - 更准确的说法： DPO 是隐式地优化 Reverse KL，但训练过程是 off-policy      
     的（类似 Q-learning vs SARSA 的区别）                                      
                                                                                
     建议补充： 在 Section 2 表格的 DPO 那一行增加脚注，说明：                  
     1. DPO 的目标函数隐式包含 Reverse KL 项，但训练是 off-policy               
     的（不需要从 π_θ 采样）                                                    
     2. 这导致 DPO 比 PPO 更稳定（无需 on-policy rollout），但也更受限于        
     preference 数据的覆盖度                                                    
     3. 在图像生成中，GRPO（online PPO）和 DPO（offline preference              
     learning）各有适用场景：GRPO 适合有 dense                                  
     reward（CLIP/aesthetic），DPO 适合有 human ranking 数据                    
     总结                                                                       
                                                                                
     以上五条意见核心逻辑：                                                     
                                                                                
     1. Forward KL 在连续空间的失效机制和离散空间不同，需要区分                 
     2. Reverse KL 的 mode collapse 风险在 GRPO 中缺乏实证数据和调参指导        
     3. OPD 在连续 latent 空间的适配方案目前不是严格的 OPD，需要技术澄清        
     4. Exposure bias 的严重性被高估，latent diffusion 不是自回归模型           
     5. DPO 的 Reverse KL 解释需要补充 off-policy 的限定，避免混淆              
                                                                                
     这些补充可以让知识库条目在你们的图像生成场景下更加严谨和可操作。           
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184720_30b777

Session:        20260627_184720_30b777
Duration:       1m 2s
Messages:       2 (1 user, 0 tool calls)