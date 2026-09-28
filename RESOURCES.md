# 扩散模型与图像生成 Resources

## Knowledge

### 训练范式 (SFT / RL / OPD)

- [OPD Survey (2604.00626)](https://arxiv.org/abs/2604.00626) — Song & Zheng, 2026. 迄今最全面的 On-Policy Distillation 综述，Table 1 覆盖所有主流方法演变。用于：KL 框架统一视角、OPD 方法演变。

- [DiffusionOPD (arXiv:2605.15055)](https://arxiv.org/abs/2605.15055) — Li et al., 2026, ali-vilab. OPD 在扩散模型中的系统性工作，closed-form per-step KL 推导。用于：连续扩散空间的 OPD 实现。

- [SFT Memorizes, RL Generalizes (arXiv:2501.17161)](https://arxiv.org/abs/2501.17161) — Chu et al., 2025. 实验证明 SFT 记忆训练分布、RL 学习可泛化能力。用于：SFT vs RL 泛化性对比。

- [RL's Razor (arXiv:2509.04259)](https://arxiv.org/abs/2509.04259) — Shenfeld et al., 2025. 遗忘预测变量是 KL 而不是负样本，on-policy 低 KL 偏置。用于：RL 遗忘机制理解。

- [MiniLLM (arXiv:2306.08543)](https://arxiv.org/abs/2306.08543) — Gu et al., 2023. 首次将 Forward/Reverse KL 框架引入 LLM 蒸馏。用于：KL 框架基础。

- [GKD (arXiv:2306.13649)](https://arxiv.org/abs/2306.13649) — Agarwal et al., 2023. On-Policy Distillation 统一框架，on/off-policy mixture。用于：OPD 形式化定义。

- [Quagmires in SFT-RL (arXiv:2510.01624)](https://arxiv.org/abs/2510.01624) — Kang et al., 2025. SFT score 高 ≠ 好的 RL 起点，过拟合风险。用于：SFT→RL 管线陷阱。

- [When RL Fails after SFT (arXiv:2606.09932)](https://arxiv.org/abs/2606.09932) — Liu et al., 2026. SFT 过多导致 plasticity loss，RL 无法提升。用于：训练管线 failure mode。

### 蒸馏方法 (DMD / CM)

- [DMD (arXiv:2311.18828)](https://arxiv.org/abs/2311.18828) — Yin et al., 2023. 分布匹配蒸馏，双向 KL 组合，one-step 生成。用于：DMD 原理、与 OPD 的区别。

- [Flow-OPD (arXiv:2605.08063)](https://arxiv.org/abs/2605.08063) — Fang et al., 2026. Flow Matching 上的 OPD，Manifold Anchor Regularization。用于：连续流空间 OPD。

### 视觉编码器

- [CLIP (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — Radford et al., 2021. 对比学习视觉-语言对齐。用于：视觉表征对比基准。

- [DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — Oquab et al., 2023. 自监督视觉特征，dense feature 质量高。用于：结构保留 reward 设计。

## Wisdom (Communities)

- [Hugging Face Papers](https://huggingface.co/papers) — 每日最新 arxiv 精选，评论区有作者互动。用于：追踪领域动态。

- [Reddit r/MachineLearning](https://reddit.com/r/MachineLearning) — 高信噪比技术讨论，重要论文发布后有深度解读。用于：论文批判性评估。

## Gaps

- 连续扩散空间 OPD 的严格数学等价性：目前 MSE-based dense supervision 是否严格等价于 KL 形式的 OPD，文献中缺乏清晰推导
- GRPO 在图像生成中的 β 调参规律：缺乏系统性公开消融研究
