# 图像文本条件 Context Scaling：信息量而非 Prompt 长度

> 记录日期：2026-08-03
> 触发人：李承燕
> 论文：Scaling Properties of Text Conditioning in Visual Generation（arXiv:2607.29679，ByteDance Seed）

## 一句话结论

图像生成的第四条 scaling axis 不是 prompt token 数，而是 caption 对图像真实内容的可绑定信息量；产品上应把 Prompt Enhancer 从“扩写文案”升级为“用户意图 → 可执行视觉规格”的编译器。

## 核心证据

同一个 Qwen-Image 与相同 seed 下，自然语言描述从 470 增至 2,130 tokens，DINOv3 仅在 0.46–0.51 波动、LPIPS 在 0.68–0.69，重建没有变好；结构化 Prompt 从 L5 增长到 L10 时，DINOv3 从 0.41 提升到 0.65，LPIPS 从 0.72 降至 0.59。

论文用两个互补指标测 caption 信息量：

- **GPG（Grounded Perplexity Gain）**：白盒指标，计算冻结 VLM 看到配对图像后，caption 内容 token 的 log-likelihood 相对无图条件提升了多少。
- **ED（Effective Detailness）**：黑盒指标，把图像与 caption 都拆成实体/属性/关系事实，采用 F0.5 衡量 precision/recall，更重惩罚无中生有。工程含义是：训练 caption 宁可少写，也不要写错，错误属性会形成脏的文本—像素绑定。

在 15 组 caption 配置、相同图片/架构/初始化、共同 2.84×10^10 image-token 预算的 BAGEL 实验中：

- MSE = 0.4549 − 8.45×10^-5 × GPG，Pearson r = −0.984；
- MSE = 0.4200 × ED^-0.2073，log-log Pearson r = −0.971；
- 两种指标对配置排序的 Spearman ρ = 0.96；
- 在未参与拟合的空间与字段变体上，MSE 预测 MAE 分别为 5.0×10^-4 和 8.1×10^-4。

这是一条 **recipe-specific empirical calibration**，不是跨模型普适定律。它共同依赖 backbone、数据和训练 recipe；每个配置只训练一次，不能把图中带状范围理解为经典多次重复置信区间。

## Diffusability × Promptability

论文把端到端质量概括为：

Quality(f, π) = Diffusability(f) × Promptability(f, π)

这里是概念分解，不是经过拟合的乘法定律。

### Diffusability：让 DiT 真正读懂训练条件

结构化 Prompt 是 typed JSON，主要字段包括 intent、scene、elements、bbox、depth、materials、pose/action、relationships、photography。训练图像通过五段式标注管线产生 L10：

1. VLM 提取全局意图、场景、光线、摄影与元素清单；
2. 对元素 crop 再识别局部属性与动作；人类元素结合 Sapiens 133 keypoints；
3. DepthAnything V2 提供相对深度，SAM 2.1 提供 mask 与遮挡；
4. VLM 汇总语义和几何证据，生成完整 L10 JSON；
5. 按字段组 mask，生成 L5–L9 的控制梯度。

L5→L10：平均 token 447→1,374，GenEval2 GM 46.79→57.70，GSB 相对 L5 0→26%。字段消融显示 global scene context 的贡献最大，其次是 bounding box。

### Promptability：让 LLM 把用户意图填进视觉规格

固定 schema 和 Qwen-Image diffuser，只更换零样本 Qwen3.5 prompter：thinking 模式下，GenEval++ 从 0.8B 的 46.4% 提升到 397B 的 86.8%。但 397B 零样本依然会生成合法却不够详细的结构化 Prompt，因此作者继续做：

- **SFT**：学习 DiT 所需要的 SP 内容分布；structure 4.860→6.273，是最大单阶段提升。
- **Cold-start**：利用带参考图的 VLM 教师生成“如何从无图用户请求推导 SP”的 reasoning trace，并过滤泄漏图像实例信息的 trace。
- **RFT / verifier-gated OPSD**：固定 DiT；student 对原始 prompt 做 on-policy rollout，渲染后由 verifier 高精度筛选；接受的轨迹上，同一 Qwen3.5-397B-A17B base 的 image-conditioned teacher 提供 token-level KL 目标，只有 rank-128 LoRA student 更新。实现上使用 teacher top-64 logits，单 token divergence clip 5.0。

最终单轮 prompter：DPG 90.71、structure 7.600、alignment 9.047、GSB 42%。

## 端到端结果与关键 matched control

同为 Qwen-Image：

| 接口 | GenEval2 GM | CoReBench |
|---|---:|---:|
| 官方 Prompt Enhancer | 52.8 | 74.7 |
| Matched NL：同数据、同阶段、同预算重训 | 56.2 | 76.1 |
| Structured Prompt 系统 | 72.5 | 85.2 |

因此增益不能简单归因为“多训了一遍”。最终系统还达到 GenEval 0.94、DPG 90.71、WISE 0.89。

## Agentic inference：短闭环有效，长闭环迅速饱和

在线 loop：用户 prompt → prompter 生成 SP → 固定 DiT 渲染 → Gemini judge 只看原请求与图像，返回 structure/alignment/aesthetic、PASS/FAIL 与字段级错误。失败后只修相关字段，直到 PASS 或 Tmax。最终结果由独立 GPT-5.4 离线评测。

| Trained prompter | Structure | Alignment | GSB | 平均轮数 |
|---|---:|---:|---:|---:|
| Tmax=1 | 7.600 | 9.047 | 42.0% | 1.00 |
| Tmax=2 | 7.940 | 9.173 | 49.3% | 1.51 |
| Tmax=4 | 8.213 | 9.293 | 54.0% | 2.04 |
| Tmax=8 | 8.260 | 9.313 | 54.7% | 2.31 |

Tmax 4→8 的增益近乎归零。工程上应使用强单轮 + 短纠错，并自行增加 best-so-far 与回滚；后者是产品建议，不是论文原有机制。

## 产品与训练判断

1. **Prompt Enhancer 应成为 Visual Compiler。** 自然语言仅是用户接口，模型内部条件应是对象、属性、关系、几何和摄影参数明确的 scene spec。
2. **训练数据优先优化绑定精度。** 不要把 caption 长度当 KPI；先用 ED/GPG 做离线 caption recipe 筛选，再决定是否启动昂贵 DiT 训练。
3. **C 端不要直接暴露 JSON。** 面向用户提供点选对象、拖动位置、局部修改、错误高亮等 WYSIWYG 交互，底层映射到 SP 字段；专业模式再开放 schema。
4. **简单请求动态 bypass。** 单主体、低约束任务可以绕过大 prompter 或走小模型；复杂多实体/空间/文字/知识任务再路由完整 SP。
5. **Agent loop 预算应有硬上限。** 默认 1–2 轮，最高 4 轮；必须记录 best-so-far，避免最后一轮退化。
6. **公开 demo 成本不等于生产成本。** 仓库 demo 使用 Qwen3.5-35B-A3B + Qwen-Image，两模型常驻两张 GPU，文档要求约 80GB 总 VRAM、首次约 110GB 下载；但论文 headline prompter 是 Qwen3.5-397B-A17B 上的 rank-128 LoRA。两者必须区分，生产需蒸馏和动态路由。

## 边界

- GPG/ED 都依赖 image-conditioned VLM 或离线事实抽取；换 judge/extractor 可能改变数值。
- schema 为手工设计，自动发现 schema、视频与 3D 扩展仍未解决。
- loss 下降不能自动等同于所有审美维度提升。
- 表 2 含作者复现和不同来源模型分数，不能当作完全统一 API/预算下的竞技场。

## 多模型交叉审核（2026-08-03）

- GLM-5.2：核实论文主系统 prompter 是 Qwen3.5-397B-A17B + rank-128 LoRA；HF 发布的 35B-A3B 是 demo 小号变体。补充 OPSD 的同 base teacher/student、top-64 logits 与 clip 5.0。
- Gemini 3.1 Pro：建议 C 端隐藏 JSON，以 WYSIWYG 操作映射结构字段；简单任务应动态 bypass 大 PE。强调 F0.5 背后的训练数据 precision 优先。
- Claude Opus 4.8：强调 demo 硬件数字不是论文生产 SLA；Tmax 4→8 饱和可以作为推理预算硬证据。

## 参考资料

- 论文：https://arxiv.org/abs/2607.29679
- 项目页：https://heheyas.github.io/context-scaling
- 代码：https://github.com/heheyas/context-scaling
- 模型集合：https://huggingface.co/collections/heheyas/context-scaling
- Demo：https://heheyas-context-scaling.hf.space/
