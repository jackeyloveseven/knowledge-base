# 扩散模型 GRPO 训练阶段是否开启 CFG

> 记录日期：2026-07-02
> 触发人：李承燕
> 问题：扩散模型在 GRPO 训练阶段会开启 CFG 吗？

## 一句话总结

严格说：**rollout 采样可以开 CFG，但 policy optimization / 反向训练目标里通常不要把 CFG 混合目标一起优化**。最新 Qwen-Image-2.0-RL 的结论很明确：三种策略里最稳的是 **CFG in rollout only**；rollout 用 CFG 生成高质量候选给 reward model 打分，但训练目标排除 unconditional branch，避免联合优化 cond/uncond 带来的不稳定。

更工程化地说：**采样分布按最终部署靠近，训练梯度按稳定性收敛**。

## 核心概念定义

### GRPO / diffusion RL

在扩散模型里，DDPO / FlowGRPO / GRPO-style 方法会把反向去噪过程看成多步 policy：每个 denoising step 是 action，最终图像拿 reward，然后用 policy gradient 或 group-relative advantage 更新模型。

来源：DDPO 论文把 denoising 建模成 multi-step decision-making problem；Qwen-Image-2.0-RL 明确采用 GRPO-based RL framework。

### CFG

CFG 是把 conditional prediction 和 unconditional prediction 混合：

`pred = pred_uncond + scale * (pred_cond - pred_uncond)`

在 diffusers / DDPO 代码里，`guidance_scale > 1.0` 会启用 classifier-free guidance，并把 latents / prompt embeddings 做 cond+uncond 双份 forward。

来源：DDPO PyTorch 的 `pipeline_with_logprob.py` 里 `do_classifier_free_guidance = guidance_scale > 1.0`，然后 `noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)`。

## 三种训练策略对比

| 策略 | rollout 采样 | policy optimization / 训练 loss | 结果 | 是否推荐 |
|---|---:|---:|---|---|
| 全程开 CFG | 开 | 开 | Qwen-Image-2.0-RL 报告说会严重不稳定，最后图像 collapse 成 incoherent outputs | 不推荐 |
| 全程不开 CFG | 不开 | 不开 | reward 会涨，但模型逐渐丢 stylization / world knowledge；尤其名人、风格能力退化 | 不推荐，除非目标就是 CFG-free 部署 |
| hybrid CFG | 开 | 不把 uncond branch 纳入 objective | rollout 保留预训练模型完整能力，训练目标稳定，算力也省 | 推荐默认 |

## 技术深度分析

### 为什么 rollout 阶段可以开 CFG

reward model 看到的是最终生成图。如果 rollout 不开 CFG，而 base model 推理时本来依赖 CFG 才能充分表达知识，那么 RL 采样分布会变差：图像结构、风格、名人脸、复杂 prompt 对齐都会弱，reward 信号噪声更大。

Qwen-Image-2.0-RL 的正文说：CFG-guided rollout 可以 fully leverage pre-trained model capabilities，生成 structurally coherent images，从而给 reward evaluation 更可靠的候选。

### 为什么训练 loss 里不建议联合优化 CFG

训练目标里开 CFG 等于把 cond branch 和 uncond branch 的混合结果当 policy 来优化。问题有三个：

1. **优化对象变复杂**：policy logprob / velocity matching 不再是单个 conditional prediction，而是 cond/uncond 的线性组合；unconditional branch 也会被 reward 梯度牵着走。
2. **训练不稳定**：Qwen-Image-2.0-RL 实测 CFG in both rollout and training 会 severe training instability，最终生成图 collapse。
3. **计算开销翻倍**：CFG 需要 cond + uncond 双 forward；RL 本来就要多 sample、多 reward、多 step，训练阶段再对 CFG 双分支做反向更贵。

所以 hybrid 做法是：rollout 阶段开 CFG 拿好样本和 reward；更新时 exclude unconditional branch from policy optimization objective，用更稳定的训练目标更新 conditional 能力。

### 一个容易写错的数学点：rollout 分布和 ratio 分布

要注意：**“不优化 uncond branch”不等于“完全无视 CFG 采样分布”**。

如果 rollout 样本来自 CFG-guided 分布，但训练 ratio / logprob 直接按 raw conditional 分布算，会有 behavior policy 和 training policy 分布不匹配的问题。严格做法需要确保 importance ratio / likelihood estimator 和实际 rollout 采样机制一致；只是梯度不回传到 unconditional branch，或把 unconditional branch freeze / detach。

这也是 diffusion GRPO 比 LLM GRPO 麻烦的点：LLM 的 token policy 是显式概率，diffusion 的 reverse transition likelihood / solver / CFG 都会影响 policy density。

### CFG scale 不能无脑拉满

rollout 虽然推荐开 CFG，但 scale 不宜过高。GRPO 依赖同 prompt group 内多个样本的 reward 差异来算 advantage；CFG 太高会压缩多样性，让 group 内样本过于相似，advantage 变弱或噪声变大。

实践上更建议：训练 rollout 的 CFG scale 用偏保守区间，例如比线上推理略低；再用固定 eval set 分别看 reward、diversity、prompt alignment 和 style/world knowledge 是否同时稳定。

### 和 DDPO / DPOK / FlowGRPO / DiffusionNFT 的关系

DDPO / DPOK 是 diffusion RL 的早期基线，重点在把扩散去噪变成 policy gradient。DDPO 系代码从工程上支持 `guidance_scale > 1.0` 的 guided sampling/log_prob，但它不等价于证明 GRPO 下“全程 CFG 优化”安全。

FlowGRPO / DanceGRPO 代表了更近的 reverse-process GRPO-style 路线。DiffusionNFT 则是另一条路线：它认为 reverse-process GRPO 存在 solver 限制、forward-reverse inconsistency、CFG integration complicated 等问题，所以改用 forward process / flow matching，目标是 CFG-free。

因此：

- **如果继续走 reverse-process GRPO**：Qwen-Image-2.0-RL 的 hybrid CFG 是目前最直接的工程证据。
- **如果目标是彻底 CFG-free 部署**：DiffusionNFT 这类 forward-process RL/FT 范式更值得单独评估。

## 实践建议

### 默认配置

如果训 Qwen Image / FireRed / SD3.5 / Flux 类扩散模型的 GRPO：

- rollout / generation：按最终部署习惯开 CFG，但 scale 先别太激进；
- reward scoring：用 CFG 生成的图打分；
- policy loss：不要联合优化 cond+uncond CFG 混合目标；unconditional branch 建议 freeze / detach / 排除出 objective；
- ratio / likelihood estimator：保证和实际 rollout 采样机制一致，避免 rollout 用 guided prediction，loss 却按另一个分布算；
- eval：同时看 reward、diversity、prompt adherence、style/world knowledge，不要只看 reward 均值上涨。

### 什么时候可以全程不开 CFG

只有一种场景比较合理：你明确要训练一个 **CFG-free 部署模型**，并且愿意接受 early stage 的 reward 噪声和知识能力退化风险。这类路线可以参考 DiffusionNFT 的 CFG-free 目标，但它不是标准 reverse-trajectory GRPO，而是 forward-process / flow matching 的替代范式。

### 踩坑点

- 不要简单照搬 diffusers pipeline 的 `guidance_scale=7.5` 到训练 loss 里。
- rollout 的 CFG scale 最好和最终推理 scale 有意识地对齐或略低，否则训出来的偏好分布和部署分布不一致。
- 如果训练时发现 reward 涨但出图越来越“没知识/没风格”，先检查是不是 rollout 也关了 CFG。
- 如果图像突然 collapse，先检查是不是把 CFG 同时放进 rollout 和 training objective 了。
- DDPO 代码支持 CFG 只是“能做”，不是“GRPO 里应该全程开 CFG”的证据。

## 六源搜索记录

| 来源 | 状态 | 结果 |
|---|---|---|
| 已有知识库 | 已搜索 | 命中 `05_训练数据分布控制与训推Prompt一致性.md` 的 CFG null-condition / prompt 一致性内容，命中 `01_SFT_RL_OPD在DiT图像生成中的区别.md` 的 RL/GRPO 背景 |
| zh-cache | 已搜索 | 未命中 GRPO+CFG 专门条目；仅有 diffusers、DiffSynth、Qwen/FireRed 项目相关入口 |
| arXiv | 已搜索 | 命中 Qwen-Image-2.0-RL、DiffusionNFT、DDPO、DPOK、dFlowGRPO、RL design space 等 |
| HuggingFace | 已搜索 | `diffusion GRPO CFG classifier free guidance` 无模型结果 |
| GitHub | 已搜索 | repo 级命中 DDPO PyTorch / DDPO / DPOK；GitHub code search 需认证，代码搜索 401 |
| Google 中文 | 已搜索 | r.jina.ai 访问 Google 返回 CAPTCHA |
| 知乎 | 已搜索 | 返回 403 / 安全页，无有效技术内容 |
| 小红书 | 已搜索 | 只返回页脚备案，无有效技术内容 |

## 具体度自检

| 检查项 | 状态 |
|---|---|
| 论文编号 / 链接 | 有：arXiv:2606.27608、2509.16117、2305.13301、2305.16381、2602.04663、2605.09291 |
| 代码仓库 / 实现 | 有：kvablack/ddpo-pytorch、jannerm/ddpo、google-research/google-research/dpok |
| 具体数字 | 有：Qwen-Image-2.0-RL 报告 57.84 overall score，Elo 1193/1349；DiffusionNFT 报告 25× efficiency、GenEval 0.24→0.98；RL design space 报告 GenEval 0.24→0.95、90 GPU hours、4.6× efficient than FlowGRPO |

## 多模型交叉审核（2026-07-02）

### 来自 GLM-5.2

1. Hybrid CFG 的核心风险是 rollout 从 CFG-guided 分布采样，但训练 ratio 如果按 raw conditional logprob 算，会出现分布不匹配。应写清楚：不优化 uncond branch 不等于不建模 CFG 采样分布。
2. 冻结 unconditional branch 后，conditional branch 持续更新仍会让有效引导方向 `(pred_cond - pred_uncond)` 漂移。需要监控 CFG 引导方向稳定性，并避免 rollout scale 过高。
3. DDPO 是 PPO-based，GRPO 是 group-relative advantage；DDPO pipeline 支持 CFG 只能做背景，不能直接证明 GRPO 下的安全性。

### 来自 Gemini 3.1 Pro

1. Policy optimization 不开 CFG 的工程收益不只是稳定，也省掉 unconditional backward 的显存和算力。
2. Rollout CFG scale 不能太高，否则同 prompt group 内样本多样性被压缩，advantage 估计会变差。
3. GRPO 省掉 critic/value network 的显存占用，这能部分对冲 rollout 开 CFG 的额外 forward 成本。

### 来自 Claude Opus 4.8

1. 需要区分论文原文和工程解读。Qwen-Image-2.0-RL 明说 hybrid CFG 和三策略对比；“rollout 开 / training objective 不优化 uncond branch”来自正文描述和工程解释，不应混成摘要原话。
2. DiffusionNFT 不是 hybrid CFG 的旁证，而是另一种 CFG-free forward-process 路线；它的价值是指出 reverse-process GRPO 与 CFG 集成复杂。
3. landscape 里应补 FlowGRPO / DanceGRPO 这些 reverse-process GRPO 基线；DDPO/DPOK 更适合作为早期背景。

## 参考资料

- Qwen-Image-2.0-RL Technical Report, arXiv:2606.27608, https://arxiv.org/abs/2606.27608
- DiffusionNFT: Online Diffusion Reinforcement with Forward Process, arXiv:2509.16117, https://arxiv.org/abs/2509.16117
- Training Diffusion Models with Reinforcement Learning / DDPO, arXiv:2305.13301, https://arxiv.org/abs/2305.13301
- DPOK: Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models, arXiv:2305.16381, https://arxiv.org/abs/2305.16381
- Rethinking the Design Space of Reinforcement Learning for Diffusion Models, arXiv:2602.04663, https://arxiv.org/abs/2602.04663
- dFlowGRPO: Rate-Aware Policy Optimization for Discrete Flow Models, arXiv:2605.09291, https://arxiv.org/abs/2605.09291
- DDPO PyTorch code, https://github.com/kvablack/ddpo-pytorch
- Original DDPO code, https://github.com/jannerm/ddpo
- Google Research DPOK code, https://github.com/google-research/google-research/tree/master/dpok
