# DiffusionNFT 调参经验：从论文参数到工程排障

> 记录日期：2026-07-21 / 触发人：李承燕  
> 主来源：用户提供的可视化页面《DiffusionNFT 超参数与易错点速查》；交叉来源：DiffusionNFT 论文 arXiv:2509.16117、NVlabs/DiffusionNFT 官方仓库、HuggingFace 模型/数据集、已有知识库条目。

## 一句话总结

DiffusionNFT 的调参优先级不是“先加 LR / rank / group size”，而是先保证 forward-process objective 的权重映射、virtual group 统计、old-policy EMA、CFG/分辨率/steps 评测合同完全一致；当前最关键的坑是 `adv_clip_max=5.0` 可能把 reward 排序信号稀释掉，论文 Algorithm 1 等价值更接近 `1.0`。

## 核心背景

DiffusionNFT（Diffusion Negative-aware FineTuning）是把扩散模型在线 RL 从 reverse trajectory likelihood / GRPO ratio 里绕出来，改成在 forward process 上做 flow matching。论文声称它不依赖采样轨迹 likelihood，能兼容黑盒 solver，且 CFG-free；arXiv 摘要给出的关键数字是：相对 FlowGRPO 最多 25× efficiency，GenEval 从 0.24 到 0.98 约 1k steps，而 FlowGRPO 到 0.95 需要 5k+ steps 并额外使用 CFG。

这次记录的页面不是论文复述，而是一个工程项目基于 DiffusionNFT 训练 SenseNova-U1 / text rendering / OCR 场景后的超参踩坑速查。它最有价值的点在于：把“哪些旋钮真正影响训练信号”和“哪些只是评测 recipe 错误”拆开了。

## 核心超参速查

| 参数 / 概念 | 页面记录的论文或代码口径 | 项目实测/建议 | 风险级别 | 关键解释 |
|---|---|---|---|---|
| `--adv-clip-max` | 论文 Algorithm 1 等价值约 `1.0`，clip 边界硬编码 ±1 | 扫描 `1.0 / 1.5-2.0 / 5.0`；页面认为 `5.0` 过松 | 高 | reward z-score 被映射成 optimality。`c` 太大时，80-85% 样本挤在 `[0.45,0.55]`，正负分支几乎 50/50，reward 排序信号被无信息项稀释。 |
| `group-size` vs `virtual-group-size` | 真正做 advantage 归一化的是 virtual group | physical group 用过 2/8，但 virtual 恒为 8 | 高 | 不要误以为 physical group=2 导致统计噪声；如果代码按 virtual group 归一化，group2/group8 在统计功效上可能等价。 |
| `beta` / guidance strength | 论文 β ∈ `{0.1, 1.0}`；官方 config `sd3_geneval=1.0`、`sd3_ocr/multi_reward=0.1` | `1.0` 稳但慢，`0.1` 快但风险高 | 中高 | 控制 implicit positive/negative velocity 混合强度，并影响 loss 缩放；不是随便扫的温度参数。 |
| `learning-rate` | 常规 `3e-4`；页面事故中用过 `3e-3` | LR 必须和 old EMA decay 一起看 | 高 | LR×10 但 EMA 不调，会让 policy-old 距离暴涨，页面记录约占参数总幅度 60-75%，疑似轻量 collapse。 |
| LoRA rank / alpha / 层范围 | 论文 SD3.5-M 使用 `r=32, alpha=64` | 页面项目从 rank4→8→64；仅训 layers 30-38 | 高 | rank 太低时权重确实在动，但目标 reward/OCR 分数不动；这不是没训练，而是容量覆盖不到失败模式。 |
| CFG scale / CFG norm | 论文目标是 CFG-free；官方 config 训练采样 `guidance_scale=1.0`，eval 示例可用 4.5 | rollout / ratio / KL / reference 必须同一套 CFG 定义 | 高 | 早期 CFG-free 评测天然可能差，论文约 1000-1700 iterations 后才反超；几十个 update 不能直接判定训练无效。 |
| `noise_scale σ(H,W)` | 页面给出 `σ=sqrt(N(H,W)/N0), N0=64` | 256px:1 / 512px:2 / 1024px:4 / 2048px:8 | 中 | 分辨率放大时噪声强度要跟 latent token 数缩放，不是凭感觉调。 |
| 分辨率 / steps / sampler | smoke 与正式评测分离 | smoke: 512px/10-step；正式：2048px/50-step/CFG4/timestep_shift=3.0 | 高 | 低分辨率少步数只能证明链路通，不代表质量；页面事故中 1024/10/CFG1 导致两边都糊。 |
| old-policy EMA decay | 页面记录 sample-normalized `0.9994/sample`；virtual=8 时 logical decay≈`0.995` | 按 logical update 换算，不看名义值 | 高 | decay 太快时 old policy 跟随过快；判断更新强弱要看 policy-old RMS/MSE。 |
| `kl_beta` / `weight_decay` | 页面三个对照 run 均为 0 | 先排除正则导致“权重没动”假象 | 中 | `kl_beta>0` 必须有 reference_velocity；weight_decay 会把 LoRA 权重往 0 拉。 |
| rollout group size | 论文 SD3.5-M: `G=24`；官方 config `num_image_per_prompt=24` | 增大 group 成本近似线性增加 | 中 | 只有 reward std 长期接近 0，才优先考虑增大 group；不要默认用它救所有问题。 |
| timestep 数 / transitions per update | 论文/页面默认约 9/10 timesteps；历史误配 3/10 | loss 必须按 timestep 数归一化，只做一次 optimizer.step | 高 | 多 timestep / chunk / microbatch 只能梯度累积，不能隐式放大学习率。 |

## 最关键的代数坑：为什么 `adv_clip_max` 太大会“有梯度但没信号”

页面给的推导很重要：设 optimality = `0.5 + z_clamp/(2c)`，`c = adv_clip_max`，A/B 分别是正/负分支距离。

`policy_per_sample = optimality*A + (1-optimality)*B = 0.5*(A+B) + z_clamp/(2c)*(A-B)`。

如果代码最后又把 loss 乘回 `c`，就变成：

`policy_loss = c*0.5*mean(A+B) + 0.5*mean(z_clamp*(A-B))`。

第一项是 reward 无关项，随 `c` 线性放大；第二项才是携带 reward 排序的判别项，基本不随 `c` 增强。于是 `c=5` 时，grad norm 仍然很健康，但主要来自 reward 无关的双目标拉扯，训练看起来“在动”，分数却不涨。

工程判断：这个坑比 LR/rank 更优先排查。因为它能解释“LoRA RMS 单调移动、grad norm 非零、但 reward/OCR 不变”的组合症状。

## 12 条硬约束（页面 §0 的工程合同）

1. 一个 logical step 只能 `optimizer.step()` 一次。  
2. timestep/chunk/microbatch/多 backward 只能梯度累积，不能各自 update。  
3. 一个旧 rollout 默认只对应一次 policy update。  
4. old/rollout policy 优化期间冻结，完成后只做一次慢 EMA。  
5. 禁止每个 microbatch/timestep hard-copy 或 EMA 更新 old policy。  
6. rollout、old log-prob、policy recompute、reference 必须使用一致 CFG 和采样定义。  
7. 同一个 virtual group 内必须同一个 prompt、同一份 reward metadata。  
8. 低分辨率/低步数/CFG-free smoke 只证明链路能跑，不证明质量变好。  
9. Base/RL 正式对比必须固定 prompt、seed、sampler、steps、resolution、CFG、权重来源。  
10. 正式评测禁止 best-of-N，每张直接采样图都要计分。  
11. checkpoint 必须含 policy、old、optimizer、evalEMA、进度、配置，且读回验证。  
12. 官方 benchmark test prompts 只读，不能进训练或 reward overfit 数据。

## 故障定位顺序

当“图像变化很小 / reward 不涨”时，排查顺序应是：

1. 确认 `optimizer_steps == 1`。  
2. 确认 reward std 和 advantage 非零。  
3. 确认 grad norm 非零且有限。  
4. 测 policy-old、policy-evalEMA 的参数 RMS/MSE 位移。  
5. 查 prompt exposure、round-robin、每 update prompt 数。  
6. 查 train/eval 的 CFG、steps、sampler、resolution 是否一致。  
7. 查 reward 是否真的测目标能力，而不是只测总体审美。  
8. 如果 strict 指标不涨，看 partial 指标，避免离散阈值误判。  
9. 做 multi-seed held-out，排除 fixed-seed overfit。  
10. 最后才调 LR、LoRA 范围、virtual group、noise、reward 权重。

这个顺序的本质：先证明训练信号存在且 contract 没坏，再调容量和优化强度。否则会把一个 objective scaling bug 误诊成“rank 不够”或“group 太小”。

## 页面里仍待确认的矛盾

页面明确标了一个未解矛盾：GUARDRAILS.md §7.3 曾说 GenEval track 正确兼容配置应使用 normalized advantage clip ±5；但 DiffusionNFT 论文 Algorithm 1 里的 clip 是 ±1，代入当前 `reward_to_optimality()` 实现，`adv_clip_max=1.0` 才逐项对齐论文。

我的判断：在重新核对当年“官方代码审计”的具体 commit 和 normalization convention 前，不要把 ±5 当成定论。更稳的实验设计是三点扫描：`1.0`（论文等价）、`1.5-2.0`（保守放宽）、`5.0`（历史默认），并记录 optimality 分布直方图、reward std、policy-old RMS、target reward 曲线。

## 和已有知识库的关系

- 与 `13_扩散GRPO训练CFG策略.md` 互补：该条讨论 diffusion GRPO / CFG 的训练分布一致性；本条补 DiffusionNFT forward-process 路线下的 CFG-free 与评测合同。  
- 与 `14_编辑模型Reward稀缺与RationalRewards.md` 互补：RationalRewards 的 RL 实验使用 DiffusionNFT 风格训练；本条给具体训练超参和排障顺序。  
- 与 `11_Adam_AdamW_Muon_WeightDecay_vs_L2.md` 相关：weight decay 在 LoRA RL 中会影响“权重是否真的在按 reward 方向移动”的判断。

## 六源搜索记录

| 来源 | 状态 | 结果 |
|---|---|---|
| 用户给定页面 | 已读取 | 提取了完整正文：核心超参、adv_clip 推导、12 条约束、十步排查、事故时间线、CFG 语义。 |
| arXiv | 已搜索 | arXiv:2509.16117 v2，ICLR 2026 Oral；摘要含 25× efficiency、GenEval 0.24→0.98、CFG-free 等信息。 |
| GitHub | 已搜索 | 官方仓库 `NVlabs/DiffusionNFT`，README 与 `config/nft.py`；确认 SD3.5-M、`num_image_per_prompt=24`、`guidance_scale=1.0`、`beta=1.0/0.1`、torchrun 训练入口。GitHub code search API 未认证，返回 401。 |
| HuggingFace | 已搜索 | 模型 `worstcoder/SD3.5M-DiffusionNFT-MultiReward`、`yeonwoo378/sd3.5-diffusionNFT-reproduced`；数据集 `TIGER-Lab/RationalRewards_DiffusionNFT_TrainData`。 |
| zh-cache | 已搜索 | 命中本地中文索引《DiffusionNFT 推导整理：公式（7）（8）（9）怎么连起来》，但知识库正文未找到该条 md。 |
| Google 中文 | 已搜索 | r.jina 代理 Google 只返回重定向提示，无有效结果。 |
| 知乎 | 已搜索 | 返回安全验证 / CAPTCHA，无有效技术内容。 |
| 小红书 | 已搜索 | 请求返回空内容，无有效技术内容。 |
| 已有知识库 | 已搜索 | 命中条目 13 和 14，已作为交叉引用。 |

## 具体度自检

| 检查项 | 状态 |
|---|---|
| 论文编号 / 链接 | 有：DiffusionNFT arXiv:2509.16117，RationalRewards arXiv:2604.11626 作为关联。 |
| 代码仓库 / 实现 | 有：NVlabs/DiffusionNFT；官方 `config/nft.py` 参数已核对。 |
| 具体数字 | 有：25× efficiency、GenEval 0.24→0.98、FlowGRPO 0.95 with 5k+ steps、G=24、β=0.1/1.0、`guidance_scale=1.0/4.5`、resolution/steps 配方等。 |

## 多模型交叉审核（2026-07-21）

按 knowledge-record 流程发起 GLM-5.2 / Gemini 3.1 Pro / Claude Opus 4.8 三路短 prompt 审核；本轮 `hermes chat` 进程退出但 stdout 文件为空，未产出可采纳审核意见。未编造 reviewer 结论，后续如需可重新跑审核。

## 参考资料

- 用户给定页面：DiffusionNFT 超参数与易错点速查 — https://yata-image-publics.flowgpt.com/felix/diffusionnft_hparams.html
- DiffusionNFT: Online Diffusion Reinforcement with Forward Process — https://arxiv.org/abs/2509.16117
- NVlabs/DiffusionNFT 官方仓库 — https://github.com/NVlabs/DiffusionNFT
- SD3.5M-DiffusionNFT-MultiReward — https://huggingface.co/worstcoder/SD3.5M-DiffusionNFT-MultiReward
- RationalRewards_DiffusionNFT_TrainData — https://huggingface.co/datasets/TIGER-Lab/RationalRewards_DiffusionNFT_TrainData
- 已有知识库：`13_扩散GRPO训练CFG策略.md`、`14_编辑模型Reward稀缺与RationalRewards.md`
