> 记录日期：2026-07-03 / 触发人：李承燕
> 问题：为什么现在 edit 模型的 reward 如此稀少，Flow-GRPO 也大多做的是 T2I 任务？细读 RationalRewards: Reasoning Rewards Scale Visual Generation Both Training and Test Time（arXiv:2604.11626）和 TIGER-Lab/RationalRewards-SFTData。

# 编辑模型 Reward 稀缺与 RationalRewards：为什么 Edit RL 比 T2I 更难做

## 一句话总结

Edit 模型 reward 稀少不是大家没想到，而是 image editing 的 reward 天生比 T2I 多一个“源图保持/局部改动/身份一致性”约束，数据采集、判分、RL 在线采样和防 reward hacking 都更贵；Flow-GRPO 主要先做 T2I，是因为 T2I reward 能用 prompt-image 单输入和现成 aesthetic / CLIP / ImageReward / PickScore / GenEval 组合起来跑，而 edit reward 必须同时看 source image、instruction、edited image，并把 instruction following、image faithfulness、visual quality、text rendering 拆成多维信号。RationalRewards 这篇的价值就在于把稀缺的 pairwise preference 数据通过 PARROT 转成带 reasoning 的多维 reward，用 30K edit pair 做出可用于 edit RL 和 test-time prompt tuning 的 8B reward model。

## 0. 本次检索状态

| 来源 | 状态 | 关键结果 |
|---|---|---|
| 已有知识库 | 已搜 | 命中 GRPO / RL / image edit 相关条目，如 01、03、10、12、13，但没有专门记录 edit reward 稀缺问题 |
| arXiv | 已读 | arXiv:2604.11626，RationalRewards: Reasoning Rewards Scale Visual Generation Both Training and Test Time |
| HuggingFace | 已读 | TIGER-Lab/RationalRewards-SFTData、RationalRewards-8B-T2I、RationalRewards-8B-Edit、DiffusionNFT_TrainData、EvalData |
| GitHub | 已读 | TIGER-AI-Lab/RationalRewards，包含 rationalrewards_sft、reward_model_evaluation、diffusion_rl_training、test_time_prompt_tuning |
| Google 中文 | 已尝试 | r.jina 代理在当前环境返回 HTTP 451，未拿到有效中文结果 |
| 知乎 | 已尝试 | r.jina 代理返回 HTTP 451，未拿到有效中文结果 |
| 小红书 | 已尝试 | r.jina 代理返回 HTTP 451，未拿到有效中文结果 |
| zh-cache | 已读 | 本地缓存命中 FireRed / Qwen Image Edit / DiffSynth / 训练数据 pipeline 等泛相关条目，无 RationalRewards 专项中文解析 |

## 1. 为什么 edit reward 比 T2I reward 稀缺

### 1.1 输入结构复杂一阶：T2I 是二元组，Edit 是三元组

T2I reward 的基本输入是：

- prompt
- generated image

Edit reward 的基本输入是：

- source image
- edit instruction
- edited image

这多出来的 source image 不是简单多传一张图，而是 reward 语义从“图像是否符合文本”变成“该改的改了，不该改的别动”。RationalRewards 的 edit prompt 明确把评估轴拆成四个：

| 维度 | T2I 是否需要 | Edit 是否需要 | 作用 |
|---|---:|---:|---|
| Text Faithfulness | 需要 | 需要 | 是否执行了文本指令 |
| Physical / Visual Quality | 需要 | 需要 | 是否有视觉瑕疵、几何/光照/人体错误 |
| Text Rendering | 条件需要 | 条件需要 | 涉及文字生成时是否可读、拼写正确 |
| Image Faithfulness | 不需要 | 强需要 | 非编辑区域、背景、身份、风格、光照是否保留 |

Edit reward 的难点几乎都卡在 Image Faithfulness：如果 reward 只看最终图，它会偏好“整体好看但把源图重画了”的结果；如果过度强调保持，又会惩罚必要的语义改动。这个 trade-off 很难用单个 scalar reward 表达。

注意术语口径：论文正式 rubric 里 edit-only 新增维度是 **Image Faithfulness**；质量维度在不同位置写作 Physical Quality 或 Physical / Visual Quality，实际含义是技术质量、物理 plausibility、artifact、构图一致性。本文后文统一把核心四维理解为：Text Faithfulness、Image Faithfulness（editing only）、Physical / Visual Quality、Text Rendering。

### 1.2 数据采集贵：T2I 可以从海量生成偏好里挖，Edit 需要 source/prompt/edit 三件套

T2I preference 数据常见来源是：prompt + 多张生成图 + 用户偏好/打分。很多 AIGC 平台天然会沉淀这种数据。

Edit preference 数据要满足更强条件：

1. 源图必须可用。
2. edit instruction 必须清楚。
3. 至少两个 edit 输出可比较，或者一个输出有绝对分数。
4. 标注者必须同时判断“改动正确性”和“源图保持”。
5. 数据许可更麻烦：source image 可能来自用户上传，版权/隐私/人脸风险更高。

RationalRewards 论文里也能看到这个不平衡：训练数据来自 EditReward 的 image editing 30K pair，T2I 则来自 HPDv3 + RapidData 共 50K pair。公开 SFT 数据文件名也对应这个结构：editreward_pairwisev4_42k_updated.jsonl、editpica_pairwisev4_33k_updated.jsonl、editreward_pointwise_filtered_updated.jsonl、pica_pointwise_filtered_updated.jsonl；T2I 有 t2i_pairwisev4all_85k_updated.jsonl、t2i_pointwise_updated.jsonl。

### 1.3 Reward 设计贵：edit 的“好”是条件化且局部的

T2I 的 reward 可以粗暴组合：美学分、CLIPScore、ImageReward、PickScore、OCR/GenEval 等。虽然会 hack，但至少能跑。

Edit 不能这么搞。一个 edit 输出高质量，必须同时满足：

- 指令执行：比如“把 BAR 换成 Beach”确实换了。
- 局部性：只改牌子文字，不要把店面、字体风格、光照、视角全重画。
- 源图一致：人物身份、产品外观、背景布局不能乱。
- 物理一致：阴影、反射、遮挡、材质要跟源图一致。
- 任务特定：remove/add/replace/style/compose/extract 的判分逻辑不同。

这导致一个单 scalar reward 很容易错：它可能奖励“更漂亮的新图”，但这张图作为 edit 是失败的。

### 1.4 在线 RL 成本贵：每一步 reward 都要多图 VLM judge

Flow-GRPO / DiffusionNFT 这种在线 RL 要对同一个 prompt 采样一组图，再给每张图打 reward。T2I 时 reward server 看 prompt + generated image；edit 时 reward server 要看 source + edited image + instruction。

RationalRewards 的 diffusion_rl_training 代码里 edit reward server 的调用就是 ref_images + images + prompts + metadatas，并通过 `parse_scores_from_detailed_judgement` 解析四个维度分数，最后把可用分数平均，再归一化：

- overall_score = mean(valid dimension scores)
- normalized_score = (overall_score - 1) / 3

这比 T2I reward 重：输入图更多、上下文更长、VLM 推理更慢、解析更脆、batching 更复杂。

### 1.5 Reward hacking 更隐蔽：edit 模型可以通过“重画一张更讨喜的图”骗分

论文 Figure 3 / Figure 11 / Figure 12 都强调 scalar reward 会出现 reward 增长但视觉质量下降。对 edit 来说还有一种更坏的 hack：模型学会生成 judge 喜欢的高质量通用图，而不是忠实编辑源图。

所以 edit reward 必须显式约束：

- source fidelity
- non-edited region preservation
- instruction-localized change
- identity / background / style preservation

RationalRewards 通过 critique-before-score 强迫 reward model 先解释每个维度为什么给分，再输出分数，本质上是给 reward 加了一个结构化的 regularizer，降低无理由分数膨胀。

## 2. 为什么 Flow-GRPO 大多先做 T2I

### 2.1 T2I 的 reward 函数更容易“先跑起来”

Flow-GRPO 这类方法的核心不是 reward 本身，而是把 flow matching / diffusion 采样过程接到 online RL。为了验证算法，最自然选择是 reward 最容易闭环的 T2I：

- 输入简单：prompt + image。
- 现成 reward 多：PickScore、ImageReward、HPS、CLIP、aesthetic、OCR、GenEval。
- benchmark 多：GenEval、T2I-CompBench、UniGen、DPG 等。
- 数据易构造：只要 prompt set，不需要 source image。
- 不涉及源图版权和身份保持。

所以早期 Flow-GRPO / DanceGRPO / DiffusionNFT 很多实验先打 T2I，是工程路径最短。

需要区分行业 framing 和本论文实现：用户问题里的 Flow-GRPO 代表“flow/diffusion online RL 多先做 T2I”的行业现状；RationalRewards 论文自己的 RL 实验不是 Flow-GRPO，而是基于 DiffusionNFT / Edit-R1 风格的 group sampling + weighted diffusion loss。Flow-GRPO / DanceGRPO 在论文中主要作为 related work 被引用。

### 2.2 Edit RL 要同时解决环境、数据、reward 三件事

把 Flow-GRPO 迁移到 edit，不只是换 pipeline：

| 模块 | T2I | Edit |
|---|---|---|
| policy 输入 | text prompt | source image + instruction |
| sample group | 同一 prompt 采 K 张 | 同一 source/instruction 采 K 张 |
| reward 输入 | prompt + generated | source + instruction + edited |
| reward 轴 | 文本一致、质量、文字 | 文本一致、源图保持、质量、文字 |
| 数据 | prompt list 即可 | source image + edit instruction dataset |
| 评估 | T2I benchmark | ImgEdit、GEdit、PICA 等 |
| 常见 hack | 美学投机、OCR 投机 | 重画源图、身份漂移、非编辑区污染 |

RationalRewards 的开源实现基本就是把这些补齐：基于 DiffusionNFT / Edit-R1 环境，支持 Flux Kontext 和 Qwen Image Edit，reward server 提供 edit 多维分数，训练脚本分别是 `train_flux_kontext_rationalrewards.py` 和 `train_qwen_edit_rationalrewards.py`。

### 2.3 Edit benchmark 也更碎

T2I benchmark 通常按组合、属性、布局、文字、关系等拆；edit benchmark 还要按 edit type 拆：Add、Adjust、Extract、Replace、Remove、Background、Style、Compose、Action，以及物理编辑的 LightProp、LightSrcEff、Reflection、Refraction、Deformation、Causality、StateTrans 等。

这意味着 reward 和训练集如果只覆盖“普通编辑”，到物理/局部/身份类任务上会掉。RationalRewards 在论文里专门测了 PICA-Bench 作为 OOD physics-aware editing 压测。

## 3. RationalRewards 论文核心：PARROT 怎么把稀缺 preference 变成 reasoning reward

### 3.1 目标：不是直接学 scalar，而是学“先批判再给分”

RationalRewards 的 reward model 输出不是一个裸分数，而是：

1. User Request Analysis
2. 每个维度的 justification
3. 每个维度的 score
4. Summary
5. Refined Request

T2I 用三维：Text Faithfulness、Physical and Visual Quality、Text Rendering。

Edit 用四维：Text Faithfulness、Image Faithfulness、Physical and Visual Quality、Text Rendering。

这就把 reward 从单一 scalar 变成结构化可解释信号。训练 RL 时可以聚合成 scalar；做 test-time prompt tuning 时可以直接拿 critique 生成 refined prompt。

### 3.2 PARROT 三阶段

论文的 Preference-Anchored Rationalization（PARROT）把 rationale 当 latent variable，从 pairwise preference 数据里恢复 rationale：

| 阶段 | 做什么 | 目的 |
|---|---|---|
| Anchored generation | teacher VLM 在已知 preference label 条件下生成 rationale | 用偏好标签锚定解释，避免纯 hallucination |
| Consistency filtering | 去掉去除 label hint 后不能重新预测 preference 的 rationale | 筛掉 hallucinated / 不可判别 / 忽略标签的解释 |
| Student SFT | 训练 Qwen3-VL 8B student 在不给答案时输出 critique + score | 得到可部署 reward model |

论文报告 consistency filtering 大约保留 72% rationales。被过滤的常见失败包括：视觉 hallucination、label-ignoring rationale、vague/non-predictive reasoning。

这里有一个 edit 特有坑：consistency filtering 能筛掉“解释与 preference label 不一致”的样本，但不一定能筛掉细粒度空间误判。比如 teacher 把“左下角硬币”“右侧 logo”“局部反射”看错了，仍可能写出逻辑自洽但视觉定位错误的 critique。这类 spatial hallucination 会污染局部编辑 reward，内部落地时最好加 GroundingDINO/SAM/检测器或人工抽检来校验局部区域。

### 3.3 数据规模与效率

论文中训练数据规模：

| 来源 | 任务 | Raw pairs | Post-filtering | Final pointwise samples |
|---|---|---:|---:|---:|
| EditReward | Image Editing | 30K | 约 21.6K | 约 43.2K |
| HPDv3 + RapidData | T2I | 50K | 约 36K | 约 55K |

论文强调总训练规模约 80K raw pairs、57.6K filtering 后，比 EditReward 的 200K pairs 和 UnifiedReward 的 1M+ pairs 小 10-20 倍。核心原因不是数据魔法，而是 teacher VLM 的知识通过结构化 rationale 被蒸馏进 student。

72% 是整体 filtering 存活率，不应解读成 edit-only 的特殊存活率。RapidData 在论文表格里没有单独拆出行，T2I 侧可按 HPDv3 + RapidData 合计 50K 理解。

### 3.4 模型与实现

公开模型：

- TIGER-Lab/RationalRewards-8B-T2I
- TIGER-Lab/RationalRewards-8B-Edit

HF metadata 显示 backbone 相关标签为 Qwen/Qwen3-8B / qwen3_vl，pipeline tag 是 image-to-text，任务标签包含 text-to-image、image-to-image、edit、reasoning、reward。

GitHub 实现模块：

- `rationalrewards_sft/`：基于 LLaMA-Factory 做 SFT。
- `reward_model_evaluation/`：pairwise evaluation 和解析。
- `diffusion_rl_training/`：基于 DiffusionNFT / Edit-R1 做 Flux Kontext、Qwen Image Edit 的 RL。
- `test_time_prompt_tuning/`：Generate-Critique-Refine。

关键工程点：

- reward serving 用 vLLM。
- edit prompt 包含 source image 和 edited image。
- 分数从文本 response 中 parse。
- RL 训练时每个 prompt/source 采 group，再做 within-group normalization。
- LoRA rank 64、alpha 128、512 分辨率、训练约 16 GPU-hours per generator（论文表 11）。
- reward 模型本身另用 8×A100-80GB 服务，训练侧也用 8×A100-80GB。

工程吞吐 caveat：edit reward 一次至少吃 source + edited 两张图，vision token 和 KV cache 压力明显大于 T2I。做在线 GRPO / DiffusionNFT 时，reward server 很可能先成为瓶颈；需要限制输入分辨率、预先 resize/crop、batch 内按分辨率 bucket、启用 vLLM prefix cache，并监控 reward parse failure、zero-std ratio、reward server timeout。

## 4. RationalRewards 实验结论：对 edit reward 的启发

### 4.1 Reward model 本身确实比 scalar/open-source baseline 强

论文 Table 1 的关键数：

| Judge | MMRB2 T2I | MMRB2 Edit | EditReward Bench | GenAI T2I | GenAI Edit |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | 59.4 | 61.7 | 51.9 | 55.1 | 50.1 |
| Qwen3-VL-32B | 64.1 | 67.3 | 64.2 | 66.9 | 76.3 |
| EditReward-7B | - | 67.2 | 56.99 | - | 65.72 |
| UnifiedReward-7B | 59.8 | - | - | 67.9 | - |
| RationalRewards Qwen3-VL-8B | 64.2 | 70.3 | 66.2 | 69.8 | 80.1 |
| Gemini 2.5 Pro | 70.5 | 71.3 | 71.3 | 66.2 | 78.9 |

结论：8B reasoning reward 在 edit 上能超过 EditReward-7B，也能逼近 Gemini 2.5 Pro 的 preference prediction。

### 4.2 做 RL：RationalRewards 比 scalar reward 更稳

T2I：Qwen-Image 在 UniGen Overall 从 78.36 到 82.60；FLUX.1-dev 从 60.97 到 70.34。

Edit：

| Base | 指标 | Base | +RL EditReward | +RL Qwen3-VL-32B | +RL RationalRewards | +PT RationalRewards |
|---|---|---:|---:|---:|---:|---:|
| Flux Kontext dev | ImgEdit Overall | 3.52 | 3.66 | 3.67 | 3.84 | 4.01 |
| Qwen Image Edit | ImgEdit Overall | 4.27 | 4.25 | 4.25 | 4.38 | 4.43 |
| Flux Kontext dev | GEdit O | 6.51 | 6.88 | 6.82 | 7.37 | 7.23 |
| Qwen Image Edit | GEdit O | 7.56 | 7.77 | 7.79 | 8.29 | 8.33 |

这说明 edit reward 不是不能做，而是要多维 + preference-calibrated。裸 Qwen3-VL-32B 有 reasoning 但没偏好校准，仍不如 8B RationalRewards。

### 4.3 更反直觉：test-time prompt tuning 经常接近或超过 RL

RationalRewards 的 Generate-Critique-Refine：先生成图，再让 reward model 给四维 critique 和 refined request；如果某些分数低于阈值 3.0，就用 refined prompt 再生成。

论文称这个单轮 loop 通过 vLLM prefix caching / paged attention 只增加约 0.4s VLM inference overhead，而 RL fine-tuning 一个 base model 约 384 GPU-hours（主文说法）/ 表 11 的单 generator 训练 wall-clock 约 16 GPU-hours（训练配置口径）。

对我们实践的启发：如果 edit reward 数据很少，优先把 reward model 用作 test-time critic/prompt rewriter，比直接上 RL 更划算，尤其适合线上失败样本闭环。

## 5. 对“通用 reward 数据集”的判断

用户给的 TIGER-Lab/RationalRewards-SFTData 可以算“通用视觉生成 reasoning reward SFT 数据”，但不是万能 edit reward 数据集。

它的价值：

1. 有 T2I + image-to-image 两类。
2. 有 pairwise 和 pointwise 两种格式。
3. pointwise 里包含 refined request，可直接训练 critique/refinement 能力。
4. edit 数据显式包含 Image Faithfulness 维度。
5. 许可 MIT，HF 公开。

它的限制：

1. Edit raw pair 主要来自 EditReward 30K，覆盖面仍有限。
2. Teacher 是 Qwen3-VL-32B-Instruct，系统性盲点会被 student 继承。
3. reward 输出是文本解析，生产训练时要处理 parse failure / N/A / 维度缺失。
4. 人类偏好 bias 会继承：审美、文化、内容类型偏好未完全审计。
5. 对产品图、真人身份保持、局部电商编辑、中文文字编辑等垂类，仍需要内部数据二次校准。

所以它适合作为 edit reward warm-start / teacher / judge baseline，不适合直接当最终生产 reward。

## 6. 如果我们要做 edit GRPO / DPO / OPD reward，应该怎么落地

### 6.1 最小可行路线

1. 先用 RationalRewards-8B-Edit 当 teacher / reward server。
2. 内部构造 source + instruction + A/B edit 输出 pair。
3. 按四维打分：Text Faithfulness、Image Faithfulness、Physical Quality、Text Rendering。
4. 每个维度单独记录分数和 critique，不要一开始只存总分。
5. RL 训练时先聚合成 scalar，但日志里保留维度分，方便定位 hack。
6. test-time 先上线 critique/refined prompt loop，收集失败样本。
7. 再把失败样本转成 pairwise preference，迭代 SFT 自己的 edit reward。

更现实的优先级：先上 test-time Generate-Critique-Refine，而不是直接 RL。论文里 PICA-Bench 上 Flux Kontext baseline 41.07，+PT 到 48.12，高于 +RL 的 44.25；Qwen Image Edit baseline 49.71，+PT 到 55.65，高于 +RL 的 54.11。对 reward 稀缺阶段来说，PT 是更便宜的 sanity check 和数据闭环入口。

### 6.2 训练时 reward 聚合建议

不要用简单 mean 一把梭。建议按任务动态加权：

| 任务类型 | Text Faithfulness | Image Faithfulness | Physical Quality | Text Rendering |
|---|---:|---:|---:|---:|
| 局部替换/删除 | 高 | 极高 | 中 | 条件高 |
| 风格迁移 | 中 | 中 | 高 | 条件 |
| 人像/产品保持 | 高 | 极高 | 高 | 条件 |
| 文字编辑 | 高 | 高 | 中 | 极高 |
| 物理编辑 | 高 | 高 | 极高 | 条件 |

实践上可以先做 hard gate：Image Faithfulness < 2.5 的样本，即使美学高也不能给高总 reward。

### 6.3 数据格式建议

每条 edit reward 样本至少存：

| 字段 | 说明 |
|---|---|
| source_image | 原图 |
| instruction | 编辑指令 |
| edited_image | 编辑结果 |
| task_type | add/remove/replace/style/compose/text/identity/product 等 |
| scores.text_faithfulness | 1-4 float |
| scores.image_faithfulness | 1-4 float |
| scores.physical_quality | 1-4 float |
| scores.text_rendering | 1-4 float 或 N/A |
| critique.* | 每个维度的自然语言理由 |
| refined_request | 可选，用于 test-time prompt tuning |
| pair_id / winner | 如果来自 A/B pair，保留 pairwise label |

### 6.4 和 Flow-GRPO 的结合方式

Flow-GRPO / DiffusionNFT 的 policy optimization 可以沿用，关键是替换 reward：

- T2I：prompt + generated image -> reward。
- Edit：source image + instruction + edited image -> multidim reward。

采样组内 normalization 仍然可用，但要注意：如果同组样本都烂，std 很低，会导致 reward 区分度不够；RationalRewards 代码里有 mean threshold 0.9、std threshold 0.05、ban_prompt 等过滤逻辑。我们内部最好也记录 zero-std ratio，防止 reward server 给一堆近似常数。

## 7. 我的判断

1. “Edit reward 稀缺”是结构性问题：不是缺模型，而是缺可公开、可许可、可多维判分的 source-instruction-edit preference 数据。
2. Flow-GRPO 先做 T2I 是合理工程选择：T2I reward、prompt set、benchmark 都成熟；edit 要补 source-image pipeline 和 image faithfulness reward。
3. RationalRewards 是目前很值得 follow 的方向：它证明了 30K edit pair + PARROT reasoning distillation 可以做出比 EditReward 更强的 edit reward。
4. 对我们最有价值的不是直接照搬 RL，而是拿它的四维 rubric + critique/refined request 机制搭内部数据闭环。
5. 短期优先级：先做 edit reward 数据 schema 和 test-time critic；中期再做 LoRA RL；长期再考虑把 reward 蒸馏成更快的专用 scorer，避免训练时 VLM reward server 成本太高。

## 8. 参考资料

- RationalRewards: Reasoning Rewards Scale Visual Generation Both Training and Test Time, arXiv:2604.11626 — https://arxiv.org/abs/2604.11626
- TIGER-Lab/RationalRewards GitHub — https://github.com/TIGER-AI-Lab/RationalRewards
- RationalRewards-SFTData — https://huggingface.co/datasets/TIGER-Lab/RationalRewards-SFTData
- RationalRewards-8B-Edit — https://huggingface.co/TIGER-Lab/RationalRewards-8B-Edit
- RationalRewards-8B-T2I — https://huggingface.co/TIGER-Lab/RationalRewards-8B-T2I
- RationalRewards_DiffusionNFT_TrainData — https://huggingface.co/datasets/TIGER-Lab/RationalRewards_DiffusionNFT_TrainData
- RationalRewards EvalData — https://huggingface.co/datasets/TIGER-Lab/RationalRewards-EvalData-GenAIBench-MMRB2-ERBench
- Flow-GRPO: Training Flow Matching Models via Online RL, cited in RationalRewards references — arXiv:2505.05470
- DiffusionNFT: Online Diffusion Reinforcement with Forward Process, cited in RationalRewards references — arXiv:2509.16117
- EditReward, HPDv3, RapidData, UnifiedReward — cited as training/baseline sources in RationalRewards

## 9. 六源自检

| 源 | 是否完成 | 备注 |
|---|---|---|
| zh-cache | 是 | 本地缓存已读，无 RationalRewards 专项 |
| arXiv | 是 | browser 打开 arXiv 页面，PDF 下载并抽取文本 |
| HuggingFace | 是 | API + README + dataset sample 文件 |
| GitHub | 是 | 克隆 TIGER-AI-Lab/RationalRewards 并检查实现 |
| 知乎 | 是 | r.jina 代理尝试，当前环境 HTTP 451 |
| 小红书 | 是 | r.jina 代理尝试，当前环境 HTTP 451 |
| 已有知识库 | 是 | 已搜索 reward/RL/GRPO/edit 相关条目 |


## 10. 多模型交叉审核（2026-07-03）

### Gemini 3.1 Pro 审核补充

1. 需要把 preference prediction 和 generation benchmark 区分清楚：MMRB2 / EditReward Bench / GenAI-Bench 证明“裁判能力”，ImgEdit / GEdit / PICA 才证明“用 reward 优化生成器”的收益。
2. PARROT 的 consistency filtering 不等于彻底清洗视觉幻觉，尤其 local edit 的空间定位错误可能残留。
3. RL 实操要显式处理 Image Faithfulness 与 Editability 的 Pareto 冲突，否则模型可能走两个 hack：原图不动刷保真，或重画整图刷文本一致。
4. Edit reward server 吞吐是工程瓶颈：source + edited 双图输入会显著增加 vision tokens、KV cache 和 VLM 推理成本。

### Claude Opus 4.8 审核补充

1. RationalRewards 论文自己的 RL 实验使用 DiffusionNFT，不是 Flow-GRPO；Flow-GRPO 应只作为行业现状和 related work 讨论。
2. Image Faithfulness 是 editing-only 维度，这是 edit reward 稀缺的核心，不应被写成通用 T2I/Edit 都有的维度。
3. 数据口径需要写清：约 80K raw pairs、57.6K post-filter；72% 是整体 rationales 存活率。
4. Test-time prompt tuning 是本论文最有落地价值的反直觉结论：在 PICA-Bench 上 PT 经常超过 RL，应作为 reward 稀缺阶段的首选落地路径。
5. 审核中提出“RationalRewards-8B-Edit checkpoint 不存在”的说法与实际 HF API 不符；本次检索确认存在 TIGER-Lab/RationalRewards-8B-T2I 与 TIGER-Lab/RationalRewards-8B-Edit 两个公开 model id，因此文档保留该引用，但注明 teacher/student 关系：teacher 是 Qwen3-VL-32B-Instruct，8B 是 student/reward server。

### GLM-5.2 审核状态

GLM 审核进程超时/中断，未产出最终审核意见；不纳入合并内容。
