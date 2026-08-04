# 生成与编辑联合预训练：Z-Image 如何把 T2I / I2I 放进同一基础训练

> 记录日期：2026-08-03  
> 触发人：李承燕  
> 核心来源：Z-Image Technical Report v5（arXiv:2511.22699）

## 一句话总结

Z-Image 不是等纯文本生图基座训完后才第一次接触编辑数据，而是在 omni-pre-training 阶段把 T2I 与 I2I 混合训练：用单流 S3-DiT 统一处理文本、参考图语义 token、参考图 VAE token 与目标图 token，让模型提前学习自然图像对之间的关系，再分别进入生成 SFT 与编辑 continued pre-training/SFT；这个设计降低了编辑模型的冷启动难度，但并不等于编辑后训练可以省掉。

## 1. 这项设计到底做了什么

Z-Image 的完整预训练分两段：

1. 低分辨率预训练：只做 256×256 T2I，负责基础跨模态对齐、视觉知识和中英文字渲染等能力注入。
2. Omni-pre-training：同时引入任意分辨率、T2I/I2I 联合训练、多级双语 caption。

其中联合训练的核心是：

- T2I 样本：文本条件 → 目标图像。
- I2I 样本：参考图像 + 文本条件 → 目标图像。
- I2I 文本条件随机采用目标图 caption 或图像对 difference caption：
  - 目标图 caption 对应 reference-guided image generation，模型看到参考图但按目标语义生成。
  - difference caption 对应 multi-task image editing，直接描述 source→target 的变化。

Omni-pre-training 结束后，基础模型已能同时接收图像和文本条件，并支持约 1K–1.5K 范围的任意分辨率，成为 Z-Image 与 Z-Image-Edit 的共同初始化。

## 2. 代码/架构层面怎么统一

不是额外挂一套 ControlNet 或独立 I2I UNet，而是同一个 6.15B S3-DiT 主干内化两类任务。

具体组件：

- 文本编码器：Qwen3-4B。
- 图像 tokenizer：Flux VAE。
- 编辑专用高层参考语义：SigLIP 2。
- 主干：30 层 Single-Stream MM-DiT，hidden size 3840，32 attention heads，FFN intermediate size 10240。
- 输入组织：各模态先经过轻量 modality-specific processor（论文称每个 processor 由两个 Transformer block 构成）做初始对齐；随后文本 token、SigLIP 视觉语义 token、VAE 图像 token 在 sequence 维直接拼接，进入统一 self-attention 主干。
- 位置区分：论文明确写明参考图与目标图使用对齐的空间 RoPE 坐标，同时在 temporal 维加入单位间隔偏移，以区分两张图的身份。
- 噪声区分：论文明确写明参考图和目标图使用不同 time-conditioning，区分 clean condition image 与 noisy target image；但没有进一步公开具体数值编码和实现代码。
- Attention 可见性：技术报告没有交代 reference/target token 是否使用特殊 attention mask，不能自行假设为 causal、单向或 block-diagonal。
- 训练目标：Flow Matching，目标仍是预测从高斯噪声到目标图 latent 的 velocity；T2I/I2I 的差异主要体现在条件 token 是否包含参考图。

因此它的工程本质是统一条件接口：

`conditions = [text_tokens, optional_siglip_ref_tokens, optional_vae_ref_tokens]`

`prediction = S3_DiT(conditions, noisy_target_tokens, timestep)`

T2I 时 reference 分支为空；I2I 时把参考图 token 拼入同一序列。主干参数不是两套，而是共享的。

## 3. 弱对齐自然图像对从哪里来

Z-Image 不只依赖昂贵的严格编辑 triplet，而是组合多种来源：

### 3.1 任务专家合成

先定义编辑 taxonomy，再用任务专用 expert model 生成高质量编辑对。多个编辑动作可合并到同一对中，让一条样本同时训练多种编辑能力。

### 3.2 图结构组合扩增

对同一输入生成 N 个不同编辑版本，再对原图与各版本做有向排列组合：同一组图可以构造大量 source→target / inverse pairs，不必为每条新关系重新调用生成模型。它还能把两个单任务编辑版本组合成混合编辑关系。

### 3.3 视频帧天然配对

从视频中收集天然成组的帧。它们共享主体、场景或风格，但姿态、视角、背景、光照可能同时变化，因此形成弱对齐、复杂且多样的 I2I 关系。团队再用 CN-CLIP embedding cosine similarity 过滤出语义相关度较高的帧对。

### 3.4 可控文字渲染

针对文字编辑数据稀缺问题，用可控渲染系统生成精确的文字内容、字体、颜色、大小和位置变化，从而获得带确定 instruction ground truth 的编辑对。

“弱对齐”不等于无关图片乱配，而是 source/target 具有可学习关系，但未必像人工局部编辑数据那样严格保持像素级一致。

## 4. Difference Caption 怎么生成

Z-Captioner 用三步 CoT 把图像对变成编辑指令：

1. 分别给 source 和 target 生成包含 OCR 的详细 caption。
2. 联合原图与 caption，从视觉和文字两个角度枚举差异。
3. 把差异压缩成简洁的 source→target 编辑 instruction。

这个设计把“图像对存在关系”转换成可监督的语言条件。对自然视频帧这类弱对齐数据尤其重要，因为它们没有现成编辑标注。

## 5. 为什么联合预训练可能不伤 T2I

作者报告联合方案没有观察到明显的 T2I 性能下降，但论文没有给出独立、量化的 T2I-only vs T2I+I2I ablation 表，因此应把它理解为作者的工程观察，而非已被公开数字充分证明的结论。

从实现上看，可能成立的原因有四个：

1. 目标统一：两类任务最终都对目标图 latent 做 Flow Matching，I2I 只是增加条件，不改变主目标空间。
2. 参数共享：T2I 与 I2I 共用 S3-DiT，I2I 数据也在学习构图、语义对应、风格迁移与图像统计，不是完全异质任务。
3. 条件可选：模型可以通过 reference token 是否存在、time-conditioning 与 RoPE temporal offset 区分任务，减少条件混淆。
4. 数据比例控制：在后续编辑 continued pre-training 中，团队明确保持较高 T2I 比例，建议 T2I:I2I≈4:1，以防生成质量退化。

但第 4 点来自后续编辑训练，不代表 omni-pre-training 的具体采样比例；论文没有披露 omni 阶段精确 T2I/I2I mixing ratio。

## 6. 它不等于“一次联合预训练就训完编辑模型”

Z-Image-Edit 后面仍有独立两阶段训练：

1. Continued pre-training for editing：编辑对与 T2I SFT 数据混合；先在 512×512 跑数千步快速适配，再升到 1024×1024。
2. Editing SFT：人工构造 task-balanced 高质量子集，重点增强 instruction following；与真实用户分布差异较大的合成数据会被大幅降采样。

所以 omni-pre-training 的作用是提供编辑先验与良好初始化，而真正的精确指令遵循仍靠后续高质量编辑训练。

## 7. 与常见训练路线对比

| 路线 | 基座阶段 | 编辑启动方式 | 优点 | 主要风险 |
|---|---|---|---|---|
| T2I-only → Edit SFT | 基座只见文本和目标图 | 后期第一次加入参考图 | 简单、任务边界清晰 | 编辑冷启动；昂贵编辑对承担全部能力注入 |
| Z-Image 式联合预训练 | 基座阶段同时见 T2I/I2I | 先用大规模弱对齐图像对学关系，再做编辑精训 | 编辑初始化强；共享视觉关系知识；可充分利用预训练算力 | 数据 mixing 和条件区分设计不好会导致 task interference |
| 完全独立生成/编辑模型 | 两套模型分别预训练 | 无共享初始化 | 可分别优化 | 训练成本最高；知识与数据无法复用 |
| 推理时 adapter/ControlNet | 不要求联合预训练 | 推理时增加条件模块 | 改造快 | 增加推理延迟和模型加载，不等于能力内化 |

## 8. 对我们训练 FireRed / Qwen Edit 类模型的启发

最值得抄的不是“把编辑数据提前混进去”这一句，而是以下完整组合：

1. 先用大量便宜、弱对齐但有关系的自然图像对学习通用 I2I prior。
2. 用 difference caption 将图像关系语言化，不要求所有数据都来自严格人工编辑。
3. 在统一 DiT 主干中用 optional reference tokens 区分 T2I/I2I，避免新增推理模块。
4. 后续编辑阶段继续混入较高比例 T2I 数据，防止生成质量和世界知识被窄编辑分布冲掉。
5. 最后用少量 task-balanced、真实用户分布的高质量数据收 instruction following。

建议的内部实验不是直接全量照搬，而是先做四组可证伪 ablation：

- A：T2I base → 纯高质量 Edit SFT。
- B：T2I base → 弱对齐 I2I continued pre-training → Edit SFT。
- C：从预训练期联合 T2I/I2I → Edit SFT。
- D：同 C，但改变 T2I:I2I mixing ratio。

统一评测：T2I 侧至少覆盖 GenEval / DPG-Bench、审美、OCR 与双语文字渲染；I2I 侧覆盖多类编辑成功率、source preservation、identity consistency，并记录达到同等指标所需的训练步数与严格编辑对数量。配比不要只测 4:1，可从纯 T2I、8:1、4:1、2:1 到更高 I2I 比例做 sweep。重点验证两个问题：联合预训练是否真的不伤 T2I；它节省的是编辑数据量、训练步数，还是只改善最终上限。

## 9. 已确认事实与未披露项

已确认：

- Z-Image 使用 6.15B、30 层单流 S3-DiT。
- 低分辨率预训练仅做 256×256 T2I。
- Omni-pre-training 联合 T2I 与 I2I，并使用自然存在、弱对齐图像对。
- 编辑任务使用 SigLIP 2 + VAE reference token，并与文本/目标 token 进入同一序列。
- Omni 后仍对 Z-Image-Edit 做 continued pre-training 与 SFT。
- 编辑 continued pre-training 建议 T2I:I2I=4:1，先 512×512 数千步，再升 1024×1024。
- 训练成本表给出：低分辨率预训练 147.5K H800 GPU hours、Omni-pre-training 142.5K、Post-training 24K，总计 314K。

未披露/不可直接推出：

- Omni-pre-training 阶段精确 T2I:I2I mixing ratio。
- “不损伤 T2I”的独立定量 ablation 数字。
- 弱对齐视频帧对的总规模、过滤阈值与各数据源占比。
- Z-Image-Omni-Base / Z-Image-Edit 权重当前在官方 README 中仍标为待发布；公开 HF 的 Z-Image checkpoint 标记为 text-to-image，不能直接当作编辑基座使用。

## 10. 多模型交叉审核（2026-08-03）

### 来自 GLM-5.2

- 建议把复现实验补成 mixing-ratio sweep，而不是把 4:1 当最优结论；已采纳。
- 建议补充 T2I 与 I2I 的最小回归评测；已加入 GenEval / DPG-Bench 及编辑保持类指标。
- Reviewer 质疑 Qwen3-4B 版本号；经 Z-Image v5 技术报告 Section 4.1 核验，Qwen3-4B 是官方明确配置，该质疑不采纳。
- Reviewer 提议补中文社区部署生态；与本条目“联合预训练机制”主线关联较弱，未展开。

### 来自 Gemini 3.1 Pro

- 指出 Flux VAE 与 SigLIP 2 分别承担空间细节与高层语义，且需明确注入方式；已根据论文补充“两层 modality processor 后按序列拼接”。
- 提醒 attention mask 可能影响 clean/noisy token 交互；论文未披露具体 mask，已列入未公开实现边界，未采用 reviewer 对 causal/block mask 的猜测。
- 建议把 time-conditioning 放回 Flow Matching 统一目标理解；已保留统一 velocity prediction 表述，并避免猜测参考图 time 数值。
- 建议与 channel-concat I2I 路线对比；该方向有价值，但需另行核验相关论文实现，本条目不作未经查源的强结论。

### 来自 Claude Opus 4.8

- 强调“未观察到 T2I 下降”只能写成作者观察，不能写成已证实因果；已在正文明确。
- 要求给 RoPE temporal offset 与不同 time-conditioning 标出原文依据；已由技术报告 Section 4.1 核验为 Z-Image 明确设计。
- 强调 4:1、512→1024 是 editing continued pre-training 配方，而不是 omni 阶段已证明最优值；正文已区分。
- 指出弱对齐数据与 difference caption 存在因果依赖：后者负责把复杂自然变化转成可监督编辑语义；已在 Difference Caption 小节明确。

## 参考资料

- Z-Image Technical Report v5: https://arxiv.org/abs/2511.22699
- Z-Image 官方 GitHub: https://github.com/Tongyi-MAI/Z-Image
- Z-Image Hugging Face: https://huggingface.co/Tongyi-MAI/Z-Image
- Z-Image 官方项目页: https://tongyi-mai.github.io/Z-Image-blog/
- 中文来源检索：zh-cache 仅命中与 Z-Image 参考图条件化相关的团队周报，未覆盖本问题；知乎实时搜索被 CAPTCHA 阻断；小红书搜索页未返回有效结果；Google/Jina 请求被 403 阻断。
