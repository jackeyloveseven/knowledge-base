# TBSM 迁移 FireRed：单步图像编辑的可行性、数据量与训练成本

> 记录日期：2026-08-10  
> 触发人：李承燕

## 一句话总结

TBSM 可以作为 FireRed-Image-Edit-1.1 从多步编辑器转成 one-step editor 的研究路线，但不是直接套 loss：同为 Qwen-Image 20B 骨架降低了模型适配难度，真正的硬问题是把“参考图 + 编辑指令”注入 tracker，并补足实例级保真约束。建议先用 10k–20k 高质量编辑对、300–1000 updates 做 8×H200 全参/FSDP PoC；1×H200 只能做 LoRA/局部解冻验证，且 LoRA 是否有足够容量完成单步行为迁移尚未被论文证明。

## 已核实事实

- TBSM：Three-Body Scattering for Generative Modeling，arXiv:2607.18198。
- 官方代码已发布 ImageNet 与 minimal 实现，但 README 的 Qwen-Image-20B training code/checkpoint 仍为 TODO。
- 论文 Qwen-Image-20B 初步实验：`rho=1, lambda=0`，batch 16，1000 generator updates，generator LR `5e-6`；三路冻结表征为 Qwen2.5-VL、DINOv3、SigLIP，每路一个 tracker；NFE=1，无 CFG、teacher query、score network 或 adversarial critic。
- 该 T2I 结果目前只有定性样例，作者明确称为 initial 1,000-step study，尚无编辑任务、定量 benchmark、训练吞吐或硬件配置。
- FireRed-Image-Edit-1.1 使用 `QwenImageEditPlusPipeline`。HF 权重索引显示 diffusion transformer BF16 约 40.86GB，即约 20.43B 参数；text encoder 另约 8.29B 参数且应冻结。

## 代码级训练结构

TBSM 官方实现的核心 step：

1. 从真实目标 latent `real_x` 与噪声构造 `x_t`。
2. generator 一次前向得到 projectile/final prediction。
3. 把真实目标与生成结果解码成 RGB。
4. 分别送入冻结表征网络；projectile 分支保留梯度，real/source 分支 detach。
5. tracker 估计条件散射场，generator 回归 detached target。
6. generator 与 tracker 各做一次 optimizer update，generator 维护 EMA。

重要纠正：官方代码在 `lambda=0` 时不会执行 independently-generated-source 的额外 generator forward；因此 Qwen 配置不是“两次 generator forward”。主要额外成本来自 VAE decode、三路冻结表征前向/反传到 projectile，以及三个 tracker。

## 迁移到 FireRed Edit 的关键改造

### 1. Tracker 条件必须从纯文本升级为编辑条件

T2I tracker 只 cross-attend 文本条件；编辑任务至少需要：

```text
condition = [edit instruction tokens, reference-image tokens, optional edit mask/task token]
```

否则 tracker 不知道输入人物、商品、布局和背景原本是什么，只能把输出拉向“目标图分布”，无法判断哪些区域必须保持。

推荐实现：复用 FireRed/Qwen 多模态 encoder 已产生的 reference+text condition sequence，给每个 tracker 增加 cross-attention；不要另起一个推理时 adapter，训练后 tracker 全部丢弃，保持 NFE=1 推理零额外模型。

### 2. TBSM 分布匹配不天然保证实例级编辑保真

至少要额外验证/补充：

- identity/主体保持：DINO/人脸/商品 embedding 的 source-target relation；
- 非编辑区域保持：有 mask 时做 region-aware feature loss；无 mask 时可由 change detector/pseudo mask 提供软权重；
- 指令完成：VLM edit judge 或任务专用 reward 仅用于评估/可选辅助训练；
- 文本/OCR、几何与布局任务需独立 track，不能只看通用图像表征。

### 3. 全参和 LoRA 不是等价路线

官方 TBSM 更新完整 generator backbone。20.43B transformer 的 BF16 权重约 40.9GB；仅权重+梯度+Adam moments 约 245GB，若含 FP32 master copy 可到约 327GB，尚未计激活与冻结 encoder。

- 8×H200：用 FSDP/ZeRO-3 + activation checkpointing，具备全参 PoC 条件。
- 1×H200 141GB：全参 AdamW 基本不可行；只能 LoRA、局部 block 解冻或 optimizer/offload。LoRA 能否把多步模型改造成稳定 one-step editor 是未验证假设，不能直接当正式方案。

## 数据量与更新规模

论文 Qwen PoC 的曝光量是：

```text
1000 updates × batch 16 = 16,000 sample exposures
```

对 FireRed 建议分三档：

| 阶段 | 唯一编辑对 | Updates / global batch | 曝光量 | 目的 |
|---|---:|---:|---:|---|
| Smoke | 1k–3k | 50–100 / 16 | 800–1,600 | 验证 loss、显存、梯度和一阶成图 |
| PoC | 10k–20k | 300–1,000 / 16 | 4,800–16,000 | 判断 NFE=1 是否具备指令完成与基本保真 |
| Generalization | 100k–300k 高质量池 | 3,000–10,000 / 16 | 48k–160k | 覆盖人物、商品、文字、多图、增删改、局部保持等长尾 |

关键不是把全部数据跑完一个 epoch，而是每个 update 的任务类型、难度、参考图条件和 seed 足够分散。第一轮不建议直接上 50 万低质 pair。

## 训练时长：规划区间，不是公开 benchmark

论文未公开 Qwen 的 GPU 型号和 sec/step；当前也没有 TBSM-FireRed 实测，因此下面只作为资源排期假设，必须先跑 20–50 step profiler 校准。

假设 8×H200 全参/FSDP 的真实 update time 为 30–90 秒：

| Updates | 估计 wall-clock |
|---:|---:|
| 100 | 0.8–2.5 小时 |
| 300 | 2.5–7.5 小时 |
| 1,000 | 8.3–25 小时 |
| 3,000 | 25–75 小时 |

假设 1×H200 LoRA/局部解冻、global batch 16 依赖 gradient accumulation，update time 为 180–600 秒：

| Updates | 估计 wall-clock |
|---:|---:|
| 100 | 5–16.7 小时 |
| 300 | 15–50 小时 |
| 1,000 | 50–166.7 小时（约 2–7 天） |

这些区间的最大不确定性来自 1024 分辨率 token 数、参考图数量、activation checkpointing、FSDP 通信、三路表征串并行方式和 VAE decode。

## 建议决策

- 应用可行性：**中等偏高，值得做 PoC；正式应用仍是研究项目。**
- 首选路线：8×H200、全参/FSDP、`rho=1, lambda=0` 对齐论文配置；tracker 接 FireRed 原生 reference+text condition。
- 首轮数据：10k–20k 高质量、任务均衡编辑对；300-step 看趋势，1000-step 做正式 PoC 判断。
- 不建议：直接在单 H200 上用 LoRA 跑 1000 步后据此否定方法；也不建议没有 reference-aware tracker 就直接训练。
- 成功门槛：NFE=1 相对原始多步模型不仅要“能出图”，还需同时过 edit instruction、identity/source fidelity、非编辑区保持、多 seed 稳定性与候选多样性。

## 多源检索状态

- arXiv：命中 TBSM 原论文 2607.18198。
- GitHub：命中官方实现 `sp12138/TBSM`，已核对 README、训练配置与核心 loss 代码。
- Hugging Face：TBSM 搜索无模型结果；命中 FireRed-Image-Edit-1.1 官方模型与参数索引。
- zh-cache：无 TBSM 条目，命中 FireRed 泛相关条目。
- Google：代理被 403 abuse protection 阻断。
- 知乎：返回安全验证页，未获得内容。
- 小红书：仅返回页脚备案信息，未获得内容。

## 多模型交叉审核

- GLM-5.2：强调编辑三元组与单 H200 全参显存不可行；其“需先把 FireRed 蒸馏成 one-step 再套 TBSM”表述不准确，TBSM 本身就是从多步 checkpoint 直接训练 one-step generator。
- Gemini 3.1 Pro：建议 tracker 显式注入 source image，DINO 分支承担非编辑区保持；其 100k–500k 数据建议适合作为泛化池，不适合直接作为首轮 PoC 门槛。
- Claude Opus：强调 Qwen 结果仅是初步定性研究、LoRA 无论文背书、wall-clock 必须实测；其“lambda=0 仍有额外 generated-source forward”与官方代码冲突，未采纳。

## 参考资料

- [TBSM 论文](https://arxiv.org/abs/2607.18198)
- [TBSM 官方代码](https://github.com/sp12138/TBSM)
- [FireRed-Image-Edit-1.1](https://huggingface.co/FireRedTeam/FireRed-Image-Edit-1.1)
- [FireRed-Image-Edit-1.0 Technical Report](https://arxiv.org/abs/2602.13344)
