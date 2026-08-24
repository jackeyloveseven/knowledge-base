1|# Qwen Image Edit 4-step Lightning LoRA 复现路线与 Qwen Image Flash 实现拆解
2|
3|> 记录日期：2026-07-08 / 触发人：李承燕  
4|> 原问题：qwen image edit 的 4 steps lightning 4 步加速推理 LoRA 一直备受瞩目，但是难以复现，找寻训练办法；同时从这出发看看 qwen image flash 是怎么实现的。  
5|> 关联条目：06_LightningLoRA对比.md
6|
7|---
8|
9|## 一句话总结
10|
11|公开资料没有给出 lightx2v / ModelTC 的完整训练 recipe；但从其模型卡、推理代码、FP8 兼容说明、SDXL-Lightning 论文和 OPPO Qwen pruning 训练代码交叉看，最可信的复现路线不是普通 LoRA SFT，而是：对 Qwen Image Edit 的 transformer 注入较高容量 LoRA（rank 64/128 是工程推断，不是官方公开值），固定 4-step FlowMatch Euler 轨迹，使用 base model 多步 teacher 输出 + CFG 蒸馏 + 分布/对抗或 DMD 类损失做 step distillation；Qwen Image Flash / FlashPack 则更像“工程封装版”：把 step-distilled/quantized transformer、text encoder、VAE、scheduler 打包成自定义 diffusers pipeline，并可能使用自定义 flashpack 权重格式，而不是一个新的训练范式。
12|
13|---
14|
15|## 搜索与证据状态
16|
17|| 来源 | 状态 | 关键结果 |
18||---|---|---|
19|| 已有知识库 | 已搜索 | 06 已记录 Lightning LoRA 本质：LoRA 参数化相同，训练目标是 step distillation；SDXL-Lightning 源于 Progressive Adversarial Diffusion Distillation。 |
20|| zh-cache | 已搜索 | 命中 Qwen-Image-2.0 技术报告精读、Qwen Image Edit 2511 项目相关条目；未命中 Lightning 训练细节。 |
21|| arXiv | 已搜索 | Qwen + Lightning + Flash 无直接论文；相关论文包括 SDXL-Lightning 2402.13929、LADD 2403.12015、DMD2 2405.14867、PPCL 2511.16156。 |
22|| HuggingFace | 已搜索 | lightx2v/Qwen-Image-Edit-2511-Lightning 存在，下载量约 36 万；模型卡明确说 Step Distillation + FP8 Quantization；未公开训练脚本。 |
23|| GitHub | 已搜索 | ModelTC/Qwen-Image-Lightning 只公开推理与评测；OPPO-Mente-Lab/Qwen-Image-Pruning 公开 teacher-student distillation 训练脚本；社区 repo 主要是部署封装。 |
24|| Google/Bing 中文 | 已尝试 | r.jina.ai 返回 403，未搜到。 |
25|| 知乎/小红书 | 已尝试 | r.jina.ai 返回 403，未搜到。 |
26|
27|---
28|
29|## 1. 现有公开实现到底暴露了什么
30|
31|### 1.1 lightx2v/Qwen-Image-Edit-2511-Lightning
32|
33|HuggingFace 模型卡明确列出三个核心文件：
34|
35|| 文件 | 类型 | 含义 |
36||---|---|---|
37|| Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors | 4-step Distilled LoRA | 轻量 LoRA，加到 Qwen/Qwen-Image-Edit-2511 上 4 步推理。 |
38|| Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors | 4-step Distilled LoRA | fp32 版本，通常更适合继续转换/合并。 |
39|| qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors | FP8 Quantized fused model | base + 4-step distilled LoRA + FP8 量化/scale 处理后的部署权重。 |
40|
41|模型卡直接写了两个优化：Step Distillation 和 FP8 Quantization。它没有放训练脚本，也没有公开 loss 配方。
42|
43|### 1.2 ModelTC/Qwen-Image-Lightning 推理代码泄露出的关键约束
44|
45|官方 generate_with_diffusers.py 在加载 LoRA 时重建了 scheduler：
46|
47|- scheduler：FlowMatchEulerDiscreteScheduler
48|- base_shift = log(3)
49|- max_shift = log(3)
50|- 顶层 shift = 1.0
51|- shift_terminal = None
52|- num_train_timesteps = 1000
53|- use_dynamic_shifting = True
54|- true_cfg_scale：有 LoRA 时默认 1.0；无 LoRA 时默认 4.0
55|- Qwen-Image-Edit-2509 / 2511 走 QwenImageEditPlusPipeline
56|
57|这个很关键：它说明 Qwen Lightning 不是 SDXL-Lightning 那套 “Euler + trailing + CFG=0” 原样搬运；Qwen 的 flow matching pipeline 把 distilled inference 固定到 `base_shift=max_shift=log(3)`、顶层 `shift=1.0`、`use_dynamic_shifting=True` 的 FlowMatch Euler 配置，并把 CFG 降到 1.0。也就是说训练时大概率就是按这套 scheduler / dynamic-shifting / CFG 目标蒸馏的，不能简单写成静态 `shift=log(3)`。
58|
59|### 1.3 FP8 兼容说明暴露出的训练细节
60|
61|ModelTC README 提到：直接把 BF16 base 的 Lightning LoRA 加到普通 FP8 base 会出现网格伪影。原因是普通 qwen_image_fp8_e4m3fn 是直接 downcast，没有校准 scaling。解决方式有两个：
62|
63|1. 用 BF16 guidance 去 distill FP8 base，得到专门适配 FP8 base 的 Lightning LoRA。
64|2. 发布 scaled FP8 base，让 BF16 base 上训练的 LoRA 也能兼容。
65|
66|这说明它们确实在做 teacher-student distillation，而且 LoRA 对 base quantization 分布非常敏感。复现时不能“先训 BF16 LoRA，再随便套 FP8 base”，必须把量化分布纳入训练或校准。注意：4-step LoRA 本身公开为 BF16/FP32；FP8 文件是 fuse/quantize 后的部署产物，不要把 FP8 当成 Lightning LoRA 的训练必要条件。
67|
68|---
69|
70|## 2. 为什么难以复现
71|
72|| 难点 | 具体原因 | 复现建议 |
73||---|---|---|
74|| 不是普通 LoRA SFT | 普通 LoRA 只学风格/任务，不改变采样轨迹；4-step LoRA 要把 40/50-step 去噪轨迹压进 4 step。 | 训练目标必须是 step distillation，而不是 noise MSE SFT。 |
75|| Qwen 是 Flow Matching DiT | 目标不是 SD 的 epsilon/noise prediction，而是 flow/velocity 场；scheduler shift 影响巨大。 | 固定 FlowMatchEulerDiscreteScheduler，训练/推理 shift 一致。 |
76|| 编辑任务输入复杂 | Qwen Edit Plus 支持多图输入，condition 包含图像 latent + VL text encoder + edit instruction。 | 训练样本要覆盖 single/multi image edit，并缓存 teacher condition。 |
77|| CFG 蒸馏难 | 少步下强 CFG 容易炸；Qwen Lightning 默认 true_cfg_scale=1.0。 | teacher 用高 CFG 产出目标，student 低/无 CFG 学 implicit guidance。 |
78|| LoRA rank/层选择敏感 | 4-step 需要大幅改变 vector field，低 rank 容量可能不够；官方未公开 rank。 | rank 从 64/128 起步是工程推断，优先覆盖 attention + FFN projection；必要时做 LoRA + 部分层 full FT 消融。 |
79|| 对抗/DMD 损失不稳定 | 纯轨迹 MSE 容易糊，GAN/DMD 容易崩。 | 分阶段：先 teacher regression，再加分布损失；不要一上来 GAN。 |
80|| FP8 兼容敏感 | base quantization 改变权重/激活分布，LoRA 会产生 grid artifact。 | 先 BF16 复现，再做 scaled FP8；或直接在 FP8 base 上蒸馏。 |
81|
82|---
83|
84|## 3. 最可信的训练办法：三阶段复现路线
85|
86|### Phase A：先做可跑通的 8-step distillation，不碰 GAN
87|
88|目标：先证明 LoRA 能稳定把 Qwen Image Edit 从 40 step 压到 8 step。
89|
90|模型设置：
91|
92|| 模块 | 设置 |
93||---|---|
94|| Base | Qwen/Qwen-Image-Edit-2511 或 2509 |
95|| 可训练参数 | transformer LoRA；text encoder / VAE 冻结 |
96|| LoRA 注入层 | attention q/k/v/o + FFN up/down/gate/proj，先别只训 attention |
97|| rank | r=64 起步；显存够直接 r=128。注意这是复现建议，不是官方公开超参。 |
98|| dtype | BF16 训练，AdamW 8-bit 可选 |
99|| scheduler | FlowMatchEulerDiscreteScheduler，base_shift=max_shift=log(3)，顶层 shift=1.0，shift_terminal=None，use_dynamic_shifting=True |
100|| student steps | 8 |
101|| teacher steps | 40 或 50；teacher CFG/true_cfg 使用原版推荐值 |
102|
103|训练数据：
104|
105|| 数据类型 | 比例 | 用途 |
106||---|---:|---|
107|| 原始编辑对 image_before + instruction + image_after | 40% | 保编辑任务基本能力。 |
108|| teacher 生成伪标签 image_before + instruction + teacher_output | 40% | 对齐 base teacher 的分布。 |
109|| hard cases：文字、人物身份、商品、局部编辑、多图融合 | 20% | 防止 4/8 step 最常掉的能力崩。 |
110|
111|最小损失：
112|
113|1. latent x0 reconstruction：student 8-step 输出 decoded/latent 对齐 teacher final latent 或真实 after latent。
114|2. velocity/flow matching：随机取少步 schedule 上的 t，让 student 预测 teacher 的 velocity/flow。
115|3. condition dropout / CFG distill：用 teacher 高 guidance 结果做 target，student 训练/推理保持 true_cfg_scale≈1.0。
116|
117|这个阶段不要追求 4 step，先看 8 step 是否能达到“结构不崩 + 编辑方向正确”。
118|
119|### Phase B：8-step → 4-step progressive distillation
120|
121|目标：用 8-step LoRA 初始化 4-step LoRA。
122|
123|训练方式：
124|
125|1. student 初始化：base + 8-step LoRA。
126|2. teacher：base 40/50-step teacher 或 8-step student EMA，两者混用。
127|3. schedule：固定 4 个 inference timestep，不要每 batch 乱采一套 scheduler。
128|4. loss：继续 latent/flow regression，但加 LPIPS/DINO/CLIP/Qwen-VL image-text/edit consistency 之类感知约束。
129|5. EMA：保留 student EMA；评估用 EMA 权重。
130|
131|关键点：4-step 不是“把 num_inference_steps 改成 4 继续训”。必须让训练时的 t-grid、推理时的 t-grid、scheduler shift 完全一致。否则推理 4 步时模型见不到训练分布。
132|
133|### Phase C：加分布损失，把糊图拉回来
134|
135|如果 Phase B 出图方向对但糊、塑料、细节掉，可以再加一种分布级损失：
136|
137|| 方法 | 适用性 | 说明 |
138||---|---|---|
139|| SDXL-Lightning 式 Progressive Adversarial Diffusion Distillation | 高 | 论文证明 1/2/4/8 step 有效；但判别器工程复杂。 |
140|| LADD | 高 | 用 latent diffusion feature 做 adversarial distillation，比 pixel discriminator 更适合高分辨率。 |
141|| DMD2 | 中高 | 可避免大量 teacher path regression，但训练稳定性和 fake critic 很吃实现。 |
142|| LCM/Consistency | 中 | 更简单，但 Qwen Edit 的复杂条件和文字能力可能掉。 |
143|
144|实操建议：别一上来复刻完整 SDXL-Lightning。更稳的路线是：先 regression 得到可用 4-step，再加轻量 adversarial / DMD loss 微调 5k-20k steps。
145|
146|---
147|
148|## 4. 训练伪代码骨架
149|
150|下面是逻辑骨架，不是可直接运行脚本：
151|
152|```python
153|# 1. load teacher/base
154|teacher = QwenImageEditPlusPipeline.from_pretrained(base, torch_dtype=bf16).eval()
155|student = QwenImageEditPlusPipeline.from_pretrained(base, torch_dtype=bf16)
156|
157|# 2. inject LoRA into transformer
158|student.transformer = inject_lora(
159|    student.transformer,
160|    target_modules=["to_q", "to_k", "to_v", "to_out", "ffn", "proj", "gate"],
161|    rank=128,
162|    alpha=128,
163|)
164|freeze(student.text_encoder)
165|freeze(student.vae)
166|
167|# 3. fixed distilled scheduler
168|scheduler = FlowMatchEulerDiscreteScheduler.from_config({
169|    "base_shift": math.log(3),
170|    "max_shift": math.log(3),
171|    "shift_terminal": None,
172|    "use_dynamic_shifting": True,
173|    "num_train_timesteps": 1000,
174|})
175|
176|for batch in loader:
177|    cond = encode_condition(batch.images, batch.prompt)
178|
179|    with torch.no_grad():
180|        # expensive but most faithful; production training should cache these
181|        teacher_latent = teacher.sample_latent(
182|            cond,
183|            steps=40,
184|            true_cfg_scale=4.0,
185|            scheduler=base_scheduler,
186|        )
187|
188|    # student only runs 4 or 8 distilled steps
189|    student_latent, aux = student.sample_latent_with_trace(
190|        cond,
191|        steps=4,
192|        true_cfg_scale=1.0,
193|        scheduler=distill_scheduler,
194|    )
195|
196|    loss_x0 = mse(student_latent, teacher_latent)
197|    loss_v = flow_distill_loss(aux.student_v, aux.teacher_v_on_same_t)
198|    loss_clip = edit_semantic_loss(student_latent, batch.prompt, batch.source_images)
199|    loss = loss_x0 + 0.5 * loss_v + 0.05 * loss_clip
200|
201|    if phase_c:
202|        loss += 0.01 * adversarial_or_dmd_loss(student_latent, teacher_latent, real_after=batch.after)
203|
204|    loss.backward()
205|    optimizer.step()
206|    ema.update(student.transformer.lora_params())
207|```
208|
209|工程上 teacher_latent 和 teacher_v 最好离线缓存，不然 13B 级 Qwen Edit 每 step 都跑 40 步 teacher，训练成本会爆炸。
210|
211|---
212|
213|## 5. Qwen Image Flash / FlashPack 是怎么实现的
214|
215|目前搜到的公开 “Qwen Image Flash” 更明确的是 blanchon/Qwen-Image-Edit-2509-FlashPack 这类 FlashPack。其 model_index.json 显示：
216|
217|| 组件 | 原始/常规 | FlashPack 版本 |
218||---|---|---|
219|| pipeline | QwenImageEditPlusPipeline | FlashPackQwenImageEditPlusPipeline |
220|| text_encoder | Qwen2_5_VLForConditionalGeneration | FlashPackQwen2_5_VLForConditionalGeneration |
221|| transformer | QwenImageTransformer2DModel | FlashPackQwenImageTransformer2DModel |
222|| VAE | AutoencoderKLQwenImage | FlashPackAutoencoderKLQwenImage |
223|| 权重格式 | safetensors / diffusers shard | model.flashpack |
224|| scheduler | FlowMatchEulerDiscreteScheduler | FlowMatchEulerDiscreteScheduler，base_shift=0.5/max_shift=0.9/shift_terminal=0.02 |
225|
226|所以 FlashPack 看起来不是“另一个 Lightning 训练论文”，而是一个部署打包/推理优化 pipeline：
227|
228|1. 自定义 diffusers pipeline 类。
229|2. text encoder / transformer / VAE 都换成 FlashPack wrapper。
230|3. 权重以 model.flashpack 格式存储。
231|4. scheduler_config 不同于 lightx2v Lightning 的 log(3) 配置，说明它可能是独立校准或面向 FlashPack pipeline 的默认推理配置。
232|5. HF model card 是自动生成模板，没有训练细节；不能断言它做了新的蒸馏训练。
233|
234|与 lightx2v Lightning 的区别：
235|
236|| 维度 | lightx2v Qwen Lightning | Qwen Image FlashPack |
237||---|---|---|
238|| 核心目标 | 少步 distillation LoRA + FP8 quantization | pipeline/权重格式/运行时打包优化 |
239|| 公开证据 | 模型卡明确 Step Distillation；LoRA 文件存在 | model_index 显示 FlashPack custom classes 和 model.flashpack |
240|| 是否 LoRA | 是，有 4/8-step LoRA | 未显示 LoRA 文件，像完整 pipeline 包 |
241|| 是否训练范式 | 是，至少做了 step distillation | 未公开，更多是部署形态 |
242|| 复现关注点 | 训练 loss、scheduler、teacher-student、LoRA rank | 权重转换、custom module、scheduler config、runtime 加速 |
243|
244|---
245|
246|## 6. 跟咱们相关的落地方案
247|
248|如果咱们要复现或内部训练一个 Qwen Image Edit 4-step LoRA，建议按这个顺序，不要试图一步到位：
249|
250|### 方案 1：最小可行复现
251|
252|1. 选 Qwen/Qwen-Image-Edit-2511。
253|2. 做 2k-5k prompt/edit cases 的 teacher output cache。
254|3. LoRA rank=128，训 attention + FFN projection。
255|4. 先训 8-step regression。
256|5. 评估稳定后再 8→4 progressive。
257|6. 只用 BF16，不碰 FP8。
258|
259|成功标准：同 seed 下 4-step 输出编辑方向正确，文字/主体不大崩，速度达到 base 40-step 的 8-10x。
260|
261|### 方案 2：生产部署版
262|
263|在方案 1 成功后：
264|
265|1. LoRA fuse 到 transformer。
266|2. 做 scaled FP8 quantization，而不是直接 downcast。
267|3. 用 BF16 teacher 对 FP8 student 再蒸馏/校准一轮。
268|4. 用 LightX2V / Nunchaku / torch.compile / FA3 做推理优化。
269|5. 单独维护 BF16-LoRA、FP8-fused、FP8-split 三种产物。
270|
271|### 方案 3：FlashPack 路线
272|
273|如果目标是“像 Qwen Image Flash 那样部署快”，训练不是主矛盾：
274|
275|1. 先拿已有 4-step Lightning LoRA 或内部 LoRA。
276|2. fuse 权重。
277|3. 转 custom packed runtime 格式。
278|4. 改 diffusers pipeline class / loader。
279|5. 校准 scheduler_config 和 VAE/text_encoder/transformer wrapper。
280|
281|---
282|
283|## 7. 补充：推理期约定也必须蒸馏对齐
284|
285|ModelTC 官方推理脚本里还有几个容易漏的约定，训练 teacher cache 时要一起固定：
286|
287|| 约定 | 2511 / 2509 Edit | 影响 |
288||---|---|---|
289|| true_cfg_scale | LoRA 默认 1.0；base 默认 4.0 | student 学的是“免高 CFG/单分支”行为；teacher target 应保留高 CFG 效果。 |
290|| negative_prompt | 非 2512 默认单空格 `" "` | 训练时别突然加长负面词，否则推理分布不一致。 |
291|| pipeline | QwenImageEditPlusPipeline | 多图输入和 single image edit 的 condition packing 要保持一致。 |
292|| steps | 有 LoRA 时脚本默认 8，4-step 需显式传 steps=4 且配 4-step checkpoint | checkpoint 与 steps 不匹配会直接掉质量。 |
293|| scheduler | FlowMatch Euler + base/max shift=log(3) + top-level shift=1.0 + dynamic shifting | teacher/student/sampling grid 必须一致。 |
294|
295|CFG 蒸馏的因果链要写清：teacher 可以用原版推荐 CFG/true_cfg 生成高质量 target，student 训练成 `true_cfg_scale≈1.0` 也能逼近 teacher 的 guided output，从而推理时少一次负分支/引导开销。这里不是“推理 CFG 随便设低”，而是把引导效果内化进 student vector field。
296|
297|---
298|
299|## 8. 我的判断
300|
301|1. 公开世界里，Qwen Image Edit Lightning 的“训练办法”没有完整开源；lightx2v 只公开了推理、模型和一些关键超参。想 1:1 复现，需要自己补蒸馏训练工程。
302|2. 最关键不是 LoRA 代码，而是 teacher-student trajectory distillation。普通 dreambooth / kohya / diffusers LoRA trainer 训不出来 4-step 能力。
303|3. Qwen 的 scheduler 配置是核心：FlowMatch Euler + `base_shift=max_shift=log(3)` + `shift=1.0` + dynamic shifting + true_cfg_scale≈1.0。训练/推理不一致基本必炸。
304|4. FP8 是第二阶段部署问题，不要先搞。README 已经说明 FP8 base 和 LoRA 分布不匹配会出 grid artifact；公开 LoRA 是 BF16/FP32，FP8 是 fused/quantized artifact。
305|5. Qwen Image Flash/FlashPack 更像工程部署栈，不等价于 Lightning LoRA 训练；它把组件替换为 FlashPack custom classes 和 model.flashpack 格式。
306|
307|---
308|
309|## 多模型交叉审核合并（2026-07-08）
310|
311|### 来自 GLM-5.2
312|
313|- FP8 应拆成训练产物与部署产物两层：4-step LoRA 是 BF16/FP32；FP8 是 fuse/quantize 后的部署优化，不应让复现者误以为训练阶段必须先量化。
314|- `true_cfg_scale=1.0` 的关键是 CFG distillation：teacher 保留高 CFG target，student 学成低/无 CFG 单分支推理，省掉额外 forward。
315|- 高 rank LoRA 是工程推断，不是官方事实；建议做 rank 128 LoRA 与 LoRA+部分层 full fine-tune 的消融。
316|- 若用 diffsynth 做训练，需要手写 distillation loop / scheduler override / teacher-student forward 对齐；更稳妥是先用 diffusers 训练、diffsynth 做推理 benchmark。
317|- 8→4 progressive 不要和对抗损失同时上；先让 4-step regression 稳定，再叠加 DMD/LADD/对抗项。
318|
319|### 来自 Gemini 3.1 Pro
320|
321|- 4-step 下 Euler 截断误差是主风险，timestep 采样不要只做 uniform；应考虑中段加权/trajectory consistency。
322|- CFG 内化会牺牲 prompt adherence，建议加 Qwen-VL/CLIP 类 image-text/edit consistency reward 或冻结 teacher 表征作锚。
323|- FlashPack 的速度收益可能来自 graph fusion、减少 Python 调度开销、kernel launch 优化和自定义 attention/packing；这是部署推断，不是已验证训练事实。
324|- 极少步轨迹会带来 activation outlier，FP8 PTQ 容易崩；生产路线要做 scaled FP8、QAT 或至少校准集量化。
325|
326|### 来自 Claude Opus 4.8
327|
328|- 原文把 scheduler 写成 `shift≈log3` 不够严谨；官方脚本是 `base_shift=max_shift=log(3)`，顶层 `shift=1.0`，且 `use_dynamic_shifting=True`。
329|- FP8 归属需修正：公开 LoRA 是 BF16/FP32；FP8 是 fused/quantized 部署权重。
330|- “高 rank”没有官方证据，必须标注为工程推断。
331|- 推理期约定需要写入训练 recipe：`true_cfg_scale=1.0`、非 2512 负提示为空格、EditPlus 多图输入、steps 与 checkpoint 匹配。
332|- Claude 未验证到 `blanchon/Qwen-Image-Edit-2511-FlashPack`，但本条目实际引用的是已通过 HF API 验证的 `blanchon/Qwen-Image-Edit-2509-FlashPack`，因此保留 2509 FlashPack 证据，不扩展到 2511。
333|
334|---
335|
336|## 参考资料
421|
422|- SDXL-Lightning: Progressive Adversarial Diffusion Distillation, arXiv:2402.13929 — https://arxiv.org/abs/2402.13929
423|- Latent Adversarial Diffusion Distillation, arXiv:2403.12015 — https://arxiv.org/abs/2403.12015
424|- Improved Distribution Matching Distillation for Fast Image Synthesis, arXiv:2405.14867 — https://arxiv.org/abs/2405.14867
425|- Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers, arXiv:2511.16156 — https://arxiv.org/abs/2511.16156
426|- lightx2v/Qwen-Image-Edit-2511-Lightning — https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning
427|- lightx2v/Qwen-Image-Lightning — https://huggingface.co/lightx2v/Qwen-Image-Lightning
428|- ModelTC/Qwen-Image-Lightning — https://github.com/ModelTC/Qwen-Image-Lightning
429|- ModelTC/LightX2V Qwen Image examples — https://github.com/ModelTC/LightX2V/blob/main/examples/qwen_image/README.md
430|- blanchon/Qwen-Image-Edit-2509-FlashPack — https://huggingface.co/blanchon/Qwen-Image-Edit-2509-FlashPack
431|- OPPO-Mente-Lab/Qwen-Image-Pruning — https://github.com/OPPO-Mente-Lab/Qwen-Image-Pruning
432|