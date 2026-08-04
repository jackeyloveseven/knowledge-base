# 训练数据分布控制 & 训推 Prompt 格式一致性

> 日期：2026-06-23
> 触发人：李承燕

## 一句话总结

训练数据分布控制和训推 prompt 格式一致性的本质相同：**模型不是"理解语义"，是"匹配统计模式"——你换分布，它就换行为。**

---

## 一、为什么训练数据要控制分布？

### 核心原因

| 问题 | 根因 | 后果 |
|------|------|------|
| 不控制分布 | 真实数据长尾+头部主导 | 过拟合头部、遗忘长尾、泛化差 |
| 训推 prompt 不一致 | input distribution shift | cross-attention 对齐失败、CFG 漂移、质量骤降 |

### 三大动机

1. **避免过拟合与模式崩溃**：不对数据分布控制，模型对头部类别（常见风景、人像）或特定分辨率过拟合，边缘场景生成质量下降甚至模式崩溃。

2. **防止长尾概念的灾难性遗忘**：SFT 阶段模型为快速收敛到高质量流形，稀有实体/特殊风格容易被主流模式掩盖。控制分布保留了这些小众但重要的数据片段。

3. **提升泛化能力与内容多样性**：通过控制物理属性（分辨率、宽高比）和语义特征（风格、类别）的分布均衡，模型在各类生成/编辑任务中展现出更强的鲁棒性。

---

## 二、数据分布控制的五种策略

### 1. 降采样与升采样的重平衡（Distribution Rebalancing）
对各维度（主体类别、场景、动作、分辨率）做频率量化。头部降采样，长尾升采样。构建更公平的视觉世界表示。

### 2. 基于特征聚类的均匀采样（Cluster-based Uniform Sampling）
用 CLIP 提取图像特征 → K-means/图社区检测划分语义聚类 → 每簇均匀抽样。同时实现大规模去重 + 分布多样性保障。

### 3. 知识图谱 + 细粒度语义加权（Knowledge Graph & Semantic Weighting）
Z-Image、Qwen-Image 引入 WordNet 式层级分类组织概念。文本标签映射到知识图谱节点，结合 BM25 检索得分和父子层级关系，动态计算语义级采样权重。

### 4. 闭环主动精选 & 长尾补全（Active Curation & Long-tail Supplementation）
评估模型在长尾概念上的表现 → 标记数据缺口 → 触发跨模态向量检索从海量未清洗数据池定向召回 → 生成新训练对补充训练集。

### 5. 多维度分层采样（Multi-Dimensional Stratified Sampling）
按审美得分、信息密度、清晰度等分层（高/中/低），在各层内均匀抽样。确保模型既学高质量画面，也接触丰富结构/复杂排版的多样化样本。

---

## 三、实操中的两个坑

### 坑1：重采样 ≠ 免费午餐
长尾升采样 = 同一条稀有数据被反复"咀嚼"。没配合足够数据增强（crop/flip/color jitter/文本改写），容易从"长尾遗忘"变"长尾过拟合"——模型背下稀有概念但换角度就不认识。Z-Image 的知识图谱+BM25 加权优于暴力重复，因为它在语义层面做密度调节。

### 坑2：分布粒度要匹配模型容量
小模型做太细粒度的概念平衡有害——参数量不够建模几千个语义簇的差异，强行均匀采样导致每个概念都学个半吊子。分布策略和模型 scale 要联合设计，不是越均匀越好。

---

## 四、为什么训推 Prompt 格式必须一致？

本质是 **分布偏移（Distribution Shift）** 问题：训练时模型学会 `f(P_train_format, noise) → image`，推理时喂 `P_infer_format`，如果结构不同，输入分布就变了。

### 1. Token 级别的条件依赖被打乱

Diffusion 模型通过 cross-attention 把文本 token embedding 注入去噪。训练时模型内部 attention pattern 习惯固定 token 序列结构：

```
[BOS] 主体描述 [SEP] 风格标签 [SEP] 质量词 [EOS]
```

训练用 `"A photo of a cat, oil painting style, high quality"`，推理换成 `"Style: oil painting. Subject: cat. Quality: high."`，语义相似但 token 序列完全不同 → cross-attention 的 Q-K 对齐关系失效。模型学到的是**统计共现模式**，不是真正的语义理解。

### 2. 位置编码的隐式约定被打破

Transformer 的 position embedding 在训练中学会"第5个 token 附近通常是风格信息"这类隐式先验。换格式后同样信息跑到第15个位置，模型当成完全不同的 pattern 处理。

这也是为什么 fine-tuning 阶段改 prompt 模板后，通常需要做**少量的格式对齐训练**（format alignment）——不一定要全量重训，但至少得让模型"见过"新格式。

### 3. CFG 的 null-condition 要配套

CFG 训练时 null condition（空文本/负向 prompt）格式也是固定的。推理时换 prompt 模板 → `ε_uncond` 和 `ε_cond` 差值方向偏移 → CFG scale 没变但出图风格飘了。

### 4. 实战案例

| 模型 | 训练格式 | 推理踩坑 |
|------|----------|----------|
| FireRed Image Edit | `"Reference image + edit instruction → output"` 固定单轮模板 | 改对话格式或多轮形式 → 模型直接懵 |
| Qwen-Image | system prompt + user prompt 结构训练时写死 | 改 system prompt 内容或调整消息顺序 → 效果骤降 |

---

## 五、实践建议

| 场景 | 建议 |
|------|------|
| 设计训练数据分布 | 先跑全量数据的多维频率分析（类别/分辨率/风格/美学分），再决定降/升采样比例 |
| 长尾处理 | 优先用语义加权（KG+BM25）而非暴力重复，配合强数据增强防过拟合 |
| 模型容量小（<1B） | 分布控制粒度要粗——按大类平衡即可，不要上千个语义簇 |
| 改 Prompt 模板 | 推理时必须用与训练完全一致的模板；如果必须改，至少做一轮 format alignment 微调 |
| 调试 CFG 漂移 | 检查 null-condition 文本是否与训练时一致，不一致会导致 guidance 效果偏移 |
| 多轮/对话格式 | 如果训练只有单轮固定模板，推理时不要强行套多轮对话格式 |

---

## 参考资料

- Z-Image: 知识图谱+BM25 采样的数据分布控制方案
- Qwen-Image: system prompt 固定结构的设计原理
- FireRed Image Edit: 单轮编辑模板的格式约束
- CFG (Classifier-Free Guidance) 论文: Ho & Salimans, 2021
- Stable Diffusion 系列的数据处理 pipeline 实践


---

## 🔍 多模型交叉审核（2026-06-27）

以下为三模型并行审核的补充意见：

### 来自 GLM-5.2 — 工程落地 & 国产生态

自动应用，推理时如果手动拼 prompt      
     而不是走                                                                   
     tokenizer.apply_chat_template()，格式就和训练时不一致——这和文章说的"改     
     system prompt                                                              
     顺序效果骤降"是同一类问题的工程化表现，但根因更底层（连模板都没对上）      
     。TensorRT/ONNX 导出时，prompt embedding 预计算逻辑和 diffusers 的         
     runtime embedding 计算也可能有精度差异（fp16 vs                            
     fp32），这层也属于训推一致性的范畴。                                       
                                                                                
     意见5：中英双语 prompt 的语言分布维度缺失                                  
                                                                                
     国产模型训练数据通常是中英混合，语言比例直接影响推理行为。如果训练 80%     
     中文 + 20% 英文，推理用英文 prompt，对 LLM-based encoder 来说              
     embedding 分布本身就是 OOD 的（中英文 token 频率分布差异大，BPE            
     分词边界不同）。中文技术社区讨论很多的"用中文咒语效果更好"现象，本质不     
     是模型理解中文更好，是训练数据里中文占比高、模型匹配的统计模式偏中文。     
     文章应在"训推一致性"章节补一维：语言分布也是分布控制的一维，训推语言不     
     匹配和格式不匹配是同构问题。数据分布控制策略那节也应该加"语言比例"作为     
     一个 rebalancing 维度。                                                    
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184915_04874a

Session:        20260627_184915_04874a
Duration:       1m 12s
Messages:       2 (1 user, 0 tool calls)


### 来自 Gemini 3.1 Pro — 多模态交叉 & Google 生态

oogle Veo/Lumiere 现在通用的做法是：训练时用强 VLM（比如        
     Gemini-Pro-Vision）把训练集全部打上极其冗长、结构固定的 Detail             
     Caption（包含 Subject, Lighting, Composition                               
     等范式）。推理时，用户随便输入几个词，系统先用 LLM（做 System Prompt       
     约束）把用户的短文本 Expand 成和训练集一模一样的超长 Detail Caption。      
     补充建议：第五部分的实践建议可以补一条“架构解法”：用 LLM 充当              
     Distribution matching router，强行在系统层把 User input distribution       
     映射到 Training dataset distribution，而不是指望用户自己写对。             
                                                                                
     4. 物理分布（宽高比/分辨率）对 Noise Schedule 的隐式污染                   
     草稿提到了控制分辨率分布来提升泛化，但没讲透底层的数学机制。               
     当你在训练集中混入长尾的非标准宽高比或多分辨率数据时，实际上改变了前向     
     加噪过程的                                                                 
     SNR（信噪比）。同样步数下，高分辨率/长条图的有效信息量和标准正方形图是     
     不同的。                                                                   
     补充建议：可以提一嘴，如果在数据层面做了分辨率/宽高比的分布扩充（比如      
     引入了长尾的极宽/极高图），对应的 Loss 机制必须配合 Min-SNR                
     Weighting，或者像 Google/业界近期做的那样调整 Resolution-dependent         
     noise schedule，否则这些长尾数据在算 Loss 时会带偏梯度的流形。             
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184915_3f4939

Session:        20260627_184915_3f4939
Duration:       34s
Messages:       2 (1 user, 0 tool calls)


### 来自 Claude Opus 4.6 — 论证严密性 & 结构完备性

 scale * (ε_cond - ε_uncond) 中，ε_uncond         
     是"锚点"，锚点漂移 → 整个引导方向偏                                        
                                                                                
     建议加一句："null-condition 必须与训练时的 tokenizer 输出逐 token          
     一致，不只是语义等价。"                                                    
     5. 多维度分层采样的"审美得分"黑箱                                          
     策略5 提到"按审美得分分层"，但审美得分本身是另一个分布偏移的雷：           
     - 如果审美模型（如 LAION Aesthetics                                        
     Predictor）是在特定数据集（主要是欧美摄影风格）上训练的，用它过滤亚洲      
     插画/水墨画数据 → 引入新的头部偏见                                         
     -                                                                          
     建议补充：审美打分器要么在目标域重新校准，要么只做粗分层（高/中/低），     
     不要用作硬阈值过滤                                                         
                                                                                
     总结：文档的框架和因果链清晰，但缺可操作的量化边界和底层机制的更细一层     
     拆解。三个最该补的点：Distribution Shift                                   
     的度量方法、长尾重采样的安全倍数、Position Encoding                        
     类型对格式鲁棒性的影响。                                                   
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184915_71931a

Session:        20260627_184915_71931a
Duration:       31s
Messages:       2 (1 user, 0 tool calls)
