# Adam vs AdamW vs Muon：优化器三代的本质差异 & Weight Decay 与 L2 正则化的等价性

> 记录日期：2026-06-25 / 触发人：李承燕
> 问题：adam和adamw差别在哪，为什么大家更喜欢adamw？muon改进在哪里？为什么weight decay不随梯度更新放缩会效果更好？weight decay和l2正则化区别是什么，它们为什么等价？

---

## 一句话总结

Adam 把 weight decay 和梯度自适应耦合在一起，导致正则化强度被学习率缩放污染；AdamW 把 weight decay **解耦**出来单独做，让正则化和优化各司其职，泛化显著更好。Muon 更进一步——对矩阵参数做 Newton-Schulz 正交化替代逐元素自适应，让更新方向"光谱展平"，能用更大的学习率且收敛更快。**Weight Decay 和 L2 正则化在 SGD 下等价，在 Adam 等自适应优化器下不等价。**

---

## 核心概念定义

### 1. Adam
- 逐元素自适应学习率：一阶矩（动量 m）+ 二阶矩（梯度平方的指数移动平均 v）
- 更新公式（简化）：`θ_t = θ_{t-1} - η * m̂_t / (√v̂_t + ε)`
- 通常实现中直接在 loss 上加 L2 正则化项：`loss = original_loss + λ/2 * ||θ||²`
- 这意味着正则化梯度 `λθ` 也进入了 v 的二阶矩估计，被自适应缩放

### 2. AdamW（Loshchilov & Hutter, ICLR 2019）
- 与 Adam 的唯一区别：**weight decay 解耦**
- 更新公式拆成两步：
  1. `θ_t = θ_{t-1} - η * m̂_t / (√v̂_t + ε)`  ← 梯度更新（不含正则化）
  2. `θ_t = θ_t * (1 - ηλ)`                    ← 独立的 weight decay
- **关键洞察**：weight decay 不经过 m/v 的自适应缩放，直接对参数做乘法衰减
- 参考：arXiv:1711.05101

### 3. Muon（Keller Jordan, Dec 2025）
- 全称：**M**oment**u**m **O**rthogonalized by **N**ewton-Schulz
- 只用于 2D 矩阵参数（hidden layer weights）
- 核心操作：
  1. 对梯度做 momentum（Nesterov 风格）
  2. 用 Newton-Schulz 迭代把 momentum 矩阵**近似正交化**（让奇异值都逼近 1）
  3. 把正交化后的方向作为更新
- 等价于：`Update = NewtonSchulz5(momentum_matrix)` 然后 `θ -= η * Update`
- Embedding 层、head 层、scalar/vector 参数仍用 AdamW
- 参考：kellerjordan.github.io/posts/muon/，GitHub: KellerJordan/Muon

### 4. Weight Decay（原始定义）
- 直接在参数上做乘法衰减：`θ = θ * (1 - ηλ)`
- 等价于每一步把参数往原点拉近一点
- 最早来自 Hanson & Pratt (1988)

### 5. L2 正则化
- 在 loss 上加惩罚项：`L = L_original + (λ/2) * ||θ||²`
- 梯度变成：`∇L = ∇L_original + λθ`
- 对 SGD 来说，代入更新公式后等价于 weight decay

---

## 关键区别对比

### Adam vs AdamW

| 维度 | Adam (L2 reg) | AdamW (decoupled) |
|------|--------------|-------------------|
| 正则化方式 | loss 加 λ‖θ‖²，梯度含 λθ | 参数直接乘 (1-ηλ) |
| λθ 是否进二阶矩 v | ✅ 进，被 1/√v̂ 缩放 | ❌ 不进 |
| 大梯度方向的参数 | 正则化被 √v̂ 大幅削弱 | 均匀衰减 |
| 小梯度方向的参数 | 正则化被 √v̂ 放大 | 均匀衰减 |
| 实际正则化强度 | 每个参数不同，与梯度历史耦合 | 所有参数一致 |
| 泛化表现 | 弱于 SGD+Momentum（图像任务） | 追上甚至超过 SGD+Momentum |
| 超参解耦 | λ 和 η 纠缠 | λ 和 η 独立可调 |

**为什么大家更喜欢 AdamW？** 一句话：解耦后 weight decay 的强度跟学习率调度脱钩了。你用 cosine schedule 降 lr 的时候，正则化强度不受影响。Adam 下 weight decay 的实际效果随训练阶段变化剧烈，调参极其痛苦。

### AdamW vs Muon

| 维度 | AdamW | Muon |
|------|-------|------|
| 参数粒度 | 逐元素（每个标量独立自适应） | 逐矩阵（整体正交化） |
| 更新方向 | 梯度方向被 √v̂ 缩放 | 梯度方向被 Newton-Schulz 展平为半正交 |
| 条件数 | 更新矩阵可能高条件数（几乎低秩） | 更新矩阵条件数=1（所有奇异值→1） |
| 学习率容忍度 | 中等，lr 大了震荡 | 极高，lr 可以比 AdamW 大很多 |
| 收敛速度 | 基准 | NanoGPT speedrun 比 AdamW 快 35% |
| 额外开销 | 无 | 每步 ~5 次矩阵乘法（对于 d_model=768 的 NanoGPT <0.1% FLOP overhead） |
| 缩放问题 | 大模型下稳定 | 大模型下优势缩小（需配合 Hyperball/weight norm control） |
| 适用范围 | 所有参数 | 仅 2D hidden layer 矩阵（其余用 AdamW） |
| 理论支持 | ℓ∞ 范数约束隐式偏置（arXiv:2404.04454） | 光谱展平 → 提升稀有方向学习（Spectral Flattening，arXiv:2605.13079） |

### Weight Decay vs L2 正则化

| 维度 | Weight Decay | L2 正则化 |
|------|-------------|----------|
| 作用位置 | 参数更新步骤 | Loss 函数 |
| 公式 | θ *= (1-ηλ) | L += λ/2‖θ‖² |
| 在 SGD 下 | 等价于 L2 reg（因为 SGD 更新是线性的） | 等价于 weight decay |
| 在 Adam 下 | **不等价**：λθ 不进 m/v | 等价：λθ 进 m/v 被自适应缩放 |
| 解耦性 | 天然与优化器解耦 | 与优化器耦合 |

### 等价性证明（SGD 下）

SGD 更新：
```
θ_{t+1} = θ_t - η * ∇L_original(θ_t)
```

加 L2 正则化：
```
θ_{t+1} = θ_t - η * (∇L_original(θ_t) + λθ_t)
       = (1 - ηλ) * θ_t - η * ∇L_original(θ_t)
```

Weight decay 直接：
```
θ_{t+1} = (1 - ηλ) * θ_t - η * ∇L_original(θ_t)
```

**两者完全一致**。这就是为什么老论文里常把它们混着说——在 SGD 时代它们确实等价。

但在 **Adam** 下不等价，因为 L2 reg 的 `λθ` 项会经过 `1/√v̂` 缩放，而 weight decay 直接乘 `(1-ηλ)`，不经过任何自适应。

---

## 技术深度分析

### 1. AdamW 为什么比 Adam 好：ℓ∞ 范数约束视角

arXiv:2404.04454 (2024) 给出了理论解释：

- Adam with L2 reg 优化的是 **ℓ2 约束**问题（权重被均匀拉向原点）
- AdamW (decoupled) 优化的隐式目标更接近 **ℓ∞ 范数约束**：每个参数坐标独立地被约束，允许某些坐标远大于其他坐标
- 这更符合神经网络的稀疏性结构：某些神经元/方向应该大，某些应该小。ℓ∞ 约束允许这种非均匀性，ℓ2 则强制均匀惩罚
- 实验验证：AdamW 训练出的权重矩阵比 Adam+L2 有更大的 ℓ∞/ℓ2 比值（更"尖"的分布）

### 2. 为什么 decoupled weight decay 效果更好？（不留梯度路径）

核心原因有三个：

**a) 正则化强度与梯度历史解耦**

Adam 的二阶矩 v 是梯度平方的 EMA。如果某个参数历史上梯度很大，v 就大，`λθ/√v̂` 就小——这个参数的 weight decay 就被"稀释"了。decoupled 方式不经过这个缩放，所有参数统一衰减。

**b) 学习率调度不影响正则化**

训练后期 lr 降到接近 0 时，Adam+L2 的正则化项 `η*λθ/√v̂` 也趋近于 0——等于后期没有正则化。AdamW 的 `(1-ηλ)` 中 η 乘以 λ 后仍保持恒定相对衰减率（实践中 λ 通常设为与 lr 独立的常数，如 0.01-0.1）。

arXiv:2512.08217 甚至挑战了 "λ ∝ η" 的传统设定，提出 λ 应该 ∝ η²（基于稳态正交性论证）或独立设置。

**c) "径向拉锯战" (Radial Tug-of-War)**

arXiv:2602.05136 指出：梯度倾向于增大参数范数（扩展有效容量），weight decay 倾向于抑制范数。在 Adam 中这两股力共享同一个 √v̂ 缩放通道，产生径向振荡，污染二阶矩估计。解耦后两股力独立作用，振荡消失。

### 3. Muon 的改进：从逐元素到矩阵级

Muon 的核心创新是**放弃逐元素自适应，转而对整个矩阵做光谱归一化**。

**为什么有效？**

a) **高条件数问题**：SGD/Adam 对 transformer hidden layer 产生的更新矩阵通常条件数极高（几乎低秩）——大部分更新集中在少数几个方向，大量"稀有方向"更新微弱。Newton-Schulz 迭代把奇异值全部压到 1，等效于**放大稀有方向的更新幅度**。

b) **学习率容忍度**：正交化后的更新矩阵的 spectral norm = 1，所以更新步长有上界保证。arXiv:2605.13079 证明 Muon 的最大稳定步长与最小奇异值成比例（而非 Adam 的最大奇异值），因此可以承受大得多的 lr。

c) **与 Shampoo 的关系**：Muon 可以理解为"瞬时 Shampoo"——去掉了 Shampoo 的 preconditioner 累积，直接用 Newton-Schulz 替代逆四次方根。Bernstein & Newhouse (2024) 最早指出 NS 迭代可作为 Shampoo 的高效替代。

**缩放问题与最新进展：**

Muon 在 1.5B 以下模型上显著优于 AdamW，但到更大规模优势缩小。原因：大模型训练中 weight decay 对 Muon 的 norm 控制不够好，权重矩阵的 spectral norm 会漂移。

近期解决方案：
- **Hyperball** (arXiv:2606.16899, Jun 2026)：强制 Frobenius norm 恒定，替代 weight decay，1.2B 模型上 20-30% token 等效加速
- **Muown** (arXiv:2605.10797, May 2026)：把 row-magnitude 作为独立优化变量，用 ℓ∞ 几何更新，解决 norm drift
- **Muon²** (arXiv:2604.09967)：加 adaptive second-moment preconditioning
- **OrScale** (arXiv:2605.07815)：layer-wise trust-ratio scaling

### 4. 实现对比（PyTorch 伪代码）

```python
# === Adam (L2 regularization) ===
loss = criterion(output, target) + 0.5 * weight_decay * sum(p.norm()**2 for p in model.parameters())
loss.backward()
for p in model.parameters():
    m = beta1 * m + (1-beta1) * p.grad
    v = beta2 * v + (1-beta2) * p.grad**2
    p.data -= lr * m_hat / (v_hat.sqrt() + eps)
    # L2 reg 已经包含在 p.grad 里了 (自动加了 weight_decay * p)

# === AdamW (decoupled) ===
loss = criterion(output, target)   # 不加 L2 正则化
loss.backward()
for p in model.parameters():
    m = beta1 * m + (1-beta1) * p.grad
    v = beta2 * v + (1-beta2) * p.grad**2
    p.data -= lr * m_hat / (v_hat.sqrt() + eps)
    p.data *= (1 - lr * weight_decay)   # ← 解耦！不经过 m/v

# === Muon (核心逻辑) ===
def muon_update(p, grad, momentum, lr, beta=0.95, ns_steps=5):
    # Nesterov momentum
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta)
    # Newton-Schulz 正交化
    update = newtonschulz5(update, steps=ns_steps)
    # 应用更新（Muon 本身不做 weight decay）
    p.data -= lr * update

def newtonschulz5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + 1e-7)
    if X.size(0) > X.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a*X + (b*A + c*A@A) @ X
    if G.size(0) > G.size(1): X = X.T
    return X
```

---

## 实践建议

### 什么场景用什么？

| 场景 | 推荐 | 理由 |
|------|------|------|
| 通用训练（分类/检测/分割） | **AdamW** | 成熟稳定，超参好调，社区默认 |
| LLM 预训练（<1.5B） | **Muon + AdamW（混用）** | 35% 加速，hidden layer 用 Muon，embed/head 用 AdamW |
| LLM 预训练（>1.5B） | **Muon+Hyperball 或纯 AdamW** | Muon 原生 weight decay 缩放有问题，需额外 norm 控制 |
| 微调/RL/RLVR | **AdamW** | Muon 在 RL 场景有光谱失效问题（arXiv:2605.19282） |
| 图像生成模型训练 | **AdamW**（可试 Muon） | 2D conv 可 flatten 后用 Muon，但经验有限 |

### 超参调优速查

- **AdamW**：`lr=1e-4~3e-4`，`weight_decay=0.01~0.1`，`betas=(0.9, 0.999)`
- **Muon**：`lr` 可比 AdamW 大 2-5x，`momentum=0.95`，weight decay 要更小心（易出 norm drift）
- **Weight Decay 的值**：通常在 `[0.01, 0.1]`，大模型偏小、小模型偏大。不要设 `1e-4` 量级——那几乎等于没开正则化

### 常见踩坑

1. **PyTorch 的 `AdamW(weight_decay=...)` 已正确解耦**，但早期 `Adam(weight_decay=...)` 实际上是 L2 reg
2. **Muon 的 weight decay 坑**：论文已指出无 weight decay 时 Muon 的 spectral norm 会漂移（上涨）。建议设 weight_decay 或用 norm control
3. **不要把 Muon 用在 embedding 和 head 层**——Keller Jordan 明确说这俩层用 AdamW 更好
4. **NS 迭代用 bfloat16 做**——float32 反而慢且无增益

---

## 参考资料

### 核心论文
- [AdamW: Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter, ICLR 2019
- [Implicit Bias of AdamW: ℓ∞ Norm Constrained Optimization](https://arxiv.org/abs/2404.04454) — 2024，解释 AdamW 优于 Adam 的理论
- [Decoupled Orthogonal Dynamics](https://arxiv.org/abs/2602.05136) — 2026，Radial Tug-of-War 分析
- [Correction of Decoupled Weight Decay](https://arxiv.org/abs/2512.08217) — 2025，挑战 λ ∝ η 传统设定

### Muon 相关
- [Muon: An optimizer for hidden layers](https://kellerjordan.github.io/posts/muon/) — Keller Jordan 官方博客，Muon 的原始设计文档
- [Muon GitHub](https://github.com/KellerJordan/Muon) — 官方实现
- [Modded-NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt) — 用 Muon 创 NanoGPT 训练速度记录
- [Fantastic Pretraining Optimizers II: Hyperball](https://arxiv.org/abs/2606.16899) — 2026，hyperball norm control
- [Muown: Row-Norm Control for Muon](https://arxiv.org/abs/2605.10797) — 2026，row-magnitude control
- [Spectral Flattening Is All Muon Needs](https://arxiv.org/abs/2605.13079) — 2026，理论解释
- [Spectral Scaling Laws of Muon](https://arxiv.org/abs/2606.04058) — 2026，Muon 在低精度下的 SVD 行为
- [Navigating LLM Valley](https://arxiv.org/abs/2605.09176) — 2026，全面综述（AdamW→矩阵优化器演进）

### 中文来源
- 知乎搜索：需要登录/CAPTCHA 验证，此次未获取到内容
- 小红书搜索：无有效搜索结果

### 搜索时间
- 2026-06-25，全部来源在本次会话中搜索


---

## 🔍 多模型交叉审核（2026-06-27）

以下为三模型并行审核的补充意见：

### 来自 GLM-5.2 — 工程落地 & 国产生态

用户来说，如果用 diffusers +         
     accelerate 做训练（用户的主力框架），目前没有现成的 Muon                   
     集成，需要自己写 optimizer 类并处理 accelerate 的分布式 dispatch           
     逻辑，工程量不小。草稿应补一句"diffusers/accelerate 生态暂无 Muon          
     支持，需自定义实现"。                                                      
                                                                                
     5. 中文社区对 Muon                                                         
     的讨论实际情况不应跳过。草稿中文来源部分写"知乎搜索需要登录/CAPTCHA，      
     未获取到内容"就放弃了，但中文社区确实有实质讨论可引用：苏剑林（苏神）      
     在科学空间博客（kexue.fm）讨论过矩阵正交化在优化器中的应用；智源社区和     
      PaperWeekly 都有 Muon 的中文解读文章，关注焦点集中在"Muon                 
     能否用于百亿级模型训练"和"与 Shampoo 的工程对比"上，与英文社区关注         
     speedrun 记录不同。更实际的是，中文社区有关于 Muon 在华为 Ascend NPU       
     上适配性的讨论——Newton-Schulz 的小矩阵 batched matmul 在 Ascend 的         
     Cube 算子上性能不如 NVIDIA GPU 的 Tensor                                   
     Core（主要因为矩阵维度太小，无法充分利用 Cube 的 16x16                     
     分块），这是国产芯片上跑 Muon                                              
     的实际工程瓶颈。草稿应补充这个事实，因为对用户的国产模型训练场景直接相     
     关。                                                                       
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184958_77e212

Session:        20260627_184958_77e212
Duration:       49s
Messages:       2 (1 user, 0 tool calls)


### 来自 Gemini 3.1 Pro — 多模态交叉 & Google 生态

ayer 无脑全开         
     Muon，会导致视觉特征灾难性遗忘或过拟合。必须配合 Modality-specific         
     Learning Rate Multipliers（模态专属学习率缩放），把被 Muon                 
     展平的尺度按比例手动拉回来。                                               
     3. DPO/GRPO 对齐阶段的正则化冲突（全流程 SFT/RL 维度）                     
     草稿在最后提了一句“微调/RL 用 AdamW”，但没解释对齐阶段 weight decay        
     的特殊性。                                                                 
     - 技术事实：在 DPO 或 GRPO 阶段，Loss 中包含的 Reference Model KL          
     散度惩罚，本质上已经充当了极强的动态流形正则化（Manifold                   
     Regularization），把参数强行拉向预训练的隐空间。                           
     - 补充点：如果在 RL/对齐阶段还保持预训练级别的 weight decay（如            
     0.05~0.1），AdamW 的 L2 惩罚（向原点拉）会与 KL 惩罚（向 Ref Model         
     拉）产生严重的向量空间力场冲突。这会导致 Policy                            
     模型在对齐时发生剧烈的“知识遗忘”。因此，实操中进入 DPO/RL                  
     阶段后，AdamW 的 weight decay 通常要关掉（设为 0）或降到 1e-6              
     级别，把约束权完全交还给 KL 项。                                           
                                                                                
     你把这三条补到“对比”或者“踩坑”章节，这份文档的实战含金量就直接拉满了。     
     改完需要我再过一遍随时说。                                                 
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184958_d78782

Session:        20260627_184958_d78782
Duration:       28s
Messages:       2 (1 user, 0 tool calls)


### 来自 Claude Opus 4.6 — 论证严密性 & 结构完备性

用在每个元素上，但       
     Newton-Schulz 正交化只约束矩阵的谱范数（最大奇异值），不约束 Frobenius     
     范数。当模型变大（层数增加、d_model 增大）时，权重矩阵的 Frobenius         
     norm 会随训练步数漂移上涨，即使 spectral norm 保持稳定                     
     - 数值阈值：Keller Jordan 的实验显示，当 d_model > 1024 且训练步数 >       
     10K 时，不加额外 norm control 的 Muon 会出现 loss spike；1.5B              
     模型通常对应 d_model ≈ 2048，此时 Frobenius norm drift 已显著              
     - Hyperball 的解决方式：强制 ||W||_F = const，把 Frobenius norm            
     从优化变量中解耦出来，只优化方向。这让 Muon 在 7B 模型上也能稳定           
                                                                                
     建议在"缩放问题"段落加一句："Muon 原生 weight decay 无法同时控制           
     spectral norm 和 Frobenius norm，当 d_model > 1024 时 Frobenius norm       
     会随步数漂移，导致有效学习率失控"，然后列出 Hyperball/Muown                
     等方案的数学约束差异。                                                     
                                                                                
     总结：补充这五点后，你的条目在论证完备性（等价性代数反证明）、对比结构     
     性（方差维度、衰减率异质性量化）、边界清晰性（Muon                         
     适用参数清单、失效阈值）和因果链条（AdamW                                  
     采纳历史）上会更扎实，读者能直接拿去做工程决策而不用再二次查证。           
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184958_073676

Session:        20260627_184958_073676
Duration:       54s
Messages:       2 (1 user, 0 tool calls)
