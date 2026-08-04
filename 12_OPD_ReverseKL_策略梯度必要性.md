# 为什么 OPD 最小化 Reverse KL 必须走 Policy Gradient

> 记录日期：2026-06-25 | 触发人：李承燕
> 触发问题：「为什么 OPD 最小化 Reverse KL 必须走 Policy Gradient 框架，不能像普通知识蒸馏一样直接算 KL 反向传播？我的理解是 OPD 是 on-policy 采样，采样分布和模型参数绑定，直接求 KL 梯度会缺 score function 项、产生偏差，只有策略梯度能给出无偏梯度——这个理解对吗？如果是对的，为什么策略梯度可以给出无偏梯度？」

---

## 一句话总结

**你的理解完全正确。** 核心是：普通 KD 的期望在固定分布（teacher/data）上取，梯度只需穿过被积函数；OPD 的期望在自己参数化的分布 π_θ 上取，梯度必须穿过采样过程本身。标准 autodiff 拿不到「采样分布随参数变化」这项导数，所以直接反传是有偏的。Policy Gradient 通过 log-derivative trick 把采样过程转化为 score function ∇log π_θ，给出了无偏估计。

---

## 1. 从数学上看：为什么直接反传是偏的

### 1.1 普通 KD（Off-Policy）可以直传

普通 KD 的目标（Forward KL）：

```
L_KD = E_{y ~ p_teacher}[ log p_teacher(y) - log p_student(y; θ) ]
```

求导：

```
∇_θ L_KD = - E_{y ~ p_teacher}[ ∇_θ log p_student(y; θ) ]
```

这里的期望分布 p_teacher 与 θ **无关**（teacher 固定），所以：
- 采样 y ~ p_teacher 后，直接对 log p_student 求导即可
- 标准 autodiff 能完美处理：y 已经是具体值，反向传播沿着计算图从 loss 到 θ 一路链式法则下来

**代码级直觉：**

```python
# 普通 KD — 直接 loss.backward() 就行
teacher_logits = teacher(x)  # fixed
student_logits = student(x)  # parameterized by θ
loss = F.kl_div(
    F.log_softmax(student_logits, dim=-1),
    F.softmax(teacher_logits, dim=-1),
    reduction='batchmean',
    log_target=False
)
loss.backward()  # ✓ 梯度正确，因为采样方 (teacher) 和 θ 无关
```

### 1.2 OPD Reverse KL — 采样方随 θ 变

OPD 的目标（Reverse KL）：

```
J(θ) = KL(π_θ || q) = E_{y ~ π_θ}[ log π_θ(y) - log q(y) ]
```

展开求导：

```
∇_θ J = ∇_θ Σ_y π_θ(y) [ log π_θ(y) - log q(y) ]

     = Σ_y [ ∇_θ π_θ(y) · (log π_θ - log q) + π_θ(y) · ∇_θ log π_θ(y) ]
      \_____________________/   \___________________________________/
         分布变化项 (缺!)              被积函数变化项 (autodiff 能拿到)

第一项用 log-derivative trick 改写：
    ∇_θ π_θ(y) = π_θ(y) · ∇_θ log π_θ(y)

第二项：
    Σ_y π_θ(y) · ∇_θ log π_θ(y) = Σ_y ∇_θ π_θ(y) = ∇_θ 1 = 0

最终：
    ∇_θ J = E_{y ~ π_θ}[ (log π_θ(y) - log q(y)) · ∇_θ log π_θ(y) ]
```

**关键：第一项（分布变化项）在标准 autodiff 中消失了。**

如果直接写 `loss.backward()`：

```python
# ❌ 这样做是偏的
y = student.sample(x)  # 采样时梯度断了！
student_logps = student.log_prob(x, y)
teacher_logps = teacher.log_prob(x, y).detach()
loss = (student_logps - teacher_logps).mean()
loss.backward()
# 问题：y 是通过 sampling 得到的，autodiff 链条在 .sample() 处断裂
# 你只拿到了「改变 student_logps 对 loss 的影响」
# 没拿到「改变采样分布导致 y 变成不同 token 对 loss 的影响」
```

**本质：Stochastic Computation Graph 问题（Schulman et al., NIPS 2015）。**

在一个计算图中，如果随机节点的采样分布依赖于参数 θ，那么梯度有两部分：
1. **Pathwise derivative**（路径导数）：穿过随机节点内部的确定性计算 → 需要 reparameterization
2. **Score function derivative**（得分函数导数）：穿过采样分布的参数 → 需要 ∇log p_θ

离散采样（categorical/multinomial）**没有可微的 reparameterization**（无法写成 y = g_θ(ε), ε 独立于 θ），所以只能走 score function。

---

## 2. 直观对比表

| 维度 | 普通 KD（Forward KL） | OPD（Reverse KL） |
|------|----------------------|-------------------|
| **目标** | E_{y~p_T}[log p_T - log p_S] | E_{y~p_S}[log p_S - log p_T] |
| **采样方** | Teacher（固定，不参与梯度） | Student（随 θ 变化，必须参与梯度） |
| **梯度能否直传** | ✓ loss.backward() 即可 | ✗ 采样操作断梯度 |
| **梯度形式** | -E[∇log p_S] | E[(log p_S - log p_T) · ∇log p_S] |
| **score function 项** | 不需要 | **必须有**，这是分布变化项 |
| **方差** | 低 | 高（score function 估计的固有属性） |

---

## 3. 为什么 Policy Gradient 是无偏的

### 3.1 Log-Derivative Trick（REINFORCE 的核心）

对任意期望 E_{x~p_θ}[f(x)]：

```
∇_θ E_{x~p_θ}[f(x)] = ∇_θ ∫ f(x) p_θ(x) dx
                     = ∫ f(x) ∇_θ p_θ(x) dx           (Leibniz，f 与 θ 无关时)
                     = ∫ f(x) p_θ(x) · ∇_θ log p_θ(x) dx   (因为 ∇p = p · ∇log p)
                     = E_{x~p_θ}[ f(x) · ∇_θ log p_θ(x) ]
```

**无偏性证明**：蒙特卡洛估计

```
ĝ = (1/N) Σ_i f(x_i) · ∇_θ log p_θ(x_i),  x_i ~ p_θ

E[ĝ] = E_{x~p_θ}[ f(x) · ∇_θ log p_θ(x) ]
     = ∫ f(x) p_θ(x) · ∇_θ log p_θ(x) dx
     = ∫ f(x) ∇_θ p_θ(x) dx
     = ∇_θ ∫ f(x) p_θ(x) dx
     = ∇_θ E_{x~p_θ}[f(x)]
```

**每一步都是精确等式，所以估计是无偏的。**

### 3.2 为什么无偏 ≠ 低方差

REINFORCE 的无偏性来自恒等式本身。但它方差高，因为：
- 只用 f(x_i) 这一个标量来加权 ∇log p_θ
- f(x) 的值域可能很大 → 单个样本的梯度估计波动剧烈
- 减 baseline 能降方差但保持无偏：`(f(x) - b) · ∇log p_θ`

### 3.3 MiniLLM 的 Policy Gradient 形式

MiniLLM 把 sequence-level Reverse KL 推导成了 token-level Policy Gradient：

```
∇J(θ) = -E_{y~π_θ}[ Σ_t (R_t - 1) · ∇log π_θ(y_t | y_<t, x) ]

其中 R_t = Σ_{t'=t}^T log( q(y_{t'} | y_<t') / π_θ(y_{t'} | y_<t') )
```

这是标准的 REINFORCE with baseline（baseline=1），加上 causal return-to-go。

**Token-level 简化**（丢掉未来项，更高偏差但更低方差）：

```
∇J_tok(θ) = -E_{y~π_θ}[ Σ_t r_t · ∇log π_θ(y_t | y_<t, x) ]

其中 r_t = log q(y_t | y_<t) - log π_θ(y_t | y_<t)
```

---

## 4. 代码级对比：直传 vs Policy Gradient

### 4.1 直传（有偏，OPD 不能这样做）

```python
# 这段代码在 OPD 场景下梯度是偏的
def biased_opd_step(student, teacher, prompt):
    # Student 自己 rollout
    tokens = student.generate(prompt)  # 采样断梯度！
    
    # 在每个 prefix 上算 KL
    loss = 0
    for t in range(len(tokens)):
        prefix = tokens[:t]
        s_logits = student(prefix)  # [vocab_size]
        t_logits = teacher(prefix)  # [vocab_size]
        
        # Full-vocab reverse KL
        s_probs = F.softmax(s_logits, dim=-1)
        t_probs = F.softmax(t_logits, dim=-1)
        loss += (s_probs * (s_probs.log() - t_probs.log())).sum()
    
    loss.backward()  # ← 偏了！采样分布变化项没进来
    # 这个梯度等价于「固定采样结果后，只优化该结果的概率」
    # 丢失了「采样到不同 token 会导致不同 loss」的信息
```

### 4.2 Policy Gradient（无偏，但方差大）

```python
# 这段代码给出无偏梯度
def unbiased_opd_step(student, teacher, prompt):
    # Student rollout — 记录采样时的 log_prob
    tokens = []
    log_probs = []
    for t in range(max_len):
        logits = student(tokens)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        token = dist.sample()
        tokens.append(token)
        log_probs.append(dist.log_prob(token))  # ← 每个 token 的 log π_θ
    
    # 算每个位置的 "reward": r_t = log q - log π
    rewards = []
    for t in range(len(tokens)):
        with torch.no_grad():
            t_logp = teacher.log_prob(tokens[:t+1])
            s_logp = log_probs[t]
            rewards.append(t_logp - s_logp)
    
    # REINFORCE: loss = - Σ r_t · log π_θ(y_t)
    # 这里的 log π_θ 是计算图内的 — 梯度会通过它流
    policy_loss = 0
    for t, (r, lp) in enumerate(zip(rewards, log_probs)):
        # r 是 scalar multiplier (no grad) × lp (has grad)
        policy_loss += -r.detach() * lp  
        #        r 被 detach 当常量，只对 lp 求导
        #        等价于 ∇_θ [r · (-log π_θ)]
    
    policy_loss.backward()
    # ✓ 无偏！虽然方差大
```

### 4.3 为什么 Policy Gradient 版本无偏：一个具体例子

假设 vocab_size=2（token A 和 B），当前 π_θ(A)=0.7, π_θ(B)=0.3。
期望梯度：

```
∇ J = 0.7 × (log π(A)-log q(A)) × ∇log π(A)
    + 0.3 × (log π(B)-log q(B)) × ∇log π(B)
```

Policy Gradient 采样一次，比如采到 B。此时的估计是：

```
ĝ = (log π(B) - log q(B)) × ∇log π(B)
```

对多次采样的期望：E[ĝ] = 0.3 × (log π(B)-log q(B)) × ∇log π(B) + 0.7 × (log π(A)-log q(A)) × ∇log π(A) = ∇ J ✓

而直传版本，采到 B 后固定 y=B 对 (log π(B) - log q(B)) 求导，得到的是：

```
∇_θ (log π_θ(B) - log q(B)) when y is fixed to B
= ∇_θ log π_θ(B)
```

但这不等于正确的梯度 0.3 × (log π(B)-log q(B)) × ∇log π(B) —— 缺少了 π_θ(A) 的分支，也缺少了分布权重 0.3。

---

## 5. 什么时候可以绕过 Policy Gradient

### 5.1 连续空间 + Reparameterization

对于连续分布（如 Gaussian），如果可以用 reparameterization trick：

```python
# y = μ_θ + σ_θ · ε,  ε ~ N(0,1)
# 那么可以直接反传，因为梯度穿过 y 是连续的
ε = torch.randn_like(μ)
y = μ + σ * ε  # y 的计算图不断
loss = kl_divergence(y, teacher)
loss.backward()  # ✓ 对连续 latent 可行
```

这就是 DiffusionOPD (arXiv:2605.15055) 能在连续 latent 空间直接反传 KL 的原因！Diffusion 的 denoising step 是 Gaussian transition:

```
x_{t-1} ~ N(μ_θ(x_t, t), σ²)
```

用 reparameterization 后，KL(q_θ || p) 的梯度可以直传——因为 random variable 是连续的且可重参数化。

### 5.2 Gumbel-Softmax（离散空间的近似）

```python
# 用 Gumbel-Softmax 做连续松弛，提供偏的但低方差的梯度
logits = student(prefix)
y_soft = F.gumbel_softmax(logits, tau=1.0, hard=False)
# y_soft 是连续近似，梯度可以穿过
# 但和前向采样有 gap，所以估计有偏
```

### 5.3 哪些方法用了 Policy Gradient vs Direct Loss

| 方法 | 梯度方式 | 为什么 |
|------|---------|--------|
| **MiniLLM** | Policy Gradient (REINFORCE) | 离散 token，标准 sequence-level reverse KL |
| **GKD** | Direct loss | 虽然是 on-policy prefix，但 KL 比较的是 full-vocab 分布（不依赖采样结果的那一侧） |
| **DeepSeek V4 OPD** | Direct loss (full-vocab reverse KL) | 依然是 E_{y~π_θ}[...] 期望在 π_θ 上？不是——它用的是 teacher top-k 匹配，不涉及 score function |
| **Revisiting OPD (top-K local)** | Direct loss | teacher top-K 重归一化后，梯度只穿过被积函数 |
| **DiffusionOPD** | Direct loss (reparameterization) | Gaussian transition 可重参数化，不需要 score function |
| **verl sampled-token** | Policy Gradient | 单 token reward，经典 REINFORCE |

---

## 6. 跟咱相关的

### 对图像生成（Qwen Image Edit / Klein 等）训练的影响

1. **文本生成路线（离散 token）：** 如果用的 VQ-VAE tokenizer → 离散 tokenizer → 做 OPD 必须走 Policy Gradient 或 top-K direct loss。MiniLLM 的方差问题在长序列（图像 latent 序列可以很长）会放大。

2. **连续 latent 路线（Diffusion）：** Diffusion 的 Gaussian denoising → 可重参数化 → **可以直接反传 KL 不需要 Policy Gradient**。这是 DiffusionOPD 相较于 LLM OPD 的巨大优势。

3. **目前的 GRPO 训练：** GRPO 就是 Policy Gradient 框架（group-relative advantage 本质是 REINFORCE with baseline）。你已经在用 Policy Gradient 了，只是 advantage 换成 group-relative score。

---

## 7. 参考资料

### 核心论文
- **MiniLLM**: Gu et al., "Knowledge Distillation of Large Language Models" — arXiv:2306.08543 (ICLR 2024). *首次将 reverse KL + policy gradient 引入 LLM 蒸馏。*
- **Gradient Estimation Using Stochastic Computation Graphs**: Schulman et al., NIPS 2015 — arXiv:1506.05254. *定义了 pathwise vs score function gradient，SCG 框架的根基。*
- **REINFORCE**: Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" — Machine Learning 1992. *Score function estimator 的原始论文。*
- **OPD Survey**: Song & Zheng, "A Survey of On-Policy Distillation" — arXiv:2604.00626 (2026).
- **Revisiting OPD**: "Empirical Failure Modes and Simple Fixes" (2026). *分析 sampled-token OPD 的三类失败 + top-K support matching 修复。*
- **DiffusionOPD**: Li et al., arXiv:2605.15055 (2026). *连续 latent 空间的 OPD，用 reparameterization 绕过策略梯度。*

### 代码仓库
- **Microsoft/LMOps/minillm**: MiniLLM 的官方实现 — github.com/microsoft/LMOps
- **chrisliu298/awesome-on-policy-distillation**: ⭐407, 372 条收录 — 最全的 OPD 论文/代码索引
- **volcengine/verl**: 支持 sampled-token / top-K OPD 的 RL 训练框架

### 中文社区
- **知乎《OPD 深度解析》**: zhuanlan.zhihu.com/p/2033212181823608430 — *从 MiniLLM 到 DeepSeek V4，完整推导了 sequence-level → token-level 的 policy gradient 公式，第 4 节给出了完整的梯度推导。*
- 小红书：搜索「policy gradient OPD 策略梯度」未获取到内容（超时）

---

> 📎 关联条目：`01_SFT_RL_OPD在DiT图像生成中的区别.md`（OPD 基础概念）、`10_SFT与RL的KL散度视角及On-Policy蒸馏.md`（Forward/Reverse KL 的分布论视角）


---

## 🔍 多模型交叉审核（2026-06-27）

以下为三模型并行审核的补充意见：

### 来自 GLM-5.2 — 工程落地 & 国产生态

 OPD 不自动继承 DDPM 的 reparameterization 优势，需要逐 sampler      
     验证可微性。                                                               
                                                                                
     5. 需要预判中文社区两个高频误解，否则知识库条目会被错误引用                
                                                                                
     (a) "REINFORCE 太不稳定，PPO 总是更好"——这是中文 ML                        
     社区（知乎/公众号）的高频论调。在 RLHF 场景下成立（KL 只是                 
     penalty，clipping 有意义），但在 OPD 场景下 KL 本身就是优化目标，PPO       
     的 ratio clipping 会直接 clip 掉你正在优化的信号。MiniLLM                  
     的实验明确显示 PPO clipping 损害 OPD 收敛。草稿应明确写一句：OPD           
     不适用 PPO clipping，这不是实现细节而是目标函数性质决定的。(b)             
     "on-policy = online training（无 replay                                    
     buffer）"——大量中文教程混用这两个概念。on-policy                           
     的数学定义是采样分布依赖当前 θ；importance sampling correction +           
     replay 在数学上可以保持无偏（虽然引入方差），不算严格违反                  
     on-policy。但对 MiniLLM 式 OPD，如果加 replay buffer 不做 IS               
     correction 就直接重用旧样本，梯度会有偏——因为旧样本的 ∇log π_θ 是在旧      
     θ 下采的。这个区分对实际工程选型有直接影响，建议补一句。                   
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184915_b74c8a

Session:        20260627_184915_b74c8a
Duration:       58s
Messages:       2 (1 user, 0 tool calls)


### 来自 Gemini 3.1 Pro — 多模态交叉 & Google 生态

                                               
     Shift"。它是有偏的（Biased），但在工程上收敛得极好，因为避免了 PG 那项     
     ∇log π_θ 带来的巨大方差。                                                  
                                                                                
     4. 方差控制（Baseline）的工程落差：Token-level vs Image-level              
     草稿 3.2 提到了减 Baseline，但在图像生成做 OPD/GRPO                        
     时，这里的坑极深。LLM 还可以算 token-level 的 reward（每个词的 logits      
     diff），但图像/Latent 序列如果走 discrete VQ-VAE 路线（比如 Klein /        
     Qwen Image），一整张图几百上千个 latent token，如果只给一个                
     Image-level 的 Reverse KL reward，PG 的方差会大到根本不收敛。              
     技术事实：在这里必须引入细粒度的 Advantage 估计。要么像 GRPO               
     那样在一个 prompt 下采 N 张图（group）强行做 batch                         
     内减均值（Group-Relative Advantage）；要么必须算 step-level 的 KL          
     差分。如果不强调图像场景对 Baseline                                        
     的严苛要求，用这套理论直接写代码大概率 loss 满天飞。                       
                                                                                
     建议把 DMD 的 Score Matching 等价性、DPO 的闭式解，以及 GKD                
     忽略状态分布偏移的本质加进去，这篇文档的段位就直接到顶了。                 
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184915_4b56bc

Session:        20260627_184915_4b56bc
Duration:       35s
Messages:       2 (1 user, 0 tool calls)


### 来自 Claude Opus 4.6 — 论证严密性 & 结构完备性

                                
                                                                                
     L = Σ_t E_{x_t ~ p_θ(x_t)} [ KL(p_θ(x_{t-1}|x_t) || p_T(x_{t-1}|x_t))      
     ]                                                                          
                                                                                
     这里每个 x_t 虽然是采样得到的，但KL 项内部是两个 Gaussian 分布的闭式       
     KL（不需要采样 x_{t-1}），所以可以对 μ_θ, σ_θ 直接求导。如果要算           
     sequence-level KL 并且需要 rollout 整条轨迹，依然需要策略梯度（见          
     DDPO, Black et al. 2023）。                                                
                                                                                
     建议：在 5.1 补充"Diffusion OPD 用 single-step marginal KL 而非            
     trajectory KL，前者可以用闭式 KL + reparameterization，后者依然需要        
     PG（如 DDPO）"。                                                           
     总结                                                                       
                                                                                
     这五条意见针对的是推导完备性和边界条件，不是内容错误——你的核心论证（OP     
     D 需要 PG 因为采样分布依赖 θ）是对的。补上这些后，条目会更扛得住深挖。     
                                                                                
 ────────────────────────────────────────────────────────────────────────────── 

Resume this session with:
  hermes --resume 20260627_184915_40e2b7

Session:        20260627_184915_40e2b7
Duration:       1m 5s
Messages:       2 (1 user, 0 tool calls)
