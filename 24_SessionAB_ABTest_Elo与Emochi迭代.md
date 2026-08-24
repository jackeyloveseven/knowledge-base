# Session A/B、传统 A/B Test 与 Elo：Emochi 模型评测和迭代闭环

> 记录日期：2026-08-10  
> 触发人：李承燕  
> 原问题：什么是 session AB，和 AB test 区别在哪，与 Elo 三者区别对比，对模型来说如何利用去迭代更新（业务产品：Emochi）

## 一句话总结

**Session A/B 是把“一整段会话”作为体验与评价单元的 A/B 设计；传统 A/B Test 是估计版本变化对业务指标的因果影响；Elo 是把大量 A/B 胜负聚合成模型相对排名的算法。**三者分别回答：这局谁更好、上线是否真的改善业务、候选模型长期综合排第几。

## 先消歧：Session A/B 不是统一标准术语

行业里“Session A/B”常有两种用法：

1. **会话级分流（session-level assignment）**：一个 conversation/session 从头到尾固定使用模型 A 或 B，比较两组会话的完成度、深度、留存、付费等指标。它本质上仍是 A/B Test，只是随机化/观测单元落在 session。
2. **会话级盲测（session battle / pairwise evaluation）**：同一个用户目标、角色设定和上下文分别由 A/B 完成整段多轮对话，用户或 judge 最终选整段会话谁更好。Chatbot Arena 允许用户继续多轮后再投票，其公开数据点包含两条多轮对话及一个偏好票（arXiv:2403.04132）。

对 Emochi 这种角色扮演产品，讨论“模型质量”时更可能指第 2 种；讨论线上实验、留存和收入时则是第 1 种。两者不能混着报结论。

## 三者核心区别

| 维度 | Session A/B（会话盲测/会话级实验） | 传统 A/B Test | Elo / Bradley–Terry 排名 |
|---|---|---|---|
| 本质 | 数据采集与评价设计 | 随机对照因果实验 | pairwise 结果的统计聚合/排名模型 |
| 核心问题 | 同一个完整会话里，A 还是 B 的整体体验更好？ | 把线上版本换成 B，业务 KPI 是否因果提升？ | 在很多候选模型之间，谁总体更强、相差多少？ |
| 比较粒度 | 多轮 session / trajectory | user、user-character、session、request 等实验单元 | 一场 A-vs-B 的 win/loss/tie |
| 输出 | 胜负票、会话偏好率、分维度评价 | uplift、置信区间、p-value/后验概率、护栏指标 | rating、排序、预期胜率、置信区间 |
| 是否直接证明业务收益 | 否；偏好胜出不等于留存/收入提升 | 是，前提是随机化、曝光和统计设计正确 | 否；只表达相对偏好排序 |
| 是否能支持多个模型 | 可以，但两两采样成本高 | 通常 A/B 或 A/B/n，流量成本高 | 擅长把不完全配对的多模型胜负汇总成榜单 |
| 主要偏差 | judge 偏差、位置偏差、用户自选、会话难度、长度偏好 | 样本污染、跨组串扰、分流单元错误、窥数、多重检验 | 对手/题目采样偏、时序漂移、投票作弊、分数不可跨池直接比较 |
| 对训练的价值 | 很高：能形成整段 trajectory 偏好和 turn-level pair | 间接：提供真实业务 reward 与上线门槛 | 间接：用于筛选 checkpoint、监控迭代和主动采样 |

**关键关系：Elo 不和 A/B 并列竞争。**先做很多次 A/B battle 得到胜负，再用 Elo 或 Bradley–Terry 把胜负压成全局 ranking。传统线上 A/B 则负责确认“排行榜更高”是否真的带来 Emochi 业务收益。

## Elo 到底在算什么

最简单的 Elo 假设：两个模型的 rating 差决定预期胜率；每得到一场胜负，就按“实际结果 − 预期结果”更新双方分数。它适合持续流入的在线对局。

需要把 **Elo 和 Bradley–Terry MLE 分开**：经典 Elo 是带 K-factor 的序贯更新，结果会受对局顺序和更新参数影响，更适合实时看板；冻结 checkpoint 的正式评测更适合用全部对局统一拟合 Bradley–Terry MLE，再通过 bootstrap 给置信区间。工程上大家常把二者都口语化叫“Elo 榜”，但统计报告里不能混写。

但模型榜单更推荐同时报告：

- 原始 win rate / tie rate；
- Bradley–Terry MLE 或 bootstrap Elo；
- 置信区间；
- 按场景切片后的 ranking。

Chatbot Arena 使用匿名随机 pairwise battle，并在论文中讨论 Bradley–Terry、Elo、bootstrap 和主动配对采样；截至论文统计时间有约 24 万票、9 万用户、50+ 模型。其公开实现位于 `lm-sys/FastChat`，含 `elo_analysis.py`、`rating_systems.py` 等代码。

Elo 的局限是：**一个总分会把角色一致性、剧情推进、情绪回应、文风、安全、延迟和成本揉成一维。**对 Emochi 必须有 overall Elo，也要有 category Elo；否则“更会写长回复”的模型可能凭长度偏好赢票，却降低聊天节奏和长期留存。

此外要显式控制回复长度和文风偏差：至少同时报告原始 BT 与 length/style-controlled BT，或在同长度桶内比较；否则榜首可能只是更爱写长文、消耗更多 token。角色偏好还可能是 context-dependent 的，单一潜在实力分无法表达“恋爱角色 A 更好、RPG 角色 B 更好”的循环偏好，因此分场景榜单比全局总榜更有决策价值。

## Emochi 应该如何设计评测

### 1. 明确实验单元，避免人格和记忆污染

- **仅改 temperature、采样器、短回复风格**：可按 `conversation_id/session_id` sticky 分流。
- **改 system prompt、角色塑造、长期记忆、模型底座**：推荐按 `user_id × character_id` sticky 分流，并在整个实验窗口保持不变。
- **同一个用户会跨角色感知全局模型变化**：必要时提升到 user-level sticky。

原因：Emochi 的价值来自连续关系和剧情。如果同一角色今天用 A、明天用 B，历史记忆、语气和人格发生切换，处理效应会污染，session-level 随机化反而不再能干净估计长期效果。

实验前必须把 session 写成可执行定义，例如“`user_id × character_id` 下，从首次消息开始，到主动退出或连续 30 分钟无消息结束”。Sticky 也不是免费午餐：人格/记忆有 carryover；一个用户的多个角色若共享用户画像还会产生跨角色 spillover。此时应提升到 user-level cluster randomization，或至少在分析中按 user 聚类计算标准误。长期实验还要保留稳定 control，避免历史 treatment 无法清除后又被重新随机化。

### 2. 建立三层指标，不让单一指标绑架模型

**模型质量层（session battle）**
- 角色一致性 / persona adherence
- 记忆召回与事实一致性
- 情绪理解、共情和关系推进
- 剧情推进、主动性、惊喜度
- 重复、跑题、复读、拒答、出戏
- overall preference

**用户行为层（online A/B）**
- session 有效轮数与主动续聊率
- 首次回复后立即退出率
- regenerate / edit / delete / report 率
- 同角色 D1 / D7 回访
- 新建角色或切换角色率（需区分探索与逃离）
- 付费转化、订阅续费、token 消耗

**工程护栏层**
- 首 token 延迟、整句延迟、失败率
- 单 session 推理成本
- 安全事件、举报率、过度依赖风险

Safety pipeline 必须作为控制变量：若 A/B 两组的审核器、拒答阈值、后处理不同，胜负可能主要来自审核策略而非底模能力。模型实验默认固定同一套 safety 配置；若安全策略也是 treatment，就应单独做 factorial experiment 或明确拆开归因。

不要直接把“会话越长”当 reward。长对话可能来自模型啰嗦、用户纠错或失败重试。主指标应更接近“自愿持续互动 + 次日回到同一角色”，并配 regenerate、退出率、延迟和安全护栏。

指标要按决策速度分层：离线 pairwise/BT 是小时级代理指标；regenerate、首轮退出、主动续聊是天级先行指标；同角色 D1/D7、订阅续费是低频北极星指标。统计 D1/D7 时分母必须包含全部被曝光用户，不能只看回访幸存者；同时区分“产品回访”“同角色回访”“同角色新开 session”，三者含义不同。

### 3. 推荐闭环：离线盲测 → Elo 筛选 → 线上因果验证 → 训练更新

1. **离线 session replay**：从真实流量按角色类型、语言、关系阶段、会话长度分层抽样；固定起始 context，让候选模型生成完整 trajectory。
2. **Pairwise judge**：先用人工/高质量 LLM judge 做 A/B/tie；交换左右位置，控制位置偏差；对关键切片做人审校准。MT-Bench/Chatbot Arena 论文报告强 LLM judge 与人类偏好可达到 80% 以上一致，但同时指出位置、冗长、自增强等偏差（arXiv:2306.05685）。
3. **Elo/BT 排名**：计算 overall + category + segment ranking，并用 bootstrap CI 判断排名是否稳定；优先给 rating 接近、区间重叠的模型配对，提高样本效率。
4. **线上 A/B**：只让 Elo/质量门槛合格的 1–2 个 challenger 进真实流量；用 sticky assignment 验证 D1/D7、同角色回访、付费、成本和安全。
5. **训练数据回流**：将高置信胜负对变成偏好数据，将高质量胜者变成 SFT 数据，将业务负反馈变成 hard negatives。

评测数据和训练数据必须设防火墙：固定 benchmark/长期 holdout 永不回流训练；新收集 battle 数据先按 user、character、session 分组切分，再分别进入 train/eval。否则同一批 pair 既算榜单又训练 DPO/RM，Elo 提升只是在评测集上过拟合。主观维度还应记录双标一致率或 Krippendorff's α，低一致性标签降权或丢弃。

## 这些数据如何真正更新模型

### 路线 A：SFT

把胜出的高质量回复/会话清洗后作为 target：

```text
messages = [system, character_card, memory, user/assistant history..., user_turn]
target   = winning_assistant_response
```

适合补角色语气、剧情模板、记忆使用方式。不要直接把整个赢家日志无筛选灌回去，避免复制用户诱导、隐私、低质量长回复和偶然风格。

### 路线 B：DPO / IPO / SimPO

将同一 context 下的 A/B 组成偏好对：

```text
prompt   = system + character + memory + conversation_history + current_user_turn
chosen   = preferred_response
rejected = non_preferred_response
```

Session 最终票不能粗暴复制给每一轮。更稳的做法是：
- 收集关键 turn 的局部偏好；或
- 用 turn-level judge/credit assignment 找出导致整局胜负的转折点；
- 对不确定轮次降权；
- 同一 session 的样本放在同一个 train/val split，防止泄漏。

如果无法可靠定位关键 turn，宁可保留 session-level trajectory preference 训练 trajectory reward，也不要强拆成伪 turn 标签。可行的归因流程是：从最终投票点逆向扫描 → 标出首次 persona/memory/story 失误 → 交换左右位置复评 → 仅保留人类与 RP judge 一致的高置信 turn pair。线上退出/留存是稀疏延迟 reward，不能无校正地平均摊给前面每一轮。

### 路线 C：Reward Model + RL/GRPO

用 A/B/tie 学习 `reward(context, response)` 或 `reward(trajectory)`，再做 RL/GRPO。Reward 建议拆成多头：persona、memory、engagement、story、safety、style；最后受控加权，而不是只拟合 overall vote。线上行为指标可做弱标签，但必须去除延迟、价格、UI、角色热度、用户活跃度等混杂因素。

### 路线 D：模型路由而非立刻改权重

Elo 按 segment 可能显示：模型 A 擅长 romance，B 擅长 RPG，C 擅长非英语。此时先做场景路由通常比训练一个平均模型更快：

```text
router(character_tags, language, relationship_stage, safety_level) -> model/checkpoint
```

但路由上线仍需用户级或 user-character sticky，不能每轮跳模型。

## 最小可落地方案（Emochi）

1. 建一个匿名 session battle 池：当前生产模型 + 2 个 challenger；每个 battle 使用相同 character/context，左右位置随机。
2. 每场记录 `battle_id, user_hash, character_id, segment, model_a/b, full trajectories, winner/tie, reason tags, latency, cost`。
3. 每周产出 overall/category/segment BT-Elo + bootstrap CI；低于生产模型且区间明显分离的 checkpoint 淘汰。
4. 榜单胜者进入 5%→20%→50% sticky online A/B；主指标用同角色回访/自愿续聊，护栏用 regenerate、退出、举报、延迟和成本。
5. 训练集只吸收高置信 pair；按用户和 session 去重；保留固定 holdout 和长期 control，防止评测—训练闭环自嗨。

## 我的判断

对 Emochi，**Session A/B 应成为模型团队的高频质量评测，Elo/BT 是候选管理层，传统 A/B Test 是最终上线裁判。**不要拿 Elo 上涨直接宣布产品提升，也不要拿 session 时长上涨直接训练 reward。真正有效的链路是：

```text
整段会话偏好 → 分维度/分场景 Elo → 小流量 sticky A/B → 真实留存与关系指标 → SFT/DPO/RM/RL 数据回流
```

其中最关键的工程决策不是 Elo 公式，而是**分流单元和 credit assignment**：涉及长期人格/记忆时，必须至少按 user-character 固定模型；整局偏好必须定位到关键 turn，才能变成干净训练信号。

## 多源搜索状态

- 已有知识库：未找到直接覆盖 Session A/B / Elo / Emochi 的条目。
- arXiv：命中 Chatbot Arena、MT-Bench/LLM-as-a-Judge、CHARM 等相关论文。
- Hugging Face：命中 `lmsys/chatbot_arena_conversations` 等公开会话数据集。
- GitHub：命中 `lm-sys/FastChat` 的 Arena 榜单与 rating 实现。
- 中文缓存：未命中本主题。
- Google：403/CAPTCHA，未纳入。
- 知乎：安全验证页，未搜索到有效正文。
- 小红书：仅返回页脚，未搜索到有效正文。

## 多模型交叉审核（2026-08-10）

### 来自 GLM-5.2

- 建议把角色榜单按 persona 类别而非单角色 ID 分桶，避免样本过碎；报告置信区间，不凭裸分差下结论。
- 强调 safety pipeline 会成为 A/B 混淆变量；已吸收为固定审核配置/独立 treatment 的规则。
- 建议拆分产品回访、同角色回访和同角色续聊；已纳入指标层级。

### 来自 Gemini 3.1 Pro

- 指出角色扮演评测尤其容易被回复长度和文风偏差劫持；已增加 length/style-controlled BT。
- 指出 user-character sticky 存在 carryover、迁移和跨角色 spillover；已增加 cluster randomization 与长期 control。
- 建议把 turn credit assignment 落到错误定位流程，而非只写概念；已补充逆向扫描与高置信 turn pair。

### 来自 Claude Opus 4.8

- 要求严格区分经典在线 Elo 与静态 BT-MLE；已吸收，并将 BT-MLE + bootstrap CI 设为正式报告优先项。
- 指出评测 pair 回流 DPO/RM 会导致数据泄漏；已增加 benchmark/holdout 防火墙和按 user/session 分组切分。
- 指出留存有滞后性和幸存者偏差；已增加代理—先行—北极星指标层级及完整曝光分母。

审核中的“每对至少 30–50 局”“实验至少两周”等经验阈值未直接采纳：它们缺少本次检索证据，应由 Emochi 实际方差、MDE、流量和 power analysis 决定。

## 参考资料

- [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference, arXiv:2403.04132](https://arxiv.org/abs/2403.04132)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- [CHARM: Calibrating Reward Models With Chatbot Arena Scores, arXiv:2504.10045](https://arxiv.org/abs/2504.10045)
- [FastChat Arena rating implementation](https://github.com/lm-sys/FastChat/tree/main/fastchat/serve/monitor)
- [LMSYS Chatbot Arena Conversations dataset](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations)
- [Microsoft Experimentation Platform: Deep Dive Into Variance Reduction](https://www.microsoft.com/en-us/research/group/experimentation-platform-exp/articles/deep-dive-into-variance-reduction/)
- [Emochi official website](https://emochi.com/)
