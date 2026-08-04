# Codex 加密推理状态：`encrypted_content` 机制与“CoT 泄露”辨析

> 记录日期：2026-07-22 | 触发人：李承燕
> 触发材料：《国内大模型蒸馏风波的来龙去脉.pdf》

## 一句话总结

Codex 确实会接收、保存并原样回传 OpenAI Responses API 返回的加密推理项 `encrypted_content`；实测其外层字节布局与 Fernet wire format 高度一致。但公开证据只能证明它是服务端可读取的 opaque encrypted reasoning state，不能证明客户端已经拿到、解密或可读地提取了“完整原始自然语言思维链”。PDF 把“可回传加密状态”直接升级成“CoT 已破解”，证据链断了一截。

## 1. 先说 Codex 具体怎么实现

Codex 不是在本地“读取思维链”，而是在 Responses API 的多轮工具调用中搬运一个不透明状态包：

1. 请求时主动要求返回加密推理项：
   - `codex-rs/core/src/client.rs:865-872` 构造 reasoning 参数，并设置 `include = ["reasoning.encrypted_content"]`。
2. 响应结构中保存三类内容：
   - `summary`：可读推理摘要；
   - `content`：协议层可表示的 reasoning content；
   - `encrypted_content`：不透明密文。
   - 定义位于 `codex-rs/protocol/src/models.rs:831-842`。
3. Codex 不会把原始 `ReasoningText` 序列化回普通请求：
   - `should_serialize_reasoning_content()` 在发现 `ReasoningText` 时返回 false，见 `models.rs:1322-1328`。
4. Codex 的 rollout trace 明确把密文描述为：
   - “Opaque model-visible content that is intentionally not decoded here.”
   - 见 `codex-rs/rollout-trace/src/model/conversation.rs:114-119`。
5. 后续请求重放时，Codex 将 blob 当作 reasoning item 的稳定身份：
   - `rollout-trace/src/reducer/conversation.rs:615-627` 明确说明后续 request snapshot 可能只回放 encrypted blob，并用 blob 识别同一 reasoning item。

因此工程链路是：

`模型内部推理 → API 返回 summary + opaque encrypted_content → Codex 保存 reasoning item → 工具结果回来后连同 encrypted_content 原样回传 → 服务端恢复/续接推理状态`

不是：

`Codex 拿到密文 → 本地解密 → 读取原始 CoT → 展示或训练`

## 2. OpenAI 官方怎么定义

OpenAI 官方 Reasoning Models 文档给出几个关键边界：

- reasoning tokens 不通过 API 可见，但会计入输出 token 和上下文；
- API 可选返回 reasoning summary；summary 不是 raw reasoning；
- `store=false` / ZDR 场景下，reasoning item 默认包含 `encrypted_content`；
- 客户端可以把完整 output item 回传到未来调用；
- persisted reasoning 提供连续性，但“does not expose the model’s raw reasoning”；
- 官方使用的是 encrypted reasoning tokens / opaque reasoning item，而不是“客户端可读的加密 CoT 文本”。

这说明“回传”是官方支持的状态续接协议，不等于“泄露明文”。

## 3. 本机 live API 复验

对 `gpt-5.6-sol` 的 Responses API 发起 `store=false` 请求，获得：

- status：completed
- reasoning tokens：51
- reasoning item：存在
- `encrypted_content`：1080 字符
- 前缀：`gAAAAAB...`
- 同时返回一条可读 summary；最终答案与 summary 分开存在。

对密文做 URL-safe Base64 解码后：

- 总长度：809 bytes
- version byte：`0x80`
- timestamp：8 bytes，可解析为请求时刻
- IV：16 bytes
- ciphertext：752 bytes，满足 16-byte block alignment
- 尾部：32 bytes

该布局与 Fernet token 的 wire format 完全对齐：

`0x80 version + 8B timestamp + 16B IV + block-aligned ciphertext + 32B authentication tail`

严谨表述应是：**已高度确认你当前使用的 `gpt-5.6-sol` OpenAI-compatible endpoint 返回 Fernet-compatible wire format。** 由于本次调用经过自建兼容端点而非直接请求 `api.openai.com`，不能仅凭这次抓包排除代理层封装或改写；要把 Fernet 格式最终归因给 OpenAI 上游，还需要在官方 endpoint 做同样的字节复验。在没有密钥、无法验证 authentication tag 和底层算法实现的情况下，也不应把“wire format 对齐”无限外推为服务端全部密码学实现细节都已确认。

### 长度与 reasoning tokens

补充三档测试：

| 任务 | reasoning tokens | encrypted_content 长度 |
|---|---:|---:|
| 17+28 | 0 | 无 reasoning item |
| 解三次方程 | 51 | 1080 chars |
| 证明 4k+3 型素数无限 | 114 | 1484 chars |

可观察到推理量增加时 blob 变长。但该相关性只能证明 blob 承载的信息规模随推理量增长，不能单独证明其明文是“完整自然语言 CoT 全文”。它也可能是 token 序列、结构化推理状态或其他服务端内部表示；公开资料不足以进一步定性。

## 4. PDF 各项主张的可信度

| PDF 主张 | 判断 | 依据 |
|---|---|---|
| Responses API 返回加密 reasoning blob | 已确认 | 官方文档 + live API |
| blob 前缀与结构符合 Fernet | 当前兼容端点高度确认；上游归因待官方 endpoint 复验 | live 字节解析，但请求经过自建代理 |
| Codex 会保存并回传 blob | 已确认 | OpenAI Codex 源码 |
| blob 长度随 reasoning tokens 增长 | 小样本确认 | live 三档测试 |
| blob 内必然是完整自然语言原始 CoT | 未证明 | 长度相关性不能确定明文语义；官方只称 opaque encrypted reasoning |
| 能回传 blob | 已确认 | 官方协议能力 |
| “回传”等于“破解/解密/提取 CoT” | 错误 | 客户端无密钥、源码不解码、官方明确 raw reasoning 不暴露 |
| 5.6 luna/sol/terra 共享密钥、跨模型互通 | 未独立验证 | PDF 无可核查实验数据或公开来源 |
| blob 至少 2.5 小时有效 | 未独立验证 | timestamp 不等于 TTL；需要过期边界实验 |
| 注入 blob 可诱导模型复述完整原始 CoT | 未独立验证 | 无公开可复现实验、日志或代码 |
| 历史 blob 一定由 OpenAI 丢弃 | 表述已过时/场景化 | 当前官方支持 `reasoning.context=all_turns`，行为依模型、请求模式和代理层而异 |

## 5. 对“Codex 思维链”的准确理解

Codex 里其实有三层东西：

1. **Raw internal reasoning**
   - 模型服务端内部使用；公开 API 不直接返回。
2. **Reasoning summary**
   - 模型生成的可读摘要；Codex UI 里看到的“thinking/reasoning”主要属于这一层。
3. **Encrypted reasoning state (`encrypted_content`)**
   - 客户端不可读；用于跨调用续接工具使用、规划和推理状态。

因此把 Codex 的 `encrypted_content` 叫“加密思维链”可以作为口语简称，但技术文档里最好叫：

**加密推理状态 / encrypted reasoning item**

避免直接叫：

**已提取的完整原始 CoT**

## 6. 对 PDF 的总判断

PDF 技术附录抓到了一个真实机制：OpenAI reasoning 模型确实返回 Fernet-format-compatible 的 opaque encrypted reasoning item，Codex 确实会搬运它。这个观察有技术价值。

但文档把三件事混在了一起：

1. 拿到密文；
2. 服务端能重放密文；
3. 客户端已经恢复出可读原始 CoT。

目前只能确认前两件，第三件缺少公开、可复现证据。再叠加全文大量匿名行业消息、无引用的厂商行为指控，不能把它当作“国内模型蒸馏事实报告”，最多当作“匿名爆料 + 一部分可复现 API 观察”。

综合可信度：

- Codex / Responses API 技术机制：高可信；
- Fernet-compatible 封装判断：高可信；
- “完整 CoT 已破解并被大规模蒸馏”：低可信、证据不足；
- 行业厂商时间线与具体指控：无法仅凭该 PDF 核实。

## 7. 多模型交叉审核

### GLM-5.2

有效意见：将“Fernet 完全一致”收紧为“Fernet wire-format compatible”；把“回传”与“复现/可见原始推理”做语义切割；指出长度相关性不能证明自然语言明文。

未采纳意见：其推测 `encrypted_content` 大概率是 latent/KV state，没有公开证据，不能反向替换 PDF 的另一种猜测。

### Gemini 3.1 Pro

有效意见：强调 Codex 是 passthrough；跨模型共享密钥、TTL、注入复述都需要单独实验；Fernet 固定开销使字符数不能直接等价为 token 数。

未采纳意见：把 opaque state 直接定性为 latent state，同样属于无来源猜测。

### Claude Opus 4.8

有效意见：建议把可复现技术事实与匿名传闻物理分区；明确“回传密文续接”与“解密明文”不同；TTL 不能从 timestamp 推导。

未采纳意见：声称“注入复述与 HMAC 矛盾”不严谨——原样重放合法 blob 不涉及篡改，HMAC 只阻止伪造/改写，并不阻止服务端读取合法密文。

## 8. 补充实测：公开 API 是否能返回 raw reasoning

对 `gpt-5.6-sol` 做了两组补充测试：

1. `summary=auto` 与 `summary=detailed`：分别使用 54 / 60 reasoning tokens，返回 1292 / 1316 字符的 encrypted blob；两者的 `reasoning.content` 均为空，只返回 `summary_text`。
2. 在复杂任务中尝试 `include=["reasoning.content"]`、`include=["reasoning.raw_content"]` 与文档化的 `include=["reasoning.encrypted_content"]`：当前兼容端点均未返回可读 raw reasoning；reasoning item 仍只有 summary、空 content 与 encrypted_content。未文档化字段可能被代理静默忽略，因此不能据此推断官方 endpoint 的参数校验行为。

结论：当前公开协议可获得的上限是 reasoning summary、opaque encrypted reasoning item、最终答案和完整 Agent trajectory。没有发现正常 API 参数能将 encrypted_content 转换成 raw reasoning text。蒸馏应优先记录 summary、tool call/result、错误恢复、patch diff、测试结果与最终复盘，而不是收集不可解释的密文。

## 参考资料

- OpenAI, Reasoning models: https://developers.openai.com/api/docs/guides/reasoning
- OpenAI, Conversation state: https://developers.openai.com/api/docs/guides/conversation-state
- OpenAI Codex GitHub: https://github.com/openai/codex
- Codex request builder: https://github.com/openai/codex/blob/main/codex-rs/core/src/client.rs
- Codex response item models: https://github.com/openai/codex/blob/main/codex-rs/protocol/src/models.rs
- Codex rollout trace model: https://github.com/openai/codex/blob/main/codex-rs/rollout-trace/src/model/conversation.rs

### 多源搜索状态

- 已有知识库：命中 CoT 标注与蒸馏条目，但未覆盖 Codex encrypted reasoning。
- arXiv：未找到针对该具体 GPT-5.6 blob / Codex 机制的独立论文。
- Hugging Face：API 本次超时，未取得相关模型结果。
- GitHub：已直接核查 OpenAI 官方 Codex 源码。
- Google：r.jina.ai 仅返回重定向页，无有效结果。
- 知乎：触发安全验证，无有效结果。
- 小红书：仅返回页面框架，无有效结果。
- Bing：精确短语未检索到相关公开来源。
