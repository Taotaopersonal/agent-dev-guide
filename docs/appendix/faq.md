# 常见问题 (FAQ)

本附录汇总了 Agent 开发学习和实践中的高频问题，覆盖从入门到部署的全流程。

---

## 入门常见困惑

### Q1: 学 Agent 开发需要深入理解深度学习和数学吗？

**不需要**。Agent 开发更偏向"应用层"而非"模型层"。你需要理解的是：
- LLM 是什么、能做什么、有什么局限（概念层）
- 如何通过 API 调用模型、如何设计 Prompt（实操层）
- Agent 的设计模式和工程化实践（架构层）

你不需要会推导反向传播公式或训练模型。第 2 章的 LLM 原理讲解已覆盖必要的基础知识。

### Q2: Python 基础不扎实，能学 Agent 开发吗？

如果你有其他语言的编程经验（比如 JavaScript），Python 的语法上手很快。本书第 1 章覆盖了 Agent 开发所需的 Python 知识：
- 基础语法、数据结构
- 异步编程（`async/await`）
- 类和模块系统
- 包管理（uv）

不需要掌握 NumPy、Pandas 等数据科学库。

### Q3: Agent、Copilot、Chatbot 有什么区别？

| 特性 | Chatbot | Copilot | Agent |
|------|---------|---------|-------|
| 交互方式 | 问答对话 | 辅助建议 | 自主执行 |
| 工具使用 | 无 | 有限 | 丰富 |
| 自主决策 | 无 | 人类主导 | Agent 主导 |
| 典型案例 | 客服机器人 | GitHub Copilot | Claude Agent |

简单说：Chatbot 是"你问我答"，Copilot 是"我帮你做"，Agent 是"我替你做"。

### Q4: 学 LangChain 还是直接用 SDK？

建议的学习路径是：**先裸 SDK，再学框架**。

1. 先用 Anthropic/OpenAI 原生 SDK 手写 Agent 循环（本书 P2 项目）
2. 理解框架解决的核心问题后，再学 LangGraph/CrewAI
3. 在实际项目中按需选择框架或裸 SDK

先学框架容易"知其然不知其所以然"，遇到框架无法覆盖的场景就束手无策。

---

## API 使用问题

### Q5: Anthropic 和 OpenAI 的 API 怎么选？

| 考虑因素 | Anthropic (Claude) | OpenAI (GPT) |
|---------|-------------------|--------------|
| 代码生成 | Claude Sonnet 极强 | GPT-4o 也很强 |
| 长文本理解 | 200K 上下文 | 128K 上下文 |
| Tool Use | 原生支持，设计清晰 | Function Calling，成熟稳定 |
| 中文能力 | 优秀 | 优秀 |
| 定价 | 性价比高（Haiku） | 性价比高（GPT-4o-mini） |

**建议**：学习阶段都试试，生产环境根据具体任务表现选择。本书以 Claude 为主，但所有概念和模式也适用于 OpenAI。

### Q6: API 返回 429 (Rate Limit) 怎么办？

三种处理策略：

```python
import time
import anthropic

# 策略 1：指数退避重试
def retry_with_backoff(func, max_retries=3):
    for i in range(max_retries):
        try:
            return func()
        except anthropic.RateLimitError:
            delay = 2 ** i
            time.sleep(delay)
    raise Exception("重试次数用尽")

# 策略 2：请求队列 + 速率控制
# 使用 asyncio.Semaphore 控制并发数

# 策略 3：升级 API 计划以获取更高限额
```

### Q7: 如何降低 API 费用？

| 方法 | 效果 | 说明 |
|------|------|------|
| 使用 Haiku 开发调试 | 节省 90%+ | 开发调试用 Haiku，正式用 Sonnet |
| 减少上下文长度 | 节省 30-60% | 及时截断/摘要历史消息 |
| 缓存常见查询 | 节省 50%+ | 相同输入直接返回缓存结果 |
| Prompt 精简 | 节省 10-30% | 去除冗余指令，减少 System Prompt 长度 |
| Prompt Caching | 节省 90% (缓存部分) | Anthropic 支持 Prompt Caching |

### Q8: streaming 模式下如何获取 token 使用量？

```python
# Anthropic
with client.messages.stream(...) as stream:
    for text in stream.text_stream:
        print(text, end="")

    # 流结束后获取 usage
    final = stream.get_final_message()
    print(f"Input: {final.usage.input_tokens}")
    print(f"Output: {final.usage.output_tokens}")
```

---

## Agent 开发技巧

### Q9: Agent 进入无限循环怎么办？

**必须设置最大循环次数**：

```python
MAX_ITERATIONS = 10

for i in range(MAX_ITERATIONS):
    response = llm.call(messages)
    if response.stop_reason == "end_turn":
        break
    # 处理工具调用...
else:
    # 超过最大次数，强制停止
    return "任务过于复杂，已达到最大尝试次数。"
```

常见原因及解决方法：
- **工具描述不清晰** -- 改进工具的 description，帮助 LLM 正确选择
- **任务太模糊** -- 让 LLM 在开始前先确认理解
- **工具返回错误被忽略** -- 确保错误信息清晰，LLM 能据此调整策略

### Q10: 如何让 Agent 更可靠地使用工具？

```python
# 1. 工具描述要具体
{
    "name": "search",
    "description": (
        "搜索互联网获取信息。"
        "当你需要查找最新数据、新闻、或不确定的事实时使用。"
        "不要用于回答你已经知道的常识问题。"
    ),
}

# 2. 参数描述要明确
"properties": {
    "query": {
        "type": "string",
        "description": "搜索关键词，2-5 个词最佳。不要用完整句子。",
    },
}

# 3. System Prompt 中给出使用指导
system = """你有以下工具可用：...
使用原则：
- 不确定的事实，先搜索再回答
- 数学计算，用 calculator 而不是心算
- 回答完整后才停止，不要中途返回
"""
```

### Q11: 如何处理工具执行超时？

```python
import asyncio

async def execute_tool_with_timeout(tool_func, args, timeout=30):
    """带超时的工具执行"""
    try:
        result = await asyncio.wait_for(
            tool_func(**args),
            timeout=timeout,
        )
        return result
    except asyncio.TimeoutError:
        return f"工具执行超时（{timeout}秒）。请尝试简化请求或使用其他方法。"
    except Exception as e:
        return f"工具执行错误: {str(e)}"
```

### Q12: RAG 检索到的内容不相关怎么办？

排查清单：

1. **分块太大或太小** -- 一般 200-500 字符为宜，视文档类型调整
2. **Embedding 模型不匹配** -- 中文文档用多语言模型（如 `paraphrase-multilingual-MiniLM-L12-v2`）
3. **缺少 Reranking** -- 向量检索 top-20，再用 Reranker 精排到 top-5
4. **Query 和文档风格差异大** -- 尝试 query 改写或 HyDE 技术
5. **文档质量差** -- 源文档本身就不包含答案

```python
# 添加 Reranking 示例
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, doc) for doc in candidates])
reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
```

### Q13: Multi-Agent 系统中 Agent 之间的消息越来越长怎么办？

三种策略：

1. **消息摘要** -- 定期用 LLM 摘要已处理的消息，替换原始内容
2. **黑板模式** -- Agent 只往共享状态写结论，不传递原始对话
3. **消息过期** -- 设置 TTL，自动丢弃旧消息

```python
# 消息摘要示例
def summarize_if_too_long(messages, max_tokens=4000):
    """当消息过长时自动摘要"""
    total_chars = sum(len(str(m)) for m in messages)
    if total_chars > max_tokens * 4:  # 粗略估算
        summary = llm.summarize(messages[:-2])  # 保留最近 2 条
        return [{"role": "system", "content": f"之前的对话摘要: {summary}"}] + messages[-2:]
    return messages
```

---

## 部署和运维问题

### Q14: Agent 应用适合用什么架构部署？

```
推荐架构：
├── 前端：静态部署（Vercel / Netlify / CDN）
├── 后端 API：容器化部署（Docker + K8s / ECS）
├── WebSocket：独立 WebSocket 服务或 API Gateway
├── 数据库：托管服务（RDS / Cloud SQL）
├── 向量库：托管服务或嵌入式（ChromaDB 文件挂载）
└── LLM API：直连云端 API（Anthropic / OpenAI）
```

对于小规模应用，一台 2C4G 的服务器 + Docker Compose 就够了。

### Q15: Agent 应用的延迟如何优化？

| 优化方向 | 方法 | 效果 |
|---------|------|------|
| 模型选择 | 简单任务用 Haiku | 延迟降低 3-5 倍 |
| 流式输出 | 使用 streaming | 首字延迟降至 <1s |
| Prompt 缓存 | Anthropic Prompt Caching | 减少重复计算 |
| 并行工具调用 | 多工具并行执行 | 工具调用阶段加速 2-3 倍 |
| 预热连接 | 复用 HTTP 连接池 | 减少连接建立延迟 |
| 结果缓存 | Redis 缓存常见查询 | 命中时延迟降至 <100ms |

### Q16: 如何监控 Agent 应用的运行状态？

关键监控指标：

```python
# 应用级指标
metrics = {
    "request_count": "请求总数",
    "request_latency_p50": "P50 延迟",
    "request_latency_p99": "P99 延迟",
    "error_rate": "错误率",
    "token_usage_daily": "每日 Token 消耗",
    "cost_daily": "每日 API 费用",
    "tool_call_count": "工具调用次数",
    "agent_loop_iterations": "平均 Agent 循环次数",
}
```

推荐工具：
- **LangSmith** -- Agent 专用可观测性平台
- **Prometheus + Grafana** -- 通用监控方案
- **Sentry** -- 错误追踪

### Q17: 如何处理 Agent 应用的安全问题？

安全检查清单：

- [ ] API Key 不在前端暴露，通过后端代理调用
- [ ] 用户输入经过 Prompt Injection 防护
- [ ] 工具执行在沙箱环境中运行
- [ ] 文件操作限制在安全目录内
- [ ] 代码执行有超时和资源限制
- [ ] 敏感信息（PII）在传入 LLM 前脱敏
- [ ] 输出经过内容安全过滤
- [ ] 速率限制防止滥用

详见高级篇第 13 章。

---

## 进阶和职业发展

### Q18: Agent 开发工程师需要哪些核心技能？

```
核心技能树：
├── 编程能力
│   ├── Python（主力语言）
│   ├── TypeScript（MCP / 前端 / Node Agent）
│   └── 异步编程 + 并发
├── LLM 能力
│   ├── Prompt Engineering
│   ├── Tool Use / Function Calling
│   └── 模型能力边界理解
├── 系统设计
│   ├── Agent 架构设计
│   ├── RAG 管线设计
│   └── Multi-Agent 协作设计
├── 工程化
│   ├── API 设计和开发
│   ├── 数据库设计
│   ├── Docker / K8s 部署
│   └── CI/CD 流程
└── 领域知识
    ├── 安全与隐私
    ├── 成本优化
    └── 评估和测试
```

### Q19: Agent 开发最容易踩的坑有哪些？

**Top 5 踩坑排行榜**：

1. **过度依赖 LLM** -- 把简单的 `if/else` 逻辑也让 LLM 来做。规则明确的部分用代码，模糊的部分才用 LLM。

2. **忽略错误处理** -- Agent 循环中任何一步都可能失败（API 超时、工具异常、格式错误），每一步都需要异常处理。

3. **上下文窗口溢出** -- 多轮对话后消息历史不断增长，最终超过模型上下文窗口。需要主动管理消息长度。

4. **工具描述写得太随意** -- 工具的 description 就是 LLM 的使用手册，写得不清楚就会误用。

5. **没做 Evaluation** -- 只凭感觉判断 Agent 表现，没有建立系统的评估机制。改了 Prompt 之后无法确认是变好还是变差。

### Q20: 如何持续跟进 Agent 开发领域的最新进展？

| 渠道 | 频率 | 说明 |
|------|------|------|
| Anthropic Blog | 每周 | 模型更新和最佳实践 |
| OpenAI Blog | 每周 | 模型更新和新功能 |
| arXiv (cs.AI) | 每日 | 最新研究论文 |
| Hacker News | 每日 | 技术社区热点讨论 |
| Twitter/X | 实时 | 关注 @AnthropicAI, @OpenAI, @LangChainAI |
| Latent Space Podcast | 每周 | AI 工程深度访谈 |
| 本书附录资源页 | 持续更新 | [推荐资源](./resources.md) |

### Q21: 用 Claude 还是 GPT 做 Agent 比较好？

两者各有优势，取决于具体场景：

- **Claude 优势**: 长上下文 (200K)、代码生成质量高、Tool Use API 设计清晰、MCP 生态
- **GPT 优势**: 生态成熟、多模态能力强 (DALL-E, TTS)、Azure 企业支持

实践建议：不要押注单一模型。设计好抽象层，让 Agent 可以灵活切换底层模型。

### Q22: Agent 应用有哪些成熟的商业模式？

| 模式 | 案例 | 说明 |
|------|------|------|
| SaaS 订阅 | Cursor, Replit | 按月付费使用 Agent 功能 |
| API 服务 | LangSmith, Cohere | 为 Agent 开发者提供基础设施 |
| 垂直领域 | Harvey (法律), Abridge (医疗) | 深耕特定行业的 Agent 应用 |
| 开源 + 企业版 | LangChain, CrewAI | 开源社区版 + 商业企业版 |
| 工具生态 | MCP Server | 为 AI 生态开发工具和集成 |

---

::: tip 还有问题？
如果你的问题没有在这里找到答案，可以：
1. 回到对应章节查找详细说明
2. 在 Anthropic Discord 或 LangChain 社区提问
3. 查阅[推荐资源](./resources.md)中的官方文档
:::

## 参考资源

- [Anthropic API FAQ](https://docs.anthropic.com/en/docs/resources/faq) -- 官方 API 常见问题
- [OpenAI Help Center](https://help.openai.com/) -- OpenAI 帮助中心
- [LangChain Troubleshooting](https://python.langchain.com/docs/troubleshooting/) -- LangChain 常见问题排查
- [Stack Overflow - LangChain Tag](https://stackoverflow.com/questions/tagged/langchain) -- 社区问答
