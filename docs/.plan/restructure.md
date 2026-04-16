# 目录重构方案

## 核心思路

每个章节 = 一个知识主题，内部分初/中/高三个深度层级。

- **读完所有初级** → 能写简单的 Agent 应用（调 API、用工具、单轮对话）
- **读完所有中级** → 能用框架搭建 RAG、Multi-Agent 等复杂系统
- **读完所有高级** → 能做生产部署、性能调优、安全防护、自建框架

## 新目录结构

```
docs/
├── index.md                    # 首页
├── guide.md                    # 导读（保留）
├── roadmap.md                  # 学习路线图（保留，调整内容）
│
├── 01-python/                  # 第 1 章 · Python 基础
│   ├── index.md               # 章节概览
│   ├── beginner.md            # 初级：语法速通（JS→Python 对比）
│   ├── intermediate.md        # 中级：异步编程、装饰器、上下文管理器
│   └── advanced.md            # 高级：元编程、性能优化、C 扩展调用
│
├── 02-llm-fundamentals/        # 第 2 章 · 大语言模型原理
│   ├── index.md
│   ├── beginner.md            # 初级：LLM 是什么、Token、为什么会幻觉
│   ├── intermediate.md        # 中级：Transformer 架构、注意力机制、采样策略
│   └── advanced.md            # 高级：手写注意力、KV Cache、Flash Attention、量化
│
├── 03-prompt-engineering/      # 第 3 章 · Prompt Engineering
│   ├── index.md
│   ├── beginner.md            # 初级：基本原则、System Prompt、简单示例
│   ├── intermediate.md        # 中级：CoT/ToT、结构化输出、动态 Few-shot
│   └── advanced.md            # 高级：Prompt 自动优化、A/B 测试、DSPy
│
├── 04-llm-api/                 # 第 4 章 · LLM API 调用
│   ├── index.md
│   ├── beginner.md            # 初级：Claude/OpenAI API 基础调用、多轮对话
│   ├── intermediate.md        # 中级：Streaming、多模态、统一适配层
│   └── advanced.md            # 高级：错误处理体系、速率控制、成本监控
│
├── 05-tool-use/                # 第 5 章 · Tool Use（工具调用）
│   ├── index.md
│   ├── beginner.md            # 初级：什么是 Tool Use、JSON Schema、基本流程
│   ├── intermediate.md        # 中级：多工具协同、工具链、并行调用
│   └── advanced.md            # 高级：动态工具生成、工具优先级、工具结果缓存
│
├── 06-agent-basics/            # 第 6 章 · Agent 核心原理
│   ├── index.md
│   ├── beginner.md            # 初级：ReAct 模式、手写 Agent 循环、对话管理
│   ├── intermediate.md        # 中级：Planning、Reflection、路由分发、人机协同
│   └── advanced.md            # 高级：自适应策略、元认知、Agent 自我改进
│
├── 07-rag/                     # 第 7 章 · RAG（检索增强生成）
│   ├── index.md
│   ├── beginner.md            # 初级：RAG 是什么、基本流程、Chroma 入门
│   ├── intermediate.md        # 中级：分块策略、Embedding、混合检索、Reranking
│   └── advanced.md            # 高级：Agentic RAG、Graph RAG、Self-RAG、评估优化
│
├── 08-memory/                  # 第 8 章 · 记忆系统
│   ├── index.md
│   ├── beginner.md            # 初级：对话历史管理、滑动窗口、简单持久化
│   ├── intermediate.md        # 中级：短期+长期记忆、向量存储、摘要压缩
│   └── advanced.md            # 高级：MemGPT、时间衰减、记忆整合、Generative Agents
│
├── 09-frameworks/              # 第 9 章 · Agent 框架
│   ├── index.md
│   ├── beginner.md            # 初级：为什么用框架、LangChain 基础
│   ├── intermediate.md        # 中级：LangGraph 工作流、CrewAI 多角色、Anthropic SDK
│   └── advanced.md            # 高级：框架源码分析、自建框架、插件系统设计
│
├── 10-multi-agent/             # 第 10 章 · Multi-Agent 系统
│   ├── index.md
│   ├── beginner.md            # 初级：为什么多 Agent、基本架构模式
│   ├── intermediate.md        # 中级：通信机制、任务调度、冲突解决
│   └── advanced.md            # 高级：大规模编排、动态 Agent 生成、Agent 社会模拟
│
├── 11-evaluation/              # 第 11 章 · 评估与测试
│   ├── index.md
│   ├── beginner.md            # 初级：为什么评估难、基本指标、手动测试
│   ├── intermediate.md        # 中级：自动化评估、LLM-as-Judge、基准测试
│   └── advanced.md            # 高级：可观测性系统、Tracing、线上监控、回归测试
│
├── 12-mcp/                     # 第 12 章 · MCP 协议
│   ├── index.md
│   ├── beginner.md            # 初级：MCP 是什么、为什么需要、基本概念
│   ├── intermediate.md        # 中级：开发 MCP Server/Client、工具和资源定义
│   └── advanced.md            # 高级：生态集成、安全、部署、高级 Transport
│
├── 13-security/                # 第 13 章 · 安全与对齐
│   ├── index.md
│   ├── beginner.md            # 初级：Prompt Injection 是什么、基本防护
│   ├── intermediate.md        # 中级：权限控制、沙箱、内容过滤
│   └── advanced.md            # 高级：红队测试、隐私合规、审计系统
│
├── 14-production/              # 第 14 章 · 生产工程化
│   ├── index.md
│   ├── beginner.md            # 初级：从脚本到服务、基本部署
│   ├── intermediate.md        # 中级：架构设计、错误处理、容器化
│   └── advanced.md            # 高级：高可用、灰度发布、成本优化、自动扩缩容
│
├── 15-performance/             # 第 15 章 · 性能优化
│   ├── index.md
│   ├── beginner.md            # 初级：延迟分析、Streaming 基础
│   ├── intermediate.md        # 中级：缓存策略、并发控制、模型路由
│   └── advanced.md            # 高级：语义缓存、预测执行、Fallback 降级
│
├── 16-frontier/                # 第 16 章 · 前沿方向
│   ├── index.md
│   ├── beginner.md            # 初级：了解 Computer Use、Code Agent 等概念
│   ├── intermediate.md        # 中级：实现简单的 Code Agent、Browser Agent
│   └── advanced.md            # 高级：自主学习 Agent、Agent OS、未来趋势
│
├── projects/                   # 实战项目（保留，按难度标注）
│   ├── index.md
│   ├── p1-cli-chatbot/        # 🟢 初级项目
│   ├── p2-tool-agent/         # 🟢 初级项目
│   ├── p3-rag-knowledge/      # 🟡 中级项目
│   ├── p4-multi-agent/        # 🟡 中级项目
│   ├── p5-mcp-server/         # 🔴 高级项目
│   └── p6-full-stack-agent/   # 🔴 高级项目
│
└── appendix/                   # 附录（保留）
    ├── index.md
    ├── api-reference.md
    ├── glossary.md
    ├── resources.md
    └── faq.md
```

## 每章的统一格式

```markdown
# 第 X 章 · 主题名

> 一句话说清这章讲什么

## 本章概览
- 初级：xxx（学完能做到 yyy）
- 中级：xxx（学完能做到 yyy）
- 高级：xxx（学完能做到 yyy）

---

# 初级篇

## 学习目标
...

## 内容
...

## 小结
...

---

# 中级篇

## 学习目标
...

## 内容
...

## 小结
...

---

# 高级篇

## 学习目标
...

## 内容
...

## 小结
...

---

## 参考资源
...
```

## 学习路径对照

| 目标 | 学习内容 | 能做什么 |
|------|---------|---------|
| 入门 Agent | 所有章节的初级部分 | 写简单 Agent（调 API + 工具 + 对话）|
| 胜任开发 | 所有章节的初级+中级 | 用框架搭 RAG、Multi-Agent 系统 |
| 高级工程师 | 全部三个层级 | 生产部署、调优、自建框架、架构设计 |
