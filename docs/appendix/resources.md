# 推荐资源

本附录整理了 Agent 开发相关的高质量学习资源，按类型分类，每项资源都附有简要说明。

## 必读论文

这 10 篇论文构成了 Agent 开发的理论基础。建议按顺序阅读，每篇论文后标注了对应的书中章节。

| # | 论文 | 一句话摘要 | 对应章节 |
|---|------|-----------|---------|
| 1 | [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) | 提出 Transformer 架构，奠定所有现代 LLM 的基础。 | 第 2 章 |
| 2 | [Language Models are Few-Shot Learners (GPT-3, 2020)](https://arxiv.org/abs/2005.14165) | 证明了大规模语言模型具有强大的 few-shot 学习能力。 | 第 2 章 |
| 3 | [Chain-of-Thought Prompting (2022)](https://arxiv.org/abs/2201.11903) | 通过让模型逐步推理，显著提升复杂任务的准确率。 | 第 3 章 |
| 4 | [Toolformer (2023)](https://arxiv.org/abs/2302.04761) | 教会语言模型自主决定何时以及如何使用外部工具。 | 第 5 章 |
| 5 | [ReAct: Synergizing Reasoning and Acting (2022)](https://arxiv.org/abs/2210.03629) | 提出 ReAct 模式，交替推理和行动，大幅提升 Agent 表现。 | 第 6 章 |
| 6 | [Retrieval-Augmented Generation (RAG, 2020)](https://arxiv.org/abs/2005.11401) | 提出检索增强生成范式，将外部知识注入语言模型。 | 第 7 章 |
| 7 | [Reflexion: Language Agents with Verbal Reinforcement Learning (2023)](https://arxiv.org/abs/2303.11366) | Agent 通过语言反思实现自我改进，无需梯度更新。 | 第 6 章 |
| 8 | [AutoGen: Enabling Next-Gen LLM Applications (2023)](https://arxiv.org/abs/2308.08155) | 微软提出多 Agent 对话框架，支持灵活的 Agent 协作模式。 | 第 10 章 |
| 9 | [Communicative Agents for Software Development (ChatDev, 2023)](https://arxiv.org/abs/2307.07924) | 多 Agent 协作开发软件的完整系统，展示了 Multi-Agent 的实际应用。 | 第 10 章 |
| 10 | [A Survey on Large Language Model based Autonomous Agents (2023)](https://arxiv.org/abs/2308.11432) | Agent 领域最全面的综述，梳理了架构、能力和应用场景。 | 全书 |

## 官方文档

### Anthropic 生态

| 文档 | 链接 | 说明 |
|------|------|------|
| Anthropic API 文档 | [docs.anthropic.com](https://docs.anthropic.com/) | Claude API 的完整参考 |
| Anthropic Cookbook | [github.com/anthropics/anthropic-cookbook](https://github.com/anthropics/anthropic-cookbook) | 官方代码示例和最佳实践 |
| Anthropic Python SDK | [github.com/anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) | Python SDK 源码 |
| Claude Prompt Engineering | [docs.anthropic.com/.../prompt-engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) | 官方 Prompt 工程指南 |
| Model Context Protocol | [modelcontextprotocol.io](https://modelcontextprotocol.io/) | MCP 协议官方站点 |
| MCP Python SDK | [github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) | MCP Python SDK |

### OpenAI 生态

| 文档 | 链接 | 说明 |
|------|------|------|
| OpenAI API 文档 | [platform.openai.com/docs](https://platform.openai.com/docs/) | GPT 系列 API 完整参考 |
| OpenAI Cookbook | [cookbook.openai.com](https://cookbook.openai.com/) | 官方代码示例 |
| OpenAI Agents SDK | [github.com/openai/openai-agents-python](https://github.com/openai/openai-agents-python) | OpenAI 的 Agent 框架 |

### 框架文档

| 框架 | 链接 | 说明 |
|------|------|------|
| LangChain | [python.langchain.com](https://python.langchain.com/) | 最流行的 LLM 应用框架 |
| LangGraph | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) | 图状态机 Agent 编排框架 |
| LangSmith | [docs.smith.langchain.com](https://docs.smith.langchain.com/) | Agent 可观测性和评估平台 |
| CrewAI | [docs.crewai.com](https://docs.crewai.com/) | 多 Agent 角色扮演框架 |
| LlamaIndex | [docs.llamaindex.ai](https://docs.llamaindex.ai/) | 专注于 RAG 的数据框架 |
| ChromaDB | [docs.trychroma.com](https://docs.trychroma.com/) | 轻量级开源向量数据库 |
| Sentence-Transformers | [sbert.net](https://www.sbert.net/) | 文本 Embedding 模型库 |

## 优质博客和技术文章

### Agent 架构和设计

| 文章 | 链接 | 说明 |
|------|------|------|
| Building effective agents (Anthropic) | [anthropic.com/research/building-effective-agents](https://www.anthropic.com/research/building-effective-agents) | Anthropic 官方的 Agent 构建最佳实践，必读 |
| The Shift from Models to Compound AI Systems (Berkeley) | [bair.berkeley.edu/blog/2024/02/18/compound-ai-systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) | 从单模型到复合 AI 系统的范式转变 |
| What We Learned from a Year of Building with LLMs | [oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms/) | O'Reilly 发布的 LLM 应用开发经验总结 |
| Patterns for Building LLM-based Systems | [eugeneyan.com/writing/llm-patterns](https://eugeneyan.com/writing/llm-patterns/) | Eugene Yan 的 LLM 设计模式总结 |

### RAG 技术

| 文章 | 链接 | 说明 |
|------|------|------|
| Chunking Strategies for LLM Applications | [pinecone.io/learn/chunking-strategies](https://www.pinecone.io/learn/chunking-strategies/) | Pinecone 的文档分块策略深度指南 |
| RAG Best Practices | [docs.llamaindex.ai/en/stable/optimizing/production_rag](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/) | LlamaIndex 的生产级 RAG 最佳实践 |

### 安全

| 文章 | 链接 | 说明 |
|------|------|------|
| OWASP Top 10 for LLM Applications | [owasp.org/www-project-top-10-for-large-language-model-applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | LLM 应用十大安全风险 |
| Prompt Injection 系列 | [simonwillison.net/series/prompt-injection](https://simonwillison.net/series/prompt-injection/) | Simon Willison 的 Prompt 注入攻防系列 |

## 推荐书籍

| 书名 | 作者 | 说明 |
|------|------|------|
| **Building LLMs for Production** | Shin & Vu | 涵盖 LLM 应用从原型到生产的全流程 |
| **Prompt Engineering for Generative AI** | James Phoenix & Mike Taylor | O'Reilly 出版的 Prompt 工程实战书 |
| **Designing Machine Learning Systems** | Chip Huyen | ML 系统设计经典，很多理念适用于 Agent 系统 |
| **Natural Language Processing with Transformers** | Lewis Tunstall 等 | Hugging Face 团队写的 Transformer NLP 实战书 |
| **AI Engineering** | Chip Huyen | 2024 年出版的 AI 工程实践指南 |

## 视频课程

| 课程 | 平台 | 说明 |
|------|------|------|
| [DeepLearning.AI Short Courses](https://www.deeplearning.ai/short-courses/) | DeepLearning.AI | Andrew Ng 平台的短课程，含 LangChain、RAG、Agent 等主题 |
| [Building AI Agents with LangGraph](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) | DeepLearning.AI | LangGraph 官方联合课程 |
| [Full Stack Deep Learning](https://fullstackdeeplearning.com/) | FSDL | 全栈深度学习，涵盖 LLM 应用部署 |
| [Andrej Karpathy - Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | YouTube | 从零理解神经网络和 Transformer |
| [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | YouTube | 可视化讲解神经网络原理 |

## 优质开源项目

### Agent 框架和工具

| 项目 | 链接 | 星标 | 说明 |
|------|------|------|------|
| LangChain | [github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain) | 95k+ | 最流行的 LLM 应用框架 |
| LangGraph | [github.com/langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | 8k+ | 图状态机 Agent 编排 |
| CrewAI | [github.com/crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | 25k+ | 多 Agent 角色协作框架 |
| AutoGen | [github.com/microsoft/autogen](https://github.com/microsoft/autogen) | 35k+ | 微软多 Agent 框架 |
| Haystack | [github.com/deepset-ai/haystack](https://github.com/deepset-ai/haystack) | 18k+ | 生产级 RAG 和 NLP 框架 |

### MCP 生态

| 项目 | 链接 | 说明 |
|------|------|------|
| MCP Servers | [github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) | 官方和社区 MCP Server 合集 |
| MCP Python SDK | [github.com/modelcontextprotocol/python-sdk](https://github.com/modelcontextprotocol/python-sdk) | Python SDK |
| MCP TypeScript SDK | [github.com/modelcontextprotocol/typescript-sdk](https://github.com/modelcontextprotocol/typescript-sdk) | TypeScript SDK |

### 向量数据库

| 项目 | 链接 | 说明 |
|------|------|------|
| ChromaDB | [github.com/chroma-core/chroma](https://github.com/chroma-core/chroma) | 轻量嵌入式向量数据库 |
| Qdrant | [github.com/qdrant/qdrant](https://github.com/qdrant/qdrant) | 高性能向量搜索引擎（Rust） |
| Milvus | [github.com/milvus-io/milvus](https://github.com/milvus-io/milvus) | 企业级向量数据库 |

## 社区和论坛

| 社区 | 链接 | 说明 |
|------|------|------|
| Anthropic Discord | [discord.gg/anthropic](https://discord.gg/anthropic) | Anthropic 官方 Discord，技术讨论活跃 |
| LangChain Discord | [discord.gg/langchain](https://discord.gg/langchain) | LangChain 社区讨论 |
| r/LocalLLaMA | [reddit.com/r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) | Reddit 本地 LLM 社区 |
| Hacker News | [news.ycombinator.com](https://news.ycombinator.com/) | 技术新闻，AI/Agent 话题热门 |
| AI Engineer Foundation | [ai.engineer](https://www.ai.engineer/) | AI 工程师社区和年度会议 |

## 周报和 Newsletter

| Newsletter | 链接 | 说明 |
|-----------|------|------|
| The Batch (Andrew Ng) | [deeplearning.ai/the-batch](https://www.deeplearning.ai/the-batch/) | Andrew Ng 的每周 AI 新闻 |
| Ahead of AI (Sebastian Raschka) | [magazine.sebastianraschka.com](https://magazine.sebastianraschka.com/) | 深入的 AI 技术分析 |
| AI News (Buttondown) | [buttondown.email/ainews](https://buttondown.email/ainews) | AI 领域每日新闻聚合 |
| Latent Space | [latent.space](https://www.latent.space/) | AI 工程师深度访谈播客和 Newsletter |

## 参考资源

- [Awesome LLM (GitHub)](https://github.com/Hannibal046/Awesome-LLM) -- LLM 相关资源的大型汇总列表
- [Awesome AI Agents (GitHub)](https://github.com/e2b-dev/awesome-ai-agents) -- AI Agent 项目和资源合集
- [Papers with Code](https://paperswithcode.com/) -- 论文和代码的对应平台
