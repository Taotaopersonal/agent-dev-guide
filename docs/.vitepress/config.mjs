import { defineConfig } from "vitepress";
import { withMermaid } from "vitepress-plugin-mermaid";

export default withMermaid(defineConfig({
  title: "Agent 开发学习手册",
  description: "从原理到生产：一本书掌握 AI Agent 开发",
  lang: "zh-CN",
  base: "/agent-dev-guide/",
  lastUpdated: true,

  head: [["link", { rel: "icon", href: "/agent-dev-guide/favicon.ico" }]],

  themeConfig: {
    logo: "/logo.svg",
    nav: [
      { text: "首页", link: "/" },
      { text: "导读", link: "/guide" },
      { text: "学习路线图", link: "/roadmap" },
      {
        text: "章节目录",
        items: [
          { text: "第 1 章 · Python 基础", link: "/01-python/" },
          { text: "第 2 章 · LLM 原理", link: "/02-llm-fundamentals/" },
          { text: "第 3 章 · Prompt Engineering", link: "/03-prompt-engineering/" },
          { text: "第 4 章 · LLM API 调用", link: "/04-llm-api/" },
          { text: "第 5 章 · Tool Use", link: "/05-tool-use/" },
          { text: "第 6 章 · Agent 核心原理", link: "/06-agent-basics/" },
          { text: "第 7 章 · RAG", link: "/07-rag/" },
          { text: "第 8 章 · 记忆系统", link: "/08-memory/" },
          { text: "第 9 章 · Agent 框架", link: "/09-frameworks/" },
          { text: "第 10 章 · Multi-Agent", link: "/10-multi-agent/" },
          { text: "第 11 章 · 评估与测试", link: "/11-evaluation/" },
          { text: "第 12 章 · MCP 协议", link: "/12-mcp/" },
          { text: "第 13 章 · 安全与对齐", link: "/13-security/" },
          { text: "第 14 章 · 生产工程化", link: "/14-production/" },
          { text: "第 15 章 · 性能优化", link: "/15-performance/" },
          { text: "第 16 章 · 前沿方向", link: "/16-frontier/" },
        ],
      },
      { text: "实战项目", link: "/projects/" },
      { text: "附录", link: "/appendix/" },
    ],

    sidebar: {
      "/01-python/": [{
        text: "第 1 章 · Python 基础",
        items: [
          { text: "章节概览", link: "/01-python/" },
          { text: "初级：从 JS 到 Python", link: "/01-python/beginner" },
          { text: "中级：异步与工程化", link: "/01-python/intermediate" },
          { text: "高级：FastAPI 与类型系统", link: "/01-python/advanced" },
        ],
      }],
      "/02-llm-fundamentals/": [{
        text: "第 2 章 · 大语言模型原理",
        items: [
          { text: "章节概览", link: "/02-llm-fundamentals/" },
          { text: "初级：LLM 是什么", link: "/02-llm-fundamentals/beginner" },
          { text: "中级：Transformer 架构", link: "/02-llm-fundamentals/intermediate" },
          { text: "高级：深入理解与优化", link: "/02-llm-fundamentals/advanced" },
        ],
      }],
      "/03-prompt-engineering/": [{
        text: "第 3 章 · Prompt Engineering",
        items: [
          { text: "章节概览", link: "/03-prompt-engineering/" },
          { text: "初级：Prompt 基础", link: "/03-prompt-engineering/beginner" },
          { text: "中级：高级技巧", link: "/03-prompt-engineering/intermediate" },
          { text: "高级：自动化与优化", link: "/03-prompt-engineering/advanced" },
        ],
      }],
      "/04-llm-api/": [{
        text: "第 4 章 · LLM API 调用",
        items: [
          { text: "章节概览", link: "/04-llm-api/" },
          { text: "初级：第一次 API 调用", link: "/04-llm-api/beginner" },
          { text: "中级：进阶调用技巧", link: "/04-llm-api/intermediate" },
          { text: "高级：生产级 API 使用", link: "/04-llm-api/advanced" },
        ],
      }],
      "/05-tool-use/": [{
        text: "第 5 章 · Tool Use（工具调用）",
        items: [
          { text: "章节概览", link: "/05-tool-use/" },
          { text: "初级：工具调用基础", link: "/05-tool-use/beginner" },
          { text: "中级：多工具协同", link: "/05-tool-use/intermediate" },
          { text: "高级：高级工具系统", link: "/05-tool-use/advanced" },
        ],
      }],
      "/06-agent-basics/": [{
        text: "第 6 章 · Agent 核心原理",
        items: [
          { text: "章节概览", link: "/06-agent-basics/" },
          { text: "初级：ReAct 与 Agent 循环", link: "/06-agent-basics/beginner" },
          { text: "中级：设计模式", link: "/06-agent-basics/intermediate" },
          { text: "高级：自适应与元认知", link: "/06-agent-basics/advanced" },
        ],
      }],
      "/07-rag/": [{
        text: "第 7 章 · RAG（检索增强生成）",
        items: [
          { text: "章节概览", link: "/07-rag/" },
          { text: "初级：RAG 入门", link: "/07-rag/beginner" },
          { text: "中级：优化检索质量", link: "/07-rag/intermediate" },
          { text: "高级：前沿 RAG 架构", link: "/07-rag/advanced" },
        ],
      }],
      "/08-memory/": [{
        text: "第 8 章 · 记忆系统",
        items: [
          { text: "章节概览", link: "/08-memory/" },
          { text: "初级：对话历史管理", link: "/08-memory/beginner" },
          { text: "中级：短期 + 长期记忆", link: "/08-memory/intermediate" },
          { text: "高级：前沿记忆架构", link: "/08-memory/advanced" },
        ],
      }],
      "/09-frameworks/": [{
        text: "第 9 章 · Agent 框架",
        items: [
          { text: "章节概览", link: "/09-frameworks/" },
          { text: "初级：为什么用框架", link: "/09-frameworks/beginner" },
          { text: "中级：框架深入", link: "/09-frameworks/intermediate" },
          { text: "高级：源码分析与自建", link: "/09-frameworks/advanced" },
        ],
      }],
      "/10-multi-agent/": [{
        text: "第 10 章 · Multi-Agent 系统",
        items: [
          { text: "章节概览", link: "/10-multi-agent/" },
          { text: "初级：多 Agent 入门", link: "/10-multi-agent/beginner" },
          { text: "中级：通信与调度", link: "/10-multi-agent/intermediate" },
          { text: "高级：大规模编排", link: "/10-multi-agent/advanced" },
        ],
      }],
      "/11-evaluation/": [{
        text: "第 11 章 · 评估与测试",
        items: [
          { text: "章节概览", link: "/11-evaluation/" },
          { text: "初级：为什么评估难", link: "/11-evaluation/beginner" },
          { text: "中级：自动化评估", link: "/11-evaluation/intermediate" },
          { text: "高级：可观测性系统", link: "/11-evaluation/advanced" },
        ],
      }],
      "/12-mcp/": [{
        text: "第 12 章 · MCP 协议",
        items: [
          { text: "章节概览", link: "/12-mcp/" },
          { text: "初级：MCP 是什么", link: "/12-mcp/beginner" },
          { text: "中级：开发 MCP Server", link: "/12-mcp/intermediate" },
          { text: "高级：生产部署与安全", link: "/12-mcp/advanced" },
        ],
      }],
      "/13-security/": [{
        text: "第 13 章 · 安全与对齐",
        items: [
          { text: "章节概览", link: "/13-security/" },
          { text: "初级：Prompt Injection 入门", link: "/13-security/beginner" },
          { text: "中级：权限控制与沙箱", link: "/13-security/intermediate" },
          { text: "高级：红队测试与合规", link: "/13-security/advanced" },
        ],
      }],
      "/14-production/": [{
        text: "第 14 章 · 生产工程化",
        items: [
          { text: "章节概览", link: "/14-production/" },
          { text: "初级：从脚本到服务", link: "/14-production/beginner" },
          { text: "中级：架构设计", link: "/14-production/intermediate" },
          { text: "高级：高可用与优化", link: "/14-production/advanced" },
        ],
      }],
      "/15-performance/": [{
        text: "第 15 章 · 性能优化",
        items: [
          { text: "章节概览", link: "/15-performance/" },
          { text: "初级：延迟分析", link: "/15-performance/beginner" },
          { text: "中级：缓存与并发", link: "/15-performance/intermediate" },
          { text: "高级：极致性能", link: "/15-performance/advanced" },
        ],
      }],
      "/16-frontier/": [{
        text: "第 16 章 · 前沿方向",
        items: [
          { text: "章节概览", link: "/16-frontier/" },
          { text: "初级：前沿概念", link: "/16-frontier/beginner" },
          { text: "中级：实现前沿 Agent", link: "/16-frontier/intermediate" },
          { text: "高级：未来趋势", link: "/16-frontier/advanced" },
        ],
      }],
      "/projects/": [{
        text: "实战项目",
        items: [
          { text: "实战项目概览", link: "/projects/" },
          { text: "P1. 命令行聊天机器人 🟢", link: "/projects/p1-cli-chatbot/" },
          { text: "P2. 多工具 Agent 🟢", link: "/projects/p2-tool-agent/" },
          { text: "P3. RAG 知识库 🟡", link: "/projects/p3-rag-knowledge/" },
          { text: "P4. Multi-Agent 团队 🟡", link: "/projects/p4-multi-agent/" },
          { text: "P5. MCP Server 🔴", link: "/projects/p5-mcp-server/" },
          { text: "P6. 全栈 Agent 应用 🔴", link: "/projects/p6-full-stack-agent/" },
        ],
      }],
      "/appendix/": [{
        text: "附录",
        items: [
          { text: "附录概览", link: "/appendix/" },
          { text: "A. 常用 API 速查", link: "/appendix/api-reference" },
          { text: "B. 术语表", link: "/appendix/glossary" },
          { text: "C. 推荐资源", link: "/appendix/resources" },
          { text: "D. 常见问题", link: "/appendix/faq" },
        ],
      }],
    },

    socialLinks: [
      { icon: "github", link: "https://github.com/taotaopersonal/agent-dev-guide" },
    ],
    search: { provider: "local" },
    outline: { level: [2, 3], label: "目录" },
    lastUpdated: { text: "最后更新" },
    docFooter: { prev: "上一篇", next: "下一篇" },
    footer: {
      message: "Agent 开发学习手册 — 从原理到生产的完整技术书籍",
      copyright: "Copyright © 2026",
    },
  },
  mermaid: {
    theme: "base",
    themeVariables: {
      fontSize: "18px",
      fontFamily: "system-ui, -apple-system, sans-serif",
      nodeTextSize: "16px",
    },
  },
}));
