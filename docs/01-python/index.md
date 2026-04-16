# 第 1 章 · Python 基础

本书第 2-6 章和第 12 章使用 TypeScript，你可以直接从第 2 章开始学习 Agent 核心概念。但第 7-11 章（RAG、记忆、框架、Multi-Agent、评估）和第 13-16 章（安全、生产化、性能、前沿）使用 Python——因为 LangChain、LangGraph、CrewAI 等主流框架只有 Python SDK。

本章帮助有 JavaScript/TypeScript 基础的开发者快速上手 Python，为后续框架章节做准备。**如果你已经熟悉 Python，可以跳过本章。**

## 本章三层结构

### 初级：从 JS 到 Python

面向有 JavaScript 经验但零 Python 基础的开发者。通过 JS vs Python 对比的方式，快速建立语法映射，完成环境搭建，能写出基本的 Python 脚本。

覆盖：为什么选 Python、环境搭建（uv）、变量/函数/数据结构对比、类与异常处理。

### 中级：异步与工程化

在基础语法之上，掌握 Agent 开发中高频使用的进阶特性。重点是异步编程（和 JS 的 Event Loop 对比）、装饰器、上下文管理器、包管理，以及 Pydantic 数据模型。

覆盖：async/await、装饰器、with 语句、虚拟环境、Pydantic 模型定义。

### 高级：FastAPI 与类型系统

具备独立开发生产级 Python 服务的能力。掌握 FastAPI 框架实战、Python 高级类型系统（Protocol、Generic），以及性能分析与优化技巧。

覆盖：FastAPI 完整项目、类型系统进阶、性能分析、生产最佳实践。

## 学完能做到

| 层级 | 学完你能 |
|------|---------|
| 初级 | 读懂 Python 代码，写简单脚本，调用 LLM API |
| 中级 | 写异步服务，用 Pydantic 定义数据模型，管理项目依赖 |
| 高级 | 用 FastAPI 构建生产级 API 服务，理解类型系统设计 |

## 前置要求

- 熟悉 JavaScript ES6+ 语法
- 有 Node.js 开发经验（对比学习效果更好）
- 不需要任何 Python 基础
