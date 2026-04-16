# 第六章 Agent 核心原理 -- 从聊天机器人到自主智能体

上一章我们学会了 Tool Use，让 LLM 能够调用外部工具。但仅仅"能调用工具"还不够 -- 一个真正的 Agent 需要**自主决策**：分析任务、制定计划、执行操作、观察结果、反思改进，循环往复直到任务完成。

**Agent = LLM + 工具 + 循环 + 自主决策。** 如果说 Tool Use 是给 LLM 装上了手脚，那 Agent 就是给它装上了"主动做事"的意识。普通聊天机器人只能一问一答，而 Agent 能接住一个复杂任务，自己拆解步骤，自己执行，自己检查结果。

这一章从 ReAct 模式讲起，手把手带你写出第一个 Agent 循环，然后深入 Planning、Reflection 等设计模式，最后探讨 Agent 的自适应和元认知机制。

## 本章内容

- [入门篇](./beginner.md) -- ReAct 模式、手写 Agent 循环（while + stop_reason）、对话管理基础、文件管理 Agent 实战
- [进阶篇](./intermediate.md) -- Planning 规划策略、Reflection 自我反思、路由分发、人机协同
- [高级篇](./advanced.md) -- 自适应策略选择、元认知机制、Agent 自我改进、高级错误恢复

## 核心概念速览

| 概念 | 一句话解释 |
|------|-----------|
| Agent | 能自主决策和执行任务的 LLM 应用 |
| ReAct | Reasoning + Acting，交替思考和行动的模式 |
| Agent 循环 | while 循环 + tool_use 检测，Agent 运行的核心引擎 |
| Planning | 先制定完整计划再逐步执行 |
| Reflection | Agent 执行后自我反思、评估、改进 |
| Router | 根据任务类型分发给不同的专业处理链 |
| Human-in-the-Loop | 关键决策点引入人类审批 |
| 元认知 | Agent 知道自己能做什么、不能做什么 |

## 学完你能做到

| 层级 | 能力 |
|------|------|
| 入门 | 理解 Agent 和聊天机器人的区别、手写 ReAct Agent 循环、构建能使用工具的 Agent |
| 进阶 | 实现 Plan-then-Execute 模式、让 Agent 自我反思改进、设计路由分发系统 |
| 高级 | 构建能根据任务难度自适应的 Agent、实现错误恢复与回滚、设计自我改进机制 |

## 前置知识

- 已完成第五章 Tool Use，理解工具定义和调用流程
- 熟悉 Python 面向对象编程（class, 继承）
- 理解异步编程基础（async/await）
