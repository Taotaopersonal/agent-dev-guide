# 第 3 章 · Prompt Engineering

Prompt 是你与 LLM 之间的"编程语言"。同一个模型，不同 Prompt 的效果可以天差地别。在 Agent 开发中，Prompt 的质量直接决定了 Agent 的能力上限。

## 本章三层结构

### 初级：Prompt 基础

掌握 Prompt 的核心原则（清晰、具体、结构化），学会 System Prompt 设计，能用 Few-shot 示例引导模型。学完能写出稳定可靠的 Prompt。

### 中级：高级推理技巧

深入 Chain-of-Thought、Tree-of-Thought 等推理技巧，掌握结构化输出（JSON/XML），学会动态 Few-shot 和 Self-Consistency。学完能设计复杂的推理 Prompt。

### 高级：Prompt 工程化

建立 Prompt 版本管理和 A/B 测试体系，掌握系统化的 Prompt 调试方法，了解自动化 Prompt 优化和 DSPy。学完能建立完整的 Prompt 工程化体系。

## 学完能做到

| 层级 | 学完你能 |
|------|---------|
| 初级 | 写出稳定的 Prompt，设计 System Prompt，使用 Few-shot |
| 中级 | 设计复杂推理 Prompt，实现结构化输出，处理解析失败 |
| 高级 | 建立 Prompt 版本管理，做 A/B 测试，系统化调试优化 |

## 前置要求

- 能调用 LLM API（Claude 或 OpenAI）
- 了解 Python 基础语法
