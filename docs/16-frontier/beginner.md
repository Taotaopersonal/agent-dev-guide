# 前沿方向 · 入门篇

::: tip 学习目标
- 理解 Computer Use（让 AI 操作电脑）的概念和意义
- 理解 Code Agent（让 AI 写代码）的概念和工作方式
- 了解这些前沿方向为什么重要，以及它们目前的局限性
:::

::: info 学完你能做到
- 向同事清楚解释 Computer Use 和 Code Agent 是什么
- 判断你的业务场景适不适合用这些技术
- 对 Agent 技术的未来趋势有基本的认知
:::

## Computer Use：让 AI 操作你的电脑

### 从 API 调用到界面操作

前面章节讲的工具调用（Tool Use），本质上是让 AI 调用写好的函数——搜索 API、数据库查询、发消息。这些都是"程序员路线"，需要提前写好接口。

Computer Use 走的是另一条路——"普通用户路线"：AI 直接看到你的屏幕画面，然后像人一样点击鼠标、敲键盘。

```
传统 Tool Use:                Computer Use:
用户 → AI → search()          用户 → AI → [看屏幕] → [点鼠标] → [打字]
      ← 结果 ←                       ← 看到结果 ←
```

打个比方：Tool Use 是给 AI 一把特定的钥匙开特定的门，Computer Use 是教 AI 自己去找门、找钥匙、开门。

### 它是怎么工作的

Computer Use 的核心是一个不断重复的循环：

```
截取屏幕截图
    ↓
把截图发给 AI 模型
    ↓
AI 分析截图，决定下一步操作（点击某个按钮、输入文字、滚动页面...）
    ↓
执行操作
    ↓
回到第一步，截取新的屏幕截图
```

就像一个远程桌面操作员——看屏幕、想一想、操作、再看屏幕...

### 什么时候该用

| 场景 | 适合程度 | 原因 |
|------|---------|------|
| 操作没有 API 的老系统 | 非常适合 | 只有 GUI，没有其他选择 |
| 跨多个 GUI 应用的操作 | 适合 | 一个流程涉及多个软件 |
| 快速验证想法 | 适合 | 先跑通再决定是否做 API 对接 |
| 有现成 API 的场景 | 不适合 | API 更快、更准、更便宜 |
| 高频重复任务 | 不适合 | 写脚本或用 Selenium 更可靠 |
| 实时交互场景 | 不适合 | 每一步都要截图 + AI 推理，太慢 |

### 当前的局限

Computer Use 目前还有明显的限制，了解这些才不会踩坑：

- **速度慢**：每一步都要截图（约 1 秒）+ AI 推理（约 2-5 秒），简单操作都要几秒钟
- **坐标不精确**：AI 对屏幕元素的定位有误差，小按钮容易点错
- **成本高**：每张截图约消耗 1000-2000 tokens，一个任务可能需要 10-30 张截图
- **安全风险**：AI 直接控制电脑，误操作不可逆（误删文件、误发邮件）

::: warning 务必在沙箱中运行
Computer Use 让 AI 获得了和你一样的电脑操作权限。Anthropic 官方提供了 Docker 镜像作为安全沙箱：

```yaml
services:
  computer-use:
    image: ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest
    ports:
      - "8501:8501"    # Streamlit UI
      - "6080:6080"    # 浏览器访问虚拟桌面
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
```
:::

## Code Agent：让 AI 成为编程搭档

### 从"写代码"到"做开发"

ChatGPT 和 Claude 早就能写代码了。但"写代码"和"做开发"是两回事。

**写代码**：你告诉 AI "写一个排序函数"，它给你一段代码。

**做开发（Code Agent）**：你说"修复登录页面的验证 bug"，AI 自己去搜索相关文件、阅读代码、理解逻辑、修改代码、运行测试、确认修复——完整的开发流程。

```
用户需求: "修复登录页面的验证 bug"
    |
    v
理解阶段: 搜索相关代码文件 → 阅读代码 → 理解结构
    |
    v
规划阶段: 分析 bug 原因 → 制定修复方案
    |
    v
修改阶段: 编辑代码 → 可能创建新文件
    |
    v
验证阶段: 运行测试 → 检查结果 → 如有问题返回修改
```

### Code Agent 的四类核心工具

Code Agent 不需要几十个工具，四类就够了：

| 工具 | 作用 | 举例 |
|------|------|------|
| **搜索** | 在代码库中找到相关文件和代码 | `grep_search("login", path="src/")` |
| **读取** | 查看文件内容 | `read_file("src/auth.py")` |
| **编辑** | 修改代码（精确替换，不是重写） | `edit_file("src/auth.py", old="...", new="...")` |
| **执行** | 运行命令（测试、lint 等） | `run_command("pytest tests/")` |

注意"编辑"这个工具的设计——**精确替换而不是重写整个文件**。这是 Claude Code 等成熟工具的核心设计思想：改动越小，引入新 bug 的概率越低。

### Claude Code 的设计理念

Claude Code 是 Anthropic 的 CLI 编程工具，它的设计值得学习：

1. **工具精而不多**：搜索、读取、编辑、终端命令四类搞定一切
2. **编辑而非重写**：用字符串替换（old_string -> new_string）修改代码
3. **渐进式理解**：先搜索 → 再阅读 → 最后才修改，像人类开发者一样
4. **项目约定**：通过 CLAUDE.md 让 Agent 理解项目的编码规范和约定

### 什么时候用 Code Agent

| 任务 | 适合程度 | 说明 |
|------|---------|------|
| Bug 修复 | 非常适合 | 搜索定位 + 精确修改 + 测试验证 |
| 代码重构 | 适合 | 理解代码结构后批量修改 |
| 写新功能 | 适合 | 参考已有代码风格来写 |
| 架构设计 | 较弱 | 还是需要人类把控方向 |

## 为什么这些方向重要

### Agent 的能力边界在扩大

回顾 Agent 的进化路径：

```
2023: 文本对话 → 调用 API（Tool Use）
2024: 操作电脑（Computer Use）→ 自主编程（Code Agent）
2025: 自主学习 → Agent 协作
未来: ???
```

每一步进化都让 Agent 能做的事情翻了一倍。Tool Use 让 Agent 能调用几十种 API，Computer Use 让 Agent 能操作任何有界面的软件，Code Agent 让 Agent 能修改自己运行的代码。

### 三个方向的共同趋势

这些前沿方向有一个共同的趋势：**Agent 正在从"工具调用者"进化为"自主行动者"**。

- Tool Use：你告诉 Agent 用哪个工具
- Computer Use：Agent 自己决定怎么操作界面
- Code Agent：Agent 自己搜索、理解、修改代码
- 自主学习（高级篇会讲）：Agent 自己创造新工具

这个趋势意味着什么？对开发者来说，你不再需要为 Agent 预先定义所有能力，Agent 可以自己去"发现"和"学习"新的能力。

## 小结

三个最值得关注的前沿方向：

1. **Computer Use**：AI 操控 GUI 界面，适合没有 API 的遗留系统和跨应用操作，但目前速度慢、成本高
2. **Code Agent**：AI 自主完成完整开发流程（搜索 -> 理解 -> 修改 -> 验证），已经进入生产可用阶段
3. **共同趋势**：Agent 的能力边界在持续扩大，从调用预定义工具到自主行动

## 练习

1. 思考你的工作中有哪些"没有 API 只有 GUI"的系统，列出 3 个可能用 Computer Use 的场景
2. 如果让你设计一个 Code Agent，除了搜索、读取、编辑、执行四类工具，你还会加什么工具？为什么？
3. 尝试用 Anthropic 的 Computer Use Docker Demo 完成一个简单任务（如打开浏览器搜索）

## 参考资源

- [Anthropic Computer Use 文档](https://docs.anthropic.com/en/docs/build-with-claude/computer-use) -- 官方 Computer Use API 文档
- [Claude Code 官方文档](https://docs.anthropic.com/en/docs/claude-code) -- Anthropic CLI 编程工具
- [Computer Use 发布公告](https://www.anthropic.com/news/3-5-sonnet-computer-use) -- Anthropic 博客介绍
- [SWE-bench](https://www.swebench.com/) -- Code Agent 基准测试（看看 AI 修 bug 的能力到哪了）
- [OSWorld Benchmark](https://os-world.github.io/) -- Computer Use 能力评测基准
