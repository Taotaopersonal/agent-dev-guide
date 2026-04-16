# 前沿方向 · 进阶篇

::: tip 学习目标
- 理解 Computer Use 的实现原理：截图-分析-操作循环
- 掌握 Anthropic Computer Use API 的使用方式
- 理解 Code Agent 的架构和核心工具设计
- 实现一个简单的测试驱动 Code Agent
:::

::: info 学完你能做到
- 用 Anthropic API 实现一个能操作电脑的 Agent
- 设计 Code Agent 的搜索、读取、编辑、执行四类工具
- 理解 Claude Code "精确替换而非重写" 的设计思想
- 构建一个能自动修 Bug 并运行测试的 Code Agent
:::

## Computer Use 实现原理

### 核心循环：截图-分析-操作

```typescript
// computer_use_loop.ts — Computer Use 核心循环

import Anthropic from "@anthropic-ai/sdk";
import { execSync } from "child_process";
import * as fs from "fs";

const client = new Anthropic();

function takeScreenshot(): string {
  /** 截取屏幕截图，返回 base64 编码 */
  const path = "/tmp/screenshot.png";
  execSync(`screencapture -x ${path}`);
  const buffer = fs.readFileSync(path);
  return buffer.toString("base64");
}

function executeAction(action: Record<string, any>): boolean {
  /** 执行 AI 返回的操作指令 */
  const actionType = action.type;

  if (actionType === "click") {
    const { x, y } = action;
    execSync(`cliclick c:${x},${y}`);
  } else if (actionType === "type") {
    execSync(`cliclick t:${action.text}`);
  } else if (actionType === "key") {
    execSync(`cliclick kp:${action.key}`);
  } else if (actionType === "done") {
    return true; // 任务完成
  }

  return false;
}

async function computerUseLoop(
  task: string,
  maxSteps: number = 20
): Promise<string> {
  /** Computer Use 主循环 */
  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: task },
  ];

  for (let step = 0; step < maxSteps; step++) {
    const screenshotB64 = takeScreenshot();

    messages.push({
      role: "user",
      content: [
        {
          type: "image",
          source: {
            type: "base64",
            media_type: "image/png",
            data: screenshotB64,
          },
        },
        { type: "text", text: "当前屏幕截图，请决定下一步操作。" },
      ],
    });

    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      system: `你是电脑操作助手。返回 JSON 格式的操作指令。
可用操作：
- {"type": "click", "x": 100, "y": 200}
- {"type": "type", "text": "hello"}
- {"type": "key", "key": "enter"}
- {"type": "done"}
只返回一个 JSON 对象。`,
      messages,
    });

    try {
      const block = response.content[0];
      if (block.type === "text") {
        const action = JSON.parse(block.text);
        if (executeAction(action)) {
          return "任务完成";
        }
      }
    } catch {
      // JSON 解析失败，继续下一步
    }

    await new Promise((r) => setTimeout(r, 500));
  }

  return "达到最大步数";
}
```

### Anthropic 官方 Computer Use API

Anthropic 把鼠标、键盘、截图操作封装成了标准的 Tool，模型可以直接调用：

```typescript
// anthropic_computer_use.ts — 官方 Computer Use API

import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// 三个内置工具
const COMPUTER_USE_TOOLS: Anthropic.Messages.Tool[] = [
  {
    type: "computer_20250124", // 电脑操控
    name: "computer",
    display_width_px: 1920,
    display_height_px: 1080,
    display_number: 0,
  },
  {
    type: "bash_20250124", // 终端命令
    name: "bash",
  },
  {
    type: "text_editor_20250124", // 文本编辑
    name: "str_replace_editor",
  },
];

function executeComputerTool(
  toolName: string,
  toolInput: Record<string, any>
): string {
  /**
   * 执行 Computer Use 工具调用
   *
   * 需根据 toolName 分发到对应的处理逻辑：
   * - "computer": 执行鼠标/键盘操作并返回截图
   * - "bash": 执行终端命令并返回输出
   * - "str_replace_editor": 执行文件编辑操作
   */
  // 这里需要实现具体的工具执行逻辑
  // 参考 Anthropic Computer Use Demo: https://github.com/anthropics/anthropic-quickstarts
  throw new Error(`请实现 ${toolName} 的执行逻辑`);
}

async function runComputerUseTask(
  task: string,
  maxSteps: number = 30
): Promise<string> {
  /** 使用官方 API 执行 Computer Use 任务 */
  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: task },
  ];

  for (let step = 0; step < maxSteps; step++) {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 4096,
      tools: COMPUTER_USE_TOOLS,
      messages,
    });

    if (response.stop_reason !== "tool_use") {
      return response.content
        .filter((b): b is Anthropic.Messages.TextBlock => b.type === "text")
        .map((b) => b.text)
        .join("");
    }

    // 处理工具调用
    messages.push({ role: "assistant", content: response.content });
    const toolResults: Anthropic.Messages.ToolResultBlockParam[] = [];

    for (const block of response.content) {
      if (block.type === "tool_use") {
        const result = executeComputerTool(block.name, block.input as Record<string, any>);
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: result,
        });
      }
    }

    messages.push({ role: "user", content: toolResults });
  }

  return "达到最大步数";
}
```

::: warning 安全提醒
Computer Use 让 AI 直接控制你的电脑，务必在 Docker 沙箱中运行：

```yaml
services:
  computer-use:
    image: ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest
    ports:
      - "8501:8501"
      - "6080:6080"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
```
:::

## Code Agent 架构与工具设计

### 四类核心工具

```typescript
// code_tools.ts — Code Agent 的四类核心工具

import { execSync } from "child_process";
import * as fs from "fs";

function grepSearch(
  pattern: string,
  path: string = ".",
  fileType: string = ""
): string {
  /** 在代码库中搜索文本模式 */
  const cmd = ["rg", "--no-heading", "-n", "--max-count", "50"];
  if (fileType) {
    cmd.push("--type", fileType);
  }
  cmd.push(pattern, path);

  try {
    const result = execSync(cmd.join(" "), {
      encoding: "utf-8",
      timeout: 10000,
    });
    return result.slice(0, 5000) || `未找到匹配 '${pattern}' 的内容`;
  } catch (e: any) {
    if (e.killed) return "搜索超时";
    return e.stdout?.slice(0, 5000) || `未找到匹配 '${pattern}' 的内容`;
  }
}

function readFile(filePath: string): string {
  /** 读取文件内容（带行号） */
  try {
    const content = fs.readFileSync(filePath, "utf-8");
    const lines = content.split("\n");
    const numbered = lines.map((line, i) => `${i + 1}\t${line}`);
    const result = numbered.join("\n");
    if (result.length > 10000) {
      return result.slice(0, 10000) + `\n... (截断，共 ${lines.length} 行)`;
    }
    return result;
  } catch {
    return `文件不存在: ${filePath}`;
  }
}

function editFile(
  filePath: string,
  oldString: string,
  newString: string
): string {
  /**
   * 精确替换文件中的代码段
   *
   * 关键设计：替换而非重写。
   * 必须精确匹配 oldString，如果匹配到多处则报错。
   */
  try {
    const content = fs.readFileSync(filePath, "utf-8");

    if (!content.includes(oldString)) {
      return "错误：未找到要替换的文本。请确认精确匹配。";
    }

    const count = content.split(oldString).length - 1;
    if (count > 1) {
      return `错误：找到 ${count} 处匹配。请提供更多上下文。`;
    }

    const newContent = content.replace(oldString, newString);
    fs.writeFileSync(filePath, newContent, "utf-8");

    return `文件已更新: ${filePath}`;
  } catch (e) {
    return `编辑失败: ${e}`;
  }
}

function runCommand(command: string): string {
  /** 执行终端命令 */
  try {
    const stdout = execSync(command, {
      encoding: "utf-8",
      timeout: 30000,
    });
    let output = stdout.slice(0, 5000);
    output += `\n[退出码: 0]`;
    return output;
  } catch (e: any) {
    if (e.killed) return "命令执行超时（30秒限制）";
    let output = (e.stdout || "").slice(0, 5000);
    if (e.stderr) {
      output += `\n[STDERR]\n${e.stderr.slice(0, 2000)}`;
    }
    output += `\n[退出码: ${e.status}]`;
    return output;
  }
}
```

### "编辑而非重写"的设计哲学

为什么 `edit_file` 用字符串替换而不是重写整个文件？

```typescript
// 方案A：重写整个文件 —— 危险
// AI 需要输出完整文件内容，可能遗漏其他代码、引入格式错误
function writeFile(path: string, content: string): void {
  fs.writeFileSync(path, content); // 100 行的文件全部重写
}

// 方案B：精确替换 —— 安全
// AI 只需要指定要改的那几行，其余代码保持不变
function editFile(path: string, oldString: string, newString: string): void {
  const content = fs.readFileSync(path, "utf-8");
  fs.writeFileSync(path, content.replace(oldString, newString)); // 只改需要改的
}
```

方案 B 的优势：改动范围小、不容易引入新 Bug、Token 消耗少。

### 测试驱动的 Code Agent

```typescript
// tdd_agent.ts — 测试驱动的 Code Agent

import Anthropic from "@anthropic-ai/sdk";

async function testDrivenCodeAgent(
  task: string,
  projectPath: string
): Promise<string> {
  /** 工作流程：理解 -> 修改 -> 测试 -> 如失败则重试 */

  const systemPrompt = `你是专业的 Code Agent，遵循测试驱动开发。

工作目录: ${projectPath}

工作流程：
1. 用 grep_search 搜索相关代码
2. 用 read_file 阅读代码，理解结构
3. 如果没有测试，先写测试
4. 用 edit_file 修改代码（精确替换，不要重写）
5. 用 run_command 运行 pytest 验证
6. 测试失败则分析原因并修复

编辑原则：
- 先 read_file 确认要修改的内容
- 使用 edit_file 做精确替换
- 保持代码风格一致`;

  const client = new Anthropic();
  const tools: Anthropic.Messages.Tool[] = [
    {
      name: "grep_search",
      description: "搜索代码模式",
      input_schema: {
        type: "object" as const,
        properties: {
          pattern: { type: "string" },
          path: { type: "string", default: "." },
        },
        required: ["pattern"],
      },
    },
    {
      name: "read_file",
      description: "读取文件",
      input_schema: {
        type: "object" as const,
        properties: {
          file_path: { type: "string" },
        },
        required: ["file_path"],
      },
    },
    {
      name: "edit_file",
      description: "编辑文件",
      input_schema: {
        type: "object" as const,
        properties: {
          file_path: { type: "string" },
          old_string: { type: "string" },
          new_string: { type: "string" },
        },
        required: ["file_path", "old_string", "new_string"],
      },
    },
    {
      name: "run_command",
      description: "执行命令",
      input_schema: {
        type: "object" as const,
        properties: {
          command: { type: "string" },
        },
        required: ["command"],
      },
    },
  ];

  const messages: Anthropic.MessageParam[] = [
    { role: "user", content: task },
  ];

  const toolHandlers: Record<string, (...args: any[]) => string> = {
    grep_search: (input: any) => grepSearch(input.pattern, input.path),
    read_file: (input: any) => readFile(input.file_path),
    edit_file: (input: any) =>
      editFile(input.file_path, input.old_string, input.new_string),
    run_command: (input: any) => runCommand(input.command),
  };

  for (let i = 0; i < 30; i++) {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 4096,
      system: systemPrompt,
      tools,
      messages,
    });

    if (response.stop_reason !== "tool_use") {
      return response.content
        .filter((b): b is Anthropic.Messages.TextBlock => b.type === "text")
        .map((b) => b.text)
        .join("");
    }

    messages.push({ role: "assistant", content: response.content });
    const toolResults: Anthropic.Messages.ToolResultBlockParam[] = [];

    for (const block of response.content) {
      if (block.type === "tool_use") {
        const handler = toolHandlers[block.name];
        const result = handler(block.input);
        toolResults.push({
          type: "tool_result",
          tool_use_id: block.id,
          content: result,
        });
      }
    }

    messages.push({ role: "user", content: toolResults });
  }

  return "达到最大迭代次数";
}
```

## Computer Use 的应用场景

### UI 自动化测试

```typescript
// 用自然语言描述测试步骤，不需要写 CSS selector
const testCases = [
  "打开登录页面，输入用户名 testuser，密码 Test123!，" +
    "点击登录按钮，验证是否跳转到首页",

  "不输入任何内容直接点击提交按钮，" +
    "检查是否显示了必填字段的错误提示",
];

for (const testCase of testCases) {
  const result = await runComputerUseTask(`执行 UI 测试：${testCase}`);
}
```

### RPA：操作没有 API 的系统

```typescript
// 跨系统数据录入
await runComputerUseTask(`
1. 打开桌面上的 data.xlsx 文件
2. 读取 A 列到 D 列的数据
3. 打开 Chrome，访问 http://erp.internal.com
4. 登录并进入数据录入页面
5. 把 Excel 数据逐行填入表单并提交
`);
```

## 小结

两大前沿方向的实现要点：

1. **Computer Use**：截图-分析-操作循环，Anthropic 提供 computer/bash/text_editor 三个内置工具，务必在沙箱中运行
2. **Code Agent**：搜索-读取-编辑-执行四类工具，"精确替换而非重写" 是核心设计原则，测试驱动确保修改质量

## 练习

1. 用 Anthropic Computer Use Docker Demo 完成一个任务：打开浏览器搜索 "Python tutorial"
2. 实现完整的 `edit_file` 工具，测试：正常替换、未找到匹配、多处匹配的三种情况
3. 写一个简单的 Code Agent，给它一个有 Bug 的 Python 文件和测试文件，让它自动修复

## 参考资源

- [Anthropic Computer Use 文档](https://docs.anthropic.com/en/docs/build-with-claude/computer-use) -- 官方 API 文档
- [Anthropic Computer Use Demo](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo) -- Docker 示例
- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) -- CLI 编程工具
- [SWE-bench](https://www.swebench.com/) -- Code Agent 基准测试
- [OpenHands](https://github.com/All-Hands-AI/OpenHands) -- 开源 Code Agent 平台
- [Aider](https://github.com/paul-gauthier/aider) -- AI 结对编程工具
