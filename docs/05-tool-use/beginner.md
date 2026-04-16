# Tool Use 入门

> **学习目标**：理解 Tool Use 的核心概念，学会用 JSON Schema 定义工具，掌握完整的 8 步调用流程，动手构建一个天气查询工具。

学完本节，你将能够：
- 解释 Tool Use 的工作原理和价值
- 用 JSON Schema 定义一个工具的名称、描述和参数
- 编写完整的 Tool Use 交互代码（定义 -> 调用 -> 执行 -> 返回）
- 构建一个能查天气的简单 Agent

## 为什么需要 Tool Use

LLM 有几个根本性的局限：

**不能访问实时信息。** 你问它"今天北京天气如何"，它只能说"我没有实时数据"。它无法上网搜索，无法查数据库，无法读你的文件。

**不能执行代码。** 它可以写出完美的 Python 代码，但不能真正运行。需要精确计算时，它只能"估算"。

**不能操作外部系统。** 它能告诉你"你应该发一封邮件给张三"，但它自己做不到。

Tool Use 解决这些问题：你定义工具（函数），LLM 决定何时调用、传什么参数，你的程序负责执行。LLM 从一个"只会说"的大脑变成了"能说能做"的助手。

::: warning 关键理解
模型本身**不会执行**任何工具。它只是生成一段 JSON，表达"我想调用某个工具，参数是什么"。实际的执行完全由你的程序负责。
:::

## 用 JSON Schema 定义工具

Tool Use 的第一步是告诉模型"有哪些工具可用"。工具定义使用 JSON Schema 格式：

```typescript
const tool: Anthropic.Tool = {
    name: "get_weather",                // 工具名称（snake_case，动词+名词）
    description: "获取指定城市的当前天气信息，包括温度、天气状况和湿度。",  // 功能描述
    input_schema: {                     // 参数定义
        type: "object",                 // 顶层必须是 object
        properties: {                   // 参数列表
            city: {
                type: "string",         // 参数类型
                description: "城市名称，如：北京、上海"  // 参数描述
            }
        },
        required: ["city"]              // 必填参数
    }
};
```

### 三要素：name、description、input_schema

**name** -- 简洁明确，`snake_case` 格式，动词开头：`get_weather`、`send_email`、`search_documents`。

**description** -- 这是整个定义中**最重要的字段**。模型主要根据它来判断何时使用工具。要写给模型看，不是写给人看的注释：

```typescript
// 差 -- 太简略
description: "搜索功能"

// 好 -- 说清功能、场景、输入、输出
description:
    "在公司内部知识库中搜索文档。" +
    "当用户询问公司政策、产品规范时使用。" +
    "返回相关文档列表，每条包含标题和摘要。"
```

**input_schema** -- 参数的精确定义。常用类型：

| 字段 | 作用 | 示例 |
|------|------|------|
| `type` | 数据类型 | `"string"`, `"number"`, `"boolean"`, `"array"`, `"object"` |
| `description` | 参数说明 | `"日期，格式为 YYYY-MM-DD，如 2026-04-16"` |
| `enum` | 限定可选值 | `["celsius", "fahrenheit"]` |
| `required` | 必填参数列表 | `["city", "date"]` |

::: tip description 写作要点
1. 说明功能：这个工具做什么
2. 说明场景：什么时候应该用它
3. 参数加示例：`"日期，格式 YYYY-MM-DD，如 2026-04-16"`
4. 善用 enum：比在 description 里写"只能是 A、B、C"更可靠
:::

### 参数类型示例

```typescript
{
    type: "object",
    properties: {
        // 字符串
        name: { type: "string", description: "用户姓名" },
        // 数字
        age: { type: "integer", description: "用户年龄，正整数" },
        // 枚举
        language: {
            type: "string",
            enum: ["python", "javascript", "go"],
            description: "编程语言"
        },
        // 数组
        tags: {
            type: "array",
            items: { type: "string" },
            description: "标签列表，如 ['技术', '教程']"
        },
    },
    required: ["name"]
}
```

## 完整的 8 步调用流程

理解了工具定义，接下来看完整的交互过程：

```
步骤1: 定义工具列表
步骤2: 发送用户消息 + 工具列表给 LLM
步骤3: LLM 决策 -- 需要工具？还是直接回答？
步骤4: LLM 返回 tool_use 消息（工具名 + 参数 + id）
步骤5: 程序解析 tool_use 块
步骤6: 程序执行实际函数
步骤7: 将 tool_result 返回给 LLM
步骤8: LLM 根据结果生成最终回答
```

关键信号是 `stop_reason`：
- `"end_turn"` -- 模型直接回答，不需要工具
- `"tool_use"` -- 模型请求调用工具，你需要执行并返回结果

### 完整可运行代码

```typescript
/**
 * Tool Use 完整流程演示
 * 运行前：npm install @anthropic-ai/sdk && export ANTHROPIC_API_KEY="your-key"
 */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// ===== 步骤1: 定义工具 =====
const tools: Anthropic.Tool[] = [
    {
        name: "get_weather",
        description: "获取指定城市的当前天气信息，包括温度、天气状况和湿度。",
        input_schema: {
            type: "object",
            properties: {
                city: {
                    type: "string",
                    description: "城市名称，如：北京、上海、广州"
                }
            },
            required: ["city"]
        }
    },
    {
        name: "calculate",
        description: "执行数学计算，返回精确结果。支持加减乘除和常用数学函数。",
        input_schema: {
            type: "object",
            properties: {
                expression: {
                    type: "string",
                    description: "数学表达式，如 '2 + 3 * 4'、'100 / 7'"
                }
            },
            required: ["expression"]
        }
    }
];

// ===== 步骤6的准备: 定义工具执行函数 =====
function getWeather(city: string): Record<string, unknown> {
    /** 模拟天气查询（实际项目替换为真实 API） */
    const weatherDb: Record<string, Record<string, unknown>> = {
        "北京": { temperature: 22, condition: "晴", humidity: 45 },
        "上海": { temperature: 26, condition: "多云", humidity: 72 },
        "广州": { temperature: 30, condition: "阵雨", humidity: 85 },
    };
    return weatherDb[city] ?? { error: `未找到城市: ${city}` };
}

function calculate(expression: string): Record<string, unknown> {
    /** 简单的数学计算器 */
    try {
        // 仅允许数字和基本运算符，防止任意代码执行
        if (!/^[\d\s+\-*/().%]+$/.test(expression)) {
            return { error: "表达式包含不允许的字符" };
        }
        const result = Function(`"use strict"; return (${expression})`)();
        return { expression, result };
    } catch (e) {
        return { error: `计算失败: ${e}` };
    }
}

const toolMap: Record<string, (...args: any[]) => Record<string, unknown>> = {
    get_weather: (input: { city: string }) => getWeather(input.city),
    calculate: (input: { expression: string }) => calculate(input.expression),
};

function executeTool(name: string, inputData: Record<string, unknown>): string {
    /** 执行工具，返回 JSON 字符串 */
    const func = toolMap[name];
    if (!func) {
        return JSON.stringify({ error: `未知工具: ${name}` });
    }
    const result = func(inputData);
    return JSON.stringify(result);
}


// ===== 主流程: 完整的 Tool Use 循环 =====
async function processQuery(userMessage: string): Promise<string> {
    /** 处理用户查询，自动处理工具调用 */
    console.log(`\n用户: ${userMessage}`);
    const messages: Anthropic.MessageParam[] = [
        { role: "user", content: userMessage }
    ];

    while (true) {
        // 步骤2: 发送消息和工具列表
        const response = await client.messages.create({
            model: "claude-sonnet-4-20250514",
            max_tokens: 1024,
            tools,
            messages,
        });
        console.log(`  stop_reason: ${response.stop_reason}`);

        // 步骤3: 检查 LLM 决策
        if (response.stop_reason === "end_turn") {
            // 不需要工具，直接返回文本
            const text = response.content
                .filter((b): b is Anthropic.TextBlock => b.type === "text")
                .map(b => b.text)
                .join("");
            console.log(`助手: ${text}`);
            return text;
        }

        if (response.stop_reason === "tool_use") {
            // 步骤4&5: 解析 tool_use
            messages.push({ role: "assistant", content: response.content });

            const toolResults: Anthropic.ToolResultBlockParam[] = [];
            for (const block of response.content) {
                if (block.type === "tool_use") {
                    console.log(`  工具调用: ${block.name}(${JSON.stringify(block.input)})`);
                    // 步骤6: 执行工具
                    const result = executeTool(block.name, block.input as Record<string, unknown>);
                    console.log(`  工具结果: ${result}`);
                    // 构建 tool_result
                    toolResults.push({
                        type: "tool_result",
                        tool_use_id: block.id,  // 必须与 tool_use 的 id 匹配
                        content: result,
                    });
                }
            }

            // 步骤7: 返回 tool_result
            messages.push({ role: "user", content: toolResults });
            // 继续循环 -> 步骤8: LLM 处理结果
        }
    }
}

// 主入口
async function main() {
    await processQuery("北京今天天气怎么样？");
    await processQuery("帮我算一下 (15 * 37 + 892) / 4.5");
    await processQuery("北京和上海哪个更热？温差是多少度？");
}

main();
```

运行后你会看到：模型收到天气问题后返回 `stop_reason: "tool_use"`，程序执行 `get_weather`，把结果返回给模型，模型生成最终回答。最后一个问题会触发**两次**工具调用（同时查两个城市），然后可能再调用计算器算温差。

::: warning 四个容易踩的坑
1. **assistant 消息必须完整保留** -- 包括文本块和 tool_use 块，不能只留一部分
2. **tool_use_id 必须匹配** -- tool_result 的 id 要和 tool_use 的 id 完全一致
3. **tool_result 放在 user 角色中** -- Anthropic API 的设计，不是 assistant
4. **结果必须是字符串** -- 即使工具返回对象，也要 `JSON.stringify` 序列化
:::

## 实战：天气查询 Agent

把上面的知识整合成一个完整的天气查询 Agent，支持多轮对话：

```typescript
/** 天气查询 Agent -- 支持多轮对话 */
import Anthropic from "@anthropic-ai/sdk";
import * as readline from "readline";

const client = new Anthropic();

const tools: Anthropic.Tool[] = [
    {
        name: "get_weather",
        description: "获取指定城市的当前天气信息，包括温度、天气状况、湿度、风力。",
        input_schema: {
            type: "object",
            properties: {
                city: { type: "string", description: "城市名称（中文）" }
            },
            required: ["city"]
        }
    },
    {
        name: "get_forecast",
        description: "获取指定城市未来几天的天气预报。当用户问明天、后天、这周天气时使用。",
        input_schema: {
            type: "object",
            properties: {
                city: { type: "string", description: "城市名称（中文）" },
                days: { type: "integer", description: "预报天数，1-7，默认 3" }
            },
            required: ["city"]
        }
    }
];

// 模拟工具实现（实际项目替换为真实天气 API）
function getWeather(city: string): Record<string, unknown> {
    const db: Record<string, { base: number; conds: string[] }> = {
        "北京": { base: 18, conds: ["晴", "多云"] },
        "上海": { base: 22, conds: ["多云", "小雨"] },
        "广州": { base: 28, conds: ["晴", "雷阵雨"] },
    };
    if (!(city in db)) {
        return { error: `暂不支持 '${city}'，支持：${Object.keys(db).join(", ")}` };
    }
    const d = db[city];
    return {
        city,
        temperature: d.base + Math.floor(Math.random() * 7) - 3,
        condition: d.conds[Math.floor(Math.random() * d.conds.length)],
        humidity: Math.floor(Math.random() * 51) + 30,
    };
}

function getForecast(city: string, days: number = 3): Record<string, unknown> {
    if (!["北京", "上海", "广州"].includes(city)) {
        return { error: `暂不支持 '${city}'` };
    }
    const forecast: Record<string, unknown>[] = [];
    const now = new Date();
    for (let i = 0; i < Math.min(days, 7); i++) {
        const date = new Date(now.getTime() + (i + 1) * 86400000);
        const month = String(date.getMonth() + 1).padStart(2, "0");
        const day = String(date.getDate()).padStart(2, "0");
        forecast.push({
            date: `${month}月${day}日`,
            high: Math.floor(Math.random() * 14) + 20,
            low: Math.floor(Math.random() * 11) + 10,
            condition: ["晴", "多云", "小雨"][Math.floor(Math.random() * 3)],
        });
    }
    return { city, forecast };
}

const toolMap: Record<string, (input: any) => Record<string, unknown>> = {
    get_weather: (input: { city: string }) => getWeather(input.city),
    get_forecast: (input: { city: string; days?: number }) => getForecast(input.city, input.days),
};


class WeatherAgent {
    private messages: Anthropic.MessageParam[] = [];
    private system =
        "你是一个友好的天气助手。根据天气给出穿衣、出行建议。" +
        "如果用户没指定城市，礼貌地询问。回答简洁友好。";

    async chat(userInput: string): Promise<string> {
        this.messages.push({ role: "user", content: userInput });

        for (let i = 0; i < 5; i++) {  // 最多 5 轮工具调用
            const response = await client.messages.create({
                model: "claude-sonnet-4-20250514",
                max_tokens: 1024,
                system: this.system,
                tools,
                messages: this.messages,
            });

            if (response.stop_reason === "end_turn") {
                this.messages.push({ role: "assistant", content: response.content });
                return response.content
                    .filter((b): b is Anthropic.TextBlock => b.type === "text")
                    .map(b => b.text)
                    .join("");
            }

            if (response.stop_reason === "tool_use") {
                this.messages.push({ role: "assistant", content: response.content });
                const results: Anthropic.ToolResultBlockParam[] = [];
                for (const block of response.content) {
                    if (block.type === "tool_use") {
                        const func = toolMap[block.name];
                        const result = func
                            ? func(block.input)
                            : { error: "未知工具" };
                        results.push({
                            type: "tool_result",
                            tool_use_id: block.id,
                            content: JSON.stringify(result),
                        });
                    }
                }
                this.messages.push({ role: "user", content: results });
            }
        }

        return "工具调用次数过多，请简化问题。";
    }
}


// 主入口
async function main() {
    const agent = new WeatherAgent();
    console.log("天气助手（输入 quit 退出）");

    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    const ask = (prompt: string): Promise<string> =>
        new Promise(resolve => rl.question(prompt, resolve));

    while (true) {
        const user = (await ask("\n你: ")).trim();
        if (user === "quit") break;
        const reply = await agent.chat(user);
        console.log(`\n助手: ${reply}`);
    }
    rl.close();
}

main();
```

试试这些对话，观察模型如何理解上下文：
- "北京今天天气怎么样？" -- 调用 get_weather
- "明天呢？需要带伞吗？" -- 记住上文是北京，调用 get_forecast
- "上海这周天气如何？" -- 切换城市，调用 get_forecast

## 小结

- **Tool Use 的本质**：LLM 生成结构化调用请求（JSON），程序负责执行，结果返回给 LLM
- **三要素**：name（清晰命名）、description（模型决策的关键）、input_schema（参数精确定义）
- **核心流程**：定义工具 -> 发送请求 -> 检查 stop_reason -> 执行工具 -> 返回 tool_result -> 循环直到 end_turn
- **消息历史要完整**：assistant 的 tool_use 和 user 的 tool_result 必须成对、id 必须匹配

## 练习

1. **动手做**：运行天气 Agent，尝试至少 5 轮不同的对话，观察模型如何选择工具和理解上下文。
2. **扩展工具**：添加一个 `get_time` 工具（返回当前时间），测试"北京现在几点了？"。
3. **设计练习**：为以下场景设计工具 Schema（只写定义，不用实现）：发送邮件、翻译文本、创建日程。
4. **思考题**：如果模型返回了两个 tool_use 块（比如同时查北京和上海），你应该顺序执行还是并行执行？各有什么利弊？

## 参考资源

- [Anthropic Tool Use 文档](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) -- 官方完整指南
- [OpenAI Function Calling 文档](https://platform.openai.com/docs/guides/function-calling) -- OpenAI 的实现方式，对比学习
- [JSON Schema 官方文档](https://json-schema.org/learn/getting-started-step-by-step) -- 深入理解 JSON Schema 规范
