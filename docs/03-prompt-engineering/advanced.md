# Prompt Engineering · 高级

::: info 学习目标
- 建立 Prompt 版本管理的体系和方法
- 掌握 A/B 测试量化 Prompt 优劣的实践
- 学会系统化的 Prompt 调试和优化流程
- 了解常见 Prompt 问题的诊断和修复方法
- 学完能建立完整的 Prompt 工程化体系

预计学习时间：2-3 小时
:::

## Prompt 版本管理

Prompt 是代码。既然是代码，就需要版本管理。在 Agent 开发中，Prompt 的变更频率甚至高于业务代码——一个词的增减都可能显著影响输出质量。

```typescript
import { createHash } from "crypto";

interface PromptVersion {
  version: string;
  content: string;
  description: string;
  author: string;
  createdAt: string;
  metrics: Record<string, number>;
  contentHash: string;
}

function createPromptVersion(
  params: Pick<PromptVersion, "version" | "content" | "description" | "author"> &
    Partial<Pick<PromptVersion, "metrics">>
): PromptVersion {
  return {
    ...params,
    createdAt: new Date().toISOString(),
    metrics: params.metrics ?? {},
    contentHash: createHash("md5").update(params.content).digest("hex").slice(0, 8),
  };
}

class PromptRegistry {
  /** Prompt 版本注册表 */
  private versions: Record<string, PromptVersion[]> = {};

  register(name: string, version: PromptVersion): void {
    if (!this.versions[name]) {
      this.versions[name] = [];
    }
    this.versions[name].push(version);
    console.log(`[注册] ${name} v${version.version} (hash: ${version.contentHash})`);
  }

  get(name: string, version: string = "latest"): PromptVersion {
    if (!this.versions[name]) {
      throw new Error(`Prompt '${name}' 不存在`);
    }
    if (version === "latest") {
      return this.versions[name][this.versions[name].length - 1];
    }
    const found = this.versions[name].find((v) => v.version === version);
    if (!found) {
      throw new Error(`版本 '${version}' 不存在`);
    }
    return found;
  }

  history(name: string): Array<Record<string, unknown>> {
    return (this.versions[name] ?? []).map((v) => ({
      version: v.version,
      description: v.description,
      hash: v.contentHash,
      metrics: v.metrics,
    }));
  }
}

// 使用示例
const registry = new PromptRegistry();

registry.register(
  "sentiment",
  createPromptVersion({
    version: "1.0",
    content: "分析以下文本的情感倾向：{text}",
    description: "初始版本，简单指令",
    author: "dev",
  })
);

registry.register(
  "sentiment",
  createPromptVersion({
    version: "1.1",
    content: `分析以下文本的情感倾向。
输出 JSON：{"sentiment": "positive/negative/neutral", "confidence": 0-1}
只输出 JSON。

文本：{text}`,
    description: "增加 JSON 格式约束",
    author: "dev",
  })
);

registry.register(
  "sentiment",
  createPromptVersion({
    version: "2.0",
    content: `分析文本情感，输出 JSON。

示例：
"产品真好用！" -> {"sentiment": "positive", "confidence": 0.95}
"太差了不想用" -> {"sentiment": "negative", "confidence": 0.9}
"还行吧一般般" -> {"sentiment": "neutral", "confidence": 0.7}

文本："{text}"
输出：`,
    description: "增加 3 个 few-shot 示例",
    author: "dev",
  })
);

for (const entry of registry.history("sentiment")) {
  console.log(`  v${entry.version}: ${entry.description} [${entry.hash}]`);
}
```

::: tip 实践建议
在正式项目中，推荐将 Prompt 存储在独立文件中（如 `prompts/v1.0/sentiment.txt`），通过 Git 管理版本。配合 CI 中的自动化评测，每次 Prompt 变更都能看到指标变化。
:::

## A/B 测试方法

Prompt 优化不能靠"感觉"，需要量化对比。A/B 测试的核心是：用同一批测试数据，分别跑两个 Prompt 版本，比较关键指标。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

interface TestCase {
  inputText: string;
  expectedOutput: Record<string, string>;
}

interface TestResult {
  inputText: string;
  output: string;
  parsed: Record<string, unknown> | null;
  isValidJson: boolean;
  matchesExpected: boolean;
  latencyMs: number;
}

async function runPrompt(
  promptTemplate: string,
  testInput: string
): Promise<[string, number]> {
  const prompt = promptTemplate.replace("{text}", testInput);
  const start = Date.now();
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 256,
    messages: [{ role: "user", content: prompt }],
  });
  const latency = Date.now() - start;
  const text =
    response.content[0].type === "text" ? response.content[0].text.trim() : "";
  return [text, latency];
}

function evaluate(
  output: string,
  expected: Record<string, string>
): [boolean, Record<string, unknown> | null] {
  try {
    const parsed = JSON.parse(output);
    const match = Object.entries(expected)
      .filter(([k]) => k !== "confidence")
      .every(([k, v]) => parsed[k] === v);
    return [match, parsed];
  } catch {
    return [false, null];
  }
}

async function abTest(
  promptA: string,
  promptB: string,
  testCases: TestCase[],
  labelA: string = "A",
  labelB: string = "B"
): Promise<Record<string, Record<string, number>>> {
  /** 运行 A/B 测试 */
  const results: Record<string, TestResult[]> = { [labelA]: [], [labelB]: [] };

  for (let i = 0; i < testCases.length; i++) {
    const testCase = testCases[i];
    console.log(
      `测试 ${i + 1}/${testCases.length}: ${testCase.inputText.slice(0, 30)}...`
    );

    for (const [label, prompt] of [
      [labelA, promptA],
      [labelB, promptB],
    ] as const) {
      const [output, latency] = await runPrompt(prompt, testCase.inputText);
      const [match, parsed] = evaluate(output, testCase.expectedOutput);
      results[label].push({
        inputText: testCase.inputText,
        output,
        parsed,
        isValidJson: parsed !== null,
        matchesExpected: match,
        latencyMs: latency,
      });
    }
  }

  // 汇总
  const summary: Record<string, Record<string, number>> = {};
  for (const [label, resList] of Object.entries(results)) {
    const total = resList.length;
    summary[label] = {
      json_valid_rate:
        resList.filter((r) => r.isValidJson).length / total,
      accuracy:
        resList.filter((r) => r.matchesExpected).length / total,
      avg_latency_ms:
        resList.reduce((sum, r) => sum + r.latencyMs, 0) / total,
    };
  }

  console.log("\n=== A/B 测试结果 ===");
  for (const [label, stats] of Object.entries(summary)) {
    console.log(`\n${label}:`);
    console.log(`  JSON 正确率: ${(stats.json_valid_rate * 100).toFixed(0)}%`);
    console.log(`  答案准确率: ${(stats.accuracy * 100).toFixed(0)}%`);
    console.log(`  平均延迟: ${stats.avg_latency_ms.toFixed(0)}ms`);
  }

  return summary;
}

// 运行测试
const testCases: TestCase[] = [
  { inputText: "这个产品太棒了！", expectedOutput: { sentiment: "positive" } },
  { inputText: "垃圾，再也不买了", expectedOutput: { sentiment: "negative" } },
  { inputText: "请问几点关门？", expectedOutput: { sentiment: "neutral" } },
  { inputText: "功能不错但价格太贵", expectedOutput: { sentiment: "mixed" } },
];

const promptV1 = '分析情感，输出 JSON（sentiment 字段）：{text}';
const promptV2 = `分析文本情感，输出 JSON。

示例：
"好评！" -> {"sentiment": "positive"}
"差评" -> {"sentiment": "negative"}

文本："{text}"
输出：`;

abTest(promptV1, promptV2, testCases, "v1-简单指令", "v2-Few-shot");
```

## 常见问题诊断

### 问题一：输出不稳定

同样的输入，多次调用结果差异很大。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

async function stabilityTest(prompt: string, n: number = 5): Promise<string[]> {
  /** 稳定性测试：同一 Prompt 运行多次，观察一致性 */
  const results: string[] = [];
  for (let i = 0; i < n; i++) {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 256,
      temperature: 0,
      messages: [{ role: "user", content: prompt }],
    });
    const text =
      response.content[0].type === "text" ? response.content[0].text.trim() : "";
    results.push(text);
  }

  const unique = new Set(results);
  const consistency = 1 - (unique.size - 1) / Math.max(n - 1, 1);
  console.log(
    `运行 ${n} 次，${unique.size} 种不同结果，一致性: ${(consistency * 100).toFixed(0)}%`
  );
  return results;
}

// 不稳定的 Prompt
console.log("=== 不稳定 ===");
await stabilityTest("这段文字是正面还是负面的：今天天气不错");

// 修复：更具体的指令 + 格式约束
console.log("\n=== 修复后 ===");
await stabilityTest('判断情感，只回复 positive/negative/neutral：\n"今天天气不错"');
```

::: warning 稳定性修复清单
- 设置 `temperature=0`（或较低值）
- 明确指定输出格式（如"只回复一个词"）
- 提供 few-shot 示例锚定输出模式
- 减少开放性指令（"分析一下" -> "从 X/Y/Z 中选一个"）
:::

### 问题二：格式不对

模型输出了内容，但格式不是你要的。

```typescript
// 原因与修复策略

// 原因 1：指令不够强调 -> 开头结尾都强调格式
const fixedV1 = `【重要】只输出 JSON，不要其他内容。

分析文本情感：{text}

JSON 格式：{"sentiment": "...", "score": 0.0}
记住：只输出 JSON，不要解释。`;

// 原因 2：模型习惯用 markdown 包裹 -> 明确禁止
const fixedV2 = `分析文本情感，输出 JSON。
不要使用 markdown 代码块（\`\`\`），直接输出原始 JSON。

文本：{text}`;

// 原因 3：字段名不一致 -> 提供 JSON Schema 或示例
const fixedV3 = `分析文本情感。

输出严格遵循（字段名大小写敏感）：
{"sentiment": "positive 或 negative 或 neutral", "confidence": 0.0到1.0}

文本：{text}
输出：`;
```

### 问题三：遗漏信息

模型回答了一部分，但遗漏了关键信息。解决方法是用检查清单模式：

```typescript
const checklistPrompt = `审查以下代码，你必须检查以下所有方面，每个都要给出结论：

□ 安全性：是否存在 SQL 注入、XSS 等漏洞
□ 性能：是否有 N+1 查询、不必要的循环
□ 错误处理：异常路径是否都有处理
□ 类型安全：类型注解是否完整
□ 可读性：命名是否清晰

代码：
{code}

请对 5 个方面逐一评价，不要跳过任何一项。`;
```

## 系统化 Prompt 优化流程

将 Prompt 优化看作工程流程，而非随意调整。核心是"假设-实验-验证"的迭代循环。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

interface OptimizationStep {
  step: number;
  change: string;
  hypothesis: string;
  prompt: string;
  score: number;
}

class PromptOptimizer {
  /** 系统化 Prompt 优化 */
  private task: string;
  private testCases: Array<{ input: Record<string, string>; expected: string }>;
  private history: OptimizationStep[] = [];

  constructor(
    task: string,
    testCases: Array<{ input: Record<string, string>; expected: string }>
  ) {
    this.task = task;
    this.testCases = testCases;
  }

  async evaluate(prompt: string): Promise<number> {
    /** 评估 Prompt 在测试集上的表现 */
    let correct = 0;
    for (const testCase of this.testCases) {
      const filledPrompt = Object.entries(testCase.input).reduce(
        (p, [k, v]) => p.replace(`{${k}}`, v),
        prompt
      );
      const response = await client.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 256,
        temperature: 0,
        messages: [{ role: "user", content: filledPrompt }],
      });
      const text =
        response.content[0].type === "text" ? response.content[0].text.trim() : "";
      if (text.includes(testCase.expected)) {
        correct += 1;
      }
    }
    return correct / this.testCases.length;
  }

  async iterate(prompt: string, change: string, hypothesis: string): Promise<number> {
    /** 执行一次优化迭代 */
    const score = await this.evaluate(prompt);
    const step: OptimizationStep = {
      step: this.history.length + 1,
      change,
      hypothesis,
      prompt,
      score,
    };
    this.history.push(step);

    const prev = this.history.length > 1 ? this.history[this.history.length - 2].score : 0;
    const delta = score - prev;
    const direction = delta > 0 ? "提升" : delta < 0 ? "下降" : "不变";

    console.log(`Step ${step.step}: ${change}`);
    console.log(`  假设: ${hypothesis}`);
    console.log(`  得分: ${(score * 100).toFixed(0)}% (${direction} ${(Math.abs(delta) * 100).toFixed(0)}%)`);
    return score;
  }

  report(): void {
    /** 输出优化报告 */
    console.log(`\n=== 优化报告 ===`);
    console.log(`任务: ${this.task}`);
    console.log(`测试用例: ${this.testCases.length} 条`);
    console.log(`迭代次数: ${this.history.length}`);
    for (const step of this.history) {
      const bar = "\u2588".repeat(Math.floor(step.score * 20));
      console.log(`  v${step.step} ${bar} ${(step.score * 100).toFixed(0)}% | ${step.change}`);
    }
    const best = this.history.reduce((a, b) => (a.score >= b.score ? a : b));
    console.log(`\n最佳版本: v${best.step} (${(best.score * 100).toFixed(0)}%)`);
  }
}

// 使用示例
const testCases = [
  { input: { text: "太好了！" }, expected: "positive" },
  { input: { text: "垃圾" }, expected: "negative" },
  { input: { text: "还行" }, expected: "neutral" },
  { input: { text: "今天几号" }, expected: "neutral" },
];

const optimizer = new PromptOptimizer("情感分类", testCases);

await optimizer.iterate(
  "分析情感：{text}",
  "基线版本",
  "建立基线"
);

await optimizer.iterate(
  "分析文本情感，只回复 positive/negative/neutral：\n{text}",
  "增加格式约束",
  "限定输出词汇提升格式一致性"
);

await optimizer.iterate(
  `判断情感（positive/negative/neutral）：

"开心" -> positive
"讨厌" -> negative
"可以" -> neutral

"{text}" ->`,
  "增加 few-shot 示例",
  "示例可以锚定分类边界"
);

optimizer.report();
```

## 调试工具与 Checklist

### Anthropic Workbench

Anthropic 提供了 Web 端的 Prompt 调试工具（console.anthropic.com），支持：
- 实时修改 System Prompt 和用户消息
- 对比不同模型/参数的输出
- 查看 Token 用量
- 保存和分享 Prompt 配置

适合快速实验，但正式迭代建议用代码化的评测体系。

### 调试 Checklist

每次调试 Prompt 时，逐项检查：

```typescript
const DEBUG_CHECKLIST = `
Prompt 调试清单：

1. [ ] 指令是否明确？（模型知道要做什么）
2. [ ] 输出格式是否定义？（JSON / 纯文本 / XML）
3. [ ] 是否有 few-shot 示例？（覆盖主要场景）
4. [ ] 约束条件是否完整？（长度、语言、禁止事项）
5. [ ] temperature 是否合适？（确定性任务用 0）
6. [ ] 是否测试了边界情况？（空输入、超长输入）
7. [ ] 输出是否稳定？（同一输入多次结果一致）
8. [ ] token 用量是否合理？（Prompt 不要过长）
`;
console.log(DEBUG_CHECKLIST);
```

## Prompt 优化的经验法则

经过大量实践总结出的经验法则，帮你少走弯路：

```typescript
// 经验 1：最重要的规则放在开头和结尾
// 首因效应 + 近因效应，中间的规则最容易被"忘记"
const effectiveLayout = `【关键规则：只输出 JSON】  ← 开头强调

## 任务
...中间是任务描述和示例...

记住：只输出 JSON，不要解释。  ← 结尾重申`;

// 经验 2：System Prompt 控制在 500-2000 tokens
// 太短缺少约束，太长模型容易遗漏细节

// 经验 3：用 Markdown 标题和列表帮助模型理解层次
// 不要写成一大段连续文本

// 经验 4：否定指令不如肯定指令有效
const bad = "不要输出 markdown 代码块";
const good = "直接输出原始 JSON 文本"; // 告诉模型"做什么"比"不做什么"更有效

// 经验 5：迭代时一次只改一个变量
// 同时改多个变量，无法知道是哪个变化导致了效果提升或下降
```

## DSPy 简介

DSPy 是一个将 Prompt 优化自动化的框架。它的核心思想是：不手写 Prompt，而是定义输入输出的"签名"，让框架自动搜索最优的 Prompt 策略。

```typescript
// DSPy 的核心概念（伪代码展示思路）
// npm install dspy （注意：DSPy 主要是 Python 生态，此处用 TypeScript 伪代码说明思路）

// 1. 定义"签名"——描述输入输出的关系
// interface SentimentClassify {
//     text: string;       // InputField: 需要分类的文本
//     sentiment: string;  // OutputField: 情感：positive/negative/neutral
// }

// 2. 定义"模块"——推理策略
// class SentimentModule extends dspy.Module {
//     private classify: ChainOfThought<SentimentClassify>;
//     constructor() {
//         super();
//         this.classify = new dspy.ChainOfThought<SentimentClassify>();
//     }
//     forward(text: string) {
//         return this.classify({ text });
//     }
// }

// 3. 自动优化——DSPy 会搜索最优的 Prompt
// const optimizer = new dspy.BootstrapFewShot({ metric: accuracyMetric });
// const optimized = optimizer.compile(new SentimentModule(), { trainset: examples });

// DSPy 自动完成的工作：
// - 从训练集中挑选最佳 few-shot 示例
// - 优化指令文本
// - 决定是否使用 CoT
// - 调整 Prompt 结构
```

::: tip DSPy 的适用场景
DSPy 适合已有标注数据集、需要系统化优化 Prompt 的场景。对于快速原型开发，手写 Prompt + 上述迭代方法通常更高效。DSPy 的价值在规模化场景中更加明显——当你有几十个 Prompt 需要维护时，手动优化不可持续。
:::

## 小结

1. **版本管理**：Prompt 是代码，需要版本化追踪，每次变更记录原因和效果
2. **A/B 测试**：用量化数据而非直觉判断 Prompt 优劣，关注准确率、格式正确率和延迟
3. **问题诊断**：输出不稳定、格式不对、遗漏信息各有对应的修复策略
4. **系统优化**：建立"假设-实验-验证"的迭代循环，一次只改一个变量
5. **工程化工具**：Anthropic Workbench 适合快速实验，代码化评测适合持续迭代，DSPy 适合规模化优化

## 练习

1. **版本管理练习**：为"代码翻译"任务（Python -> TypeScript）设计三个递进版本的 Prompt，每个版本解决前一个版本的一个具体问题。记录每个版本的变更原因和预期改善。

2. **A/B 测试练习**：准备 10 条客户评价文本（5 正面、3 负面、2 中性），分别用简单指令版和 Few-shot 版做情感分类，记录准确率、格式正确率和平均延迟，撰写测试报告。

3. **系统优化练习**：选一个你实际工作中的 Prompt 任务（如代码审查、文本摘要、数据提取），使用 `PromptOptimizer` 进行至少 5 轮迭代优化，记录每轮的假设、变更和得分变化。

4. **诊断练习**：故意写一个"差"的 Prompt（模糊指令、无格式约束、无示例），运行 5 次观察问题，然后按调试 Checklist 逐步修复，记录每步改善效果。

## 参考资源

- [Anthropic Prompt Engineering 指南](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering)
- [OpenAI Prompt Engineering 最佳实践](https://platform.openai.com/docs/guides/prompt-engineering)
- [DSPy 框架](https://github.com/stanfordnlp/dspy)
- [Prompt Engineering Guide（社区）](https://www.promptingguide.ai/)
