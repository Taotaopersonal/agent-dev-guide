# 评估进阶：自动化流水线与 LLM-as-Judge

::: tip 学习目标
- 掌握 LLM-as-Judge 的原理、实现和已知偏差的缓解方法
- 学会构建结构化的测试集，包含分类、难度和评估标准
- 能够搭建完整的自动化评估流水线：规则检查 + LLM Judge + 报告生成

**学完你能做到：** 用 LLM 自动评估 Agent 输出的质量，构建可复现的基准测试集，搭建一个自动化评估流水线并生成结构化的评估报告。
:::

## LLM-as-Judge：用 LLM 评估 LLM

人工评估质量最高但成本太大——你不可能每次迭代都让人看 1000 条输出。LLM-as-Judge 的思路很直接：让一个（通常更强的）LLM 充当评委，自动评估另一个 LLM 的输出。

核心是设计一个好的评估 Prompt：明确的评分标准、清晰的评分范围、具体的评估维度。

### 通用评估 Prompt

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const JUDGE_PROMPT = `你是一个严格公正的 AI 评估专家。请根据以下标准评估回答的质量。

## 评估任务
- 问题：{question}
- 参考答案（如有）：{reference}
- 待评估的回答：{answer}

## 评分标准（1-5 分）

**5 分 - 优秀**：完全准确，内容全面，表达清晰，有洞察力
**4 分 - 良好**：基本准确，覆盖主要信息，表达较好
**3 分 - 及格**：部分准确，有遗漏但无严重错误
**2 分 - 较差**：有明显错误或严重遗漏
**1 分 - 很差**：大部分错误或完全不相关

## 评估维度
1. **准确性** (Accuracy)：信息是否正确
2. **完整性** (Completeness)：是否覆盖关键信息
3. **清晰度** (Clarity)：表达是否清晰易懂
4. **相关性** (Relevance)：是否围绕问题回答

## 输出格式
返回 JSON：
{
  "scores": {
    "accuracy": 1-5,
    "completeness": 1-5,
    "clarity": 1-5,
    "relevance": 1-5
  },
  "overall_score": 1-5,
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["不足1", "不足2"],
  "reasoning": "评分理由"
}`;
```

### LLM Judge 实现

```typescript
interface JudgeResult {
  overall_score: number;
  scores: Record<string, number>;
  strengths: string[];
  weaknesses: string[];
  reasoning: string;
}

interface CompareResult {
  winner: "A" | "B" | "TIE";
  score_a: number;
  score_b: number;
  reasoning: string;
}

class LLMJudge {
  /** LLM-as-Judge 评估器 */

  private judgeModel: string;

  constructor(judgeModel: string = "claude-sonnet-4-20250514") {
    this.judgeModel = judgeModel;
  }

  async evaluate(
    question: string,
    answer: string,
    reference: string = "无参考答案"
  ): Promise<JudgeResult> {
    /** 评估单个回答 */
    const prompt = JUDGE_PROMPT
      .replace("{question}", question)
      .replace("{reference}", reference)
      .replace("{answer}", answer);

    const response = await client.messages.create({
      model: this.judgeModel,
      max_tokens: 1024,
      messages: [{ role: "user", content: prompt }],
    });
    const text = response.content[0].type === "text" ? response.content[0].text : "";
    return JSON.parse(text) as JudgeResult;
  }

  async compare(
    question: string,
    answerA: string,
    answerB: string
  ): Promise<CompareResult> {
    /** 对比评估两个回答 */
    const response = await client.messages.create({
      model: this.judgeModel,
      max_tokens: 1024,
      messages: [{
        role: "user",
        content: `对比以下两个回答，判断哪个更好。

问题：${question}

回答 A：${answerA}

回答 B：${answerB}

返回 JSON：
{
  "winner": "A" 或 "B" 或 "TIE",
  "score_a": 1-5,
  "score_b": 1-5,
  "reasoning": "判断理由"
}`,
      }],
    });
    const text = response.content[0].type === "text" ? response.content[0].text : "";
    return JSON.parse(text) as CompareResult;
  }
}

// 使用
const judge = new LLMJudge();

const result = await judge.evaluate(
  "解释什么是 RAG",
  "RAG 就是一种搜索技术",
  "RAG（检索增强生成）是将外部知识检索与 LLM 生成结合的技术"
);
console.log(`总分: ${result.overall_score}/5`);
console.log(`优点: ${result.strengths}`);
console.log(`不足: ${result.weaknesses}`);
```

### 针对特定场景的评估

不同场景需要不同的评估标准。RAG 系统要评估"忠实度"（是否忠于检索到的文档），代码生成要评估"正确性"和"效率"。

```typescript
const RAG_JUDGE_PROMPT = `评估以下 RAG 系统的回答质量。

问题：{question}
检索到的文档：{retrieved_docs}
系统回答：{answer}

评估维度：
1. **忠实度** (Faithfulness)：回答是否忠实于文档（不编造信息）
2. **相关性** (Relevance)：回答是否与问题相关
3. **信息完整性** (Completeness)：是否利用了文档中的关键信息
4. **无幻觉** (No Hallucination)：是否包含文档中没有的错误信息

返回 JSON：
{
  "faithfulness": 1-5,
  "relevance": 1-5,
  "completeness": 1-5,
  "hallucination_check": true/false,
  "hallucinated_content": ["幻觉内容1"],
  "overall_score": 1-5,
  "reasoning": "评估理由"
}`;
```

### LLM Judge 的偏差和缓解

::: warning 已知偏差
LLM Judge 并不完美，存在以下已知偏差：
1. **位置偏差**：倾向于选择第一个或最后一个答案
2. **长度偏差**：倾向于给更长的回答更高分
3. **风格偏差**：倾向于给与自己风格相似的回答更高分
4. **自我偏好**：评估自己生成的内容时倾向于给高分
:::

缓解策略：

```typescript
interface RobustCompareResult {
  winner: string;
  confidence: "high" | "low";
  method: string;
  note?: string;
}

interface MultiJudgeResult {
  avg_score: number;
  variance: number;
  all_scores: number[];
  is_reliable: boolean;
}

class RobustLLMJudge {
  /** 带偏差缓解的 LLM Judge */

  private judge: LLMJudge;

  constructor() {
    this.judge = new LLMJudge();
  }

  async compareRobust(
    question: string,
    answerA: string,
    answerB: string
  ): Promise<RobustCompareResult> {
    /** 位置交换法：两次评估，交换 A/B 位置 */
    // 第一次：A 在前
    const result1 = await this.judge.compare(question, answerA, answerB);
    // 第二次：B 在前
    const result2 = await this.judge.compare(question, answerB, answerA);

    // 综合判断——如果交换后结果一致，说明没有位置偏差
    const winner1 = result1.winner;
    const winner2 =
      result2.winner === "B" ? "A" : result2.winner === "A" ? "B" : "TIE";

    if (winner1 === winner2) {
      return { winner: winner1, confidence: "high", method: "position_swap" };
    } else {
      return {
        winner: "TIE",
        confidence: "low",
        method: "position_swap",
        note: "两次评估结果不一致，可能存在位置偏差",
      };
    }
  }

  async evaluateMultiJudge(
    question: string,
    answer: string,
    reference: string = "",
    nJudges: number = 3
  ): Promise<MultiJudgeResult> {
    /** 多 Judge 评估：多次评估取平均 */
    const scores: number[] = [];
    for (let i = 0; i < nJudges; i++) {
      const result = await this.judge.evaluate(question, answer, reference);
      scores.push(result.overall_score);
    }

    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    const variance =
      scores.reduce((sum, s) => sum + (s - avgScore) ** 2, 0) / scores.length;

    return {
      avg_score: Math.round(avgScore * 100) / 100,
      variance: Math.round(variance * 100) / 100,
      all_scores: scores,
      is_reliable: variance < 1.0,
    };
  }
}
```

## 构建测试集

好的测试集是评估的基础。每条用例都应该有明确的分类、难度标注和评估标准。

```typescript
import * as fs from "fs";

enum Difficulty {
  EASY = "easy",
  MEDIUM = "medium",
  HARD = "hard",
}

enum Category {
  FACTUAL = "factual",
  REASONING = "reasoning",
  CREATIVE = "creative",
  TOOL_USE = "tool_use",
  MULTI_STEP = "multi_step",
}

interface TestCase {
  id: string;
  question: string;
  expected_answer: string;
  category: string;
  difficulty: string;
  tags: string[];
  required_tools: string[];
  evaluation_criteria: string[];
}

class TestSuiteBuilder {
  /** 测试集构建器 */

  name: string;
  cases: TestCase[] = [];

  constructor(name: string) {
    this.name = name;
  }

  addCase(params: TestCase): TestSuiteBuilder {
    this.cases.push(params);
    return this;
  }

  save(path: string): void {
    const data = {
      name: this.name,
      total_cases: this.cases.length,
      cases: this.cases,
    };
    fs.writeFileSync(path, JSON.stringify(data, null, 2), "utf-8");
  }

  static load(path: string): TestSuiteBuilder {
    const raw = fs.readFileSync(path, "utf-8");
    const data = JSON.parse(raw);
    const builder = new TestSuiteBuilder(data.name);
    for (const c of data.cases) {
      builder.addCase(c);
    }
    return builder;
  }

  stats(): Record<string, unknown> {
    /** 测试集统计 */
    const byCategory: Record<string, number> = {};
    const byDifficulty: Record<string, number> = {};
    for (const c of this.cases) {
      byCategory[c.category] = (byCategory[c.category] ?? 0) + 1;
      byDifficulty[c.difficulty] = (byDifficulty[c.difficulty] ?? 0) + 1;
    }
    return {
      total: this.cases.length,
      by_category: byCategory,
      by_difficulty: byDifficulty,
    };
  }
}

// 构建示例测试集
const suite = new TestSuiteBuilder("RAG Agent 评估集");

suite.addCase({
  id: "fact_001",
  question: "Python 是什么类型的编程语言？",
  expected_answer: "解释型",
  category: "factual",
  difficulty: "easy",
  tags: ["python", "basic"],
  required_tools: [],
  evaluation_criteria: ["提到'解释型'", "提到'面向对象'（加分）"],
});

suite.addCase({
  id: "reason_001",
  question: "如果一个 API 的 P99 延迟是 200ms，P50 是 50ms，说明了什么？",
  expected_answer: "尾部延迟较高，有少量请求耗时显著增大",
  category: "reasoning",
  difficulty: "medium",
  tags: ["performance"],
  required_tools: [],
  evaluation_criteria: ["提到尾部延迟", "解释 P99 和 P50 的含义"],
});

suite.save("./test_suite.json");
console.log(suite.stats());
```

## 自动化评估流水线

把上面的组件串起来：自动运行测试集、用规则检查 + LLM Judge 评估、生成结构化报告。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

interface EvalResult {
  case_id: string;
  category: string;
  difficulty?: string;
  auto_score: number;
  llm_score: number | null;
  latency_ms?: number;
  actual_answer?: string;
  error: string | null;
}

class EvaluationPipeline {
  /** 自动化评估流水线 */

  private agentFn: (question: string) => Promise<string>;
  private suite: TestSuiteBuilder;

  constructor(
    agentFn: (question: string) => Promise<string>,
    testSuitePath: string
  ) {
    this.agentFn = agentFn;
    this.suite = TestSuiteBuilder.load(testSuitePath);
  }

  async run(useLlmJudge: boolean = true): Promise<Record<string, unknown>> {
    /** 运行完整评估 */
    const results: EvalResult[] = [];

    for (const cas of this.suite.cases) {
      process.stdout.write(
        `评估 [${cas.id}] ${cas.question.slice(0, 40)}... `
      );
      const start = Date.now();

      try {
        const actualAnswer = await this.agentFn(cas.question);
        const latency = Date.now() - start;

        const autoScore = this.autoCheck(actualAnswer, cas);

        let llmScore: number | null = null;
        if (useLlmJudge) {
          llmScore = await this.llmJudge(cas, actualAnswer);
        }

        const result: EvalResult = {
          case_id: cas.id,
          category: cas.category,
          difficulty: cas.difficulty,
          auto_score: autoScore,
          llm_score: llmScore,
          latency_ms: latency,
          actual_answer: actualAnswer.slice(0, 300),
          error: null,
        };
        console.log(`auto=${autoScore.toFixed(1)} llm=${llmScore}`);

        results.push(result);
      } catch (e) {
        const result: EvalResult = {
          case_id: cas.id,
          category: cas.category,
          auto_score: 0,
          llm_score: 0,
          error: String(e),
        };
        console.log(`ERROR: ${e}`);
        results.push(result);
      }
    }

    return this.generateReport(results);
  }

  private autoCheck(actual: string, cas: TestCase): number {
    /** 基于规则的自动检查 */
    const criteria = cas.evaluation_criteria;
    if (!criteria || criteria.length === 0) {
      return cas.expected_answer.toLowerCase().includes(actual.toLowerCase())
        ? 1.0
        : 0.0;
    }

    let score = 0;
    for (const criterion of criteria) {
      const keyTerms = criterion
        .replace("提到", "")
        .replace(/'/g, "")
        .replace("（加分）", "")
        .trim();
      if (actual.toLowerCase().includes(keyTerms.toLowerCase())) {
        score += 1;
      }
    }
    return criteria.length > 0 ? score / criteria.length : 0;
  }

  private async llmJudge(cas: TestCase, actual: string): Promise<number> {
    /** 用 LLM 评估答案质量（0-1 分） */
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 256,
      messages: [{
        role: "user",
        content: `评估以下回答的质量（0-1 分）。

问题：${cas.question}
预期答案要点：${cas.expected_answer}
评估标准：${JSON.stringify(cas.evaluation_criteria)}
实际回答：${actual.slice(0, 500)}

只返回一个 0 到 1 之间的数字：`,
      }],
    });
    try {
      const text =
        response.content[0].type === "text" ? response.content[0].text : "";
      return parseFloat(text.trim());
    } catch {
      return 0.5;
    }
  }

  private generateReport(results: EvalResult[]): Record<string, unknown> {
    /** 生成评估报告 */
    const total = results.length;
    const autoScores = results
      .filter((r) => r.auto_score != null)
      .map((r) => r.auto_score);
    const llmScores = results
      .filter((r) => r.llm_score != null)
      .map((r) => r.llm_score!);
    const latencies = results
      .filter((r) => r.latency_ms != null)
      .map((r) => r.latency_ms!);
    const errors = results.filter((r) => r.error);

    const report: Record<string, unknown> = {
      summary: {
        total_cases: total,
        avg_auto_score:
          autoScores.length > 0
            ? autoScores.reduce((a, b) => a + b, 0) / autoScores.length
            : 0,
        avg_llm_score:
          llmScores.length > 0
            ? llmScores.reduce((a, b) => a + b, 0) / llmScores.length
            : 0,
        avg_latency_ms:
          latencies.length > 0
            ? latencies.reduce((a, b) => a + b, 0) / latencies.length
            : 0,
        error_rate: total > 0 ? errors.length / total : 0,
      },
      by_category: {} as Record<string, unknown>,
      worst_cases: [...results]
        .sort((a, b) => (a.llm_score ?? 0) - (b.llm_score ?? 0))
        .slice(0, 5),
      details: results,
    };

    // 按类别统计
    const categories = [...new Set(results.map((r) => r.category))];
    const byCategory: Record<string, unknown> = {};
    for (const category of categories) {
      const catResults = results.filter((r) => r.category === category);
      const catScores = catResults
        .filter((r) => r.llm_score != null)
        .map((r) => r.llm_score!);
      byCategory[category] = {
        count: catResults.length,
        avg_score:
          catScores.length > 0
            ? catScores.reduce((a, b) => a + b, 0) / catScores.length
            : 0,
      };
    }
    report.by_category = byCategory;

    return report;
  }
}
```

::: warning 评估不是一次性的
评估应该是持续的流程。建议在 CI/CD 中集成自动评估，每次代码变更自动运行回归测试。
:::

## 小结

- LLM-as-Judge 用 LLM 自动评估输出质量，大幅降低评估成本
- 好的评估 Prompt 需要明确的标准、清晰的分级和具体的维度
- LLM Judge 存在位置偏差、长度偏差等，需要用位置交换和多 Judge 缓解
- 测试集要有覆盖性、代表性，每条用例有明确的评估标准
- 自动化评估流水线结合规则检查和 LLM Judge，按类别和难度细分报告

## 练习

1. 设计一个针对"客服回答"的 LLM Judge Prompt，包含专业性、态度、解决问题能力三个维度。
2. 实现位置交换法，对比它与普通对比评估的结果差异。
3. 为一个客服 Agent 构建包含 30 个用例的测试集，覆盖常见问题和边界情况。

## 参考资源

- [Judging LLM-as-a-Judge (arXiv:2306.05685)](https://arxiv.org/abs/2306.05685) -- LLM-as-Judge 系统研究
- [LLM-as-Judge: Position Bias (arXiv:2305.17926)](https://arxiv.org/abs/2305.17926) -- 位置偏差研究
- [G-Eval: NLG Evaluation with GPT-4 (arXiv:2303.16634)](https://arxiv.org/abs/2303.16634) -- G-Eval 论文
- [Anthropic: Using Claude as a Judge](https://docs.anthropic.com/en/docs/build-with-claude/develop-tests) -- Anthropic 评估文档
- [RAGAS: Faithfulness and Relevance Metrics](https://docs.ragas.io/en/latest/concepts/metrics/) -- RAGAS 评估指标
- [OpenAI Evals](https://github.com/openai/evals) -- OpenAI 评估框架
