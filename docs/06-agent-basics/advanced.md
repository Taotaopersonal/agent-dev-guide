# Agent 高级：自适应与元认知

> **学习目标**：掌握自适应策略选择、元认知机制、Agent 自我改进和高级错误恢复，构建能感知自身能力边界的 Agent。

学完本节，你将能够：
- 让 Agent 根据任务难度动态调整执行策略
- 实现元认知机制（Agent 知道自己不知道什么）
- 设计 Agent 自我改进系统（从历史执行中学习）
- 构建长期执行任务的可靠性保障
- 实现错误恢复与回滚机制

## 自适应策略选择

不同的任务需要不同的处理方式。简单问题直接回答，复杂问题需要规划，模糊问题需要先澄清。自适应 Agent 会先评估任务，再选择策略：

```typescript
/** 自适应策略 Agent -- 根据任务难度选择不同处理方式 */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

interface TaskAssessment {
  complexity: "simple" | "moderate" | "complex";
  clarity: "clear" | "ambiguous";
  risk_level: "low" | "medium" | "high";
  estimated_steps: number;
  strategy: "direct" | "plan_first" | "clarify_first";
}

/** 能根据任务特征自适应调整策略的 Agent */
class AdaptiveAgent {
  private tools: Anthropic.Tool[];
  private toolMap: Record<string, (...args: any[]) => any>;

  constructor(tools: Anthropic.Tool[] = [], toolMap: Record<string, (...args: any[]) => any> = {}) {
    this.tools = tools;
    this.toolMap = toolMap;
  }

  /** 评估任务特征，决定执行策略 */
  async assessTask(task: string): Promise<TaskAssessment> {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 300,
      messages: [{ role: "user", content: `评估以下任务，返回 JSON 格式的评估结果：

任务：${task}

评估维度：
1. complexity: "simple"(可一步完成) / "moderate"(需2-3步) / "complex"(需要规划)
2. clarity: "clear"(意图明确) / "ambiguous"(需要澄清)
3. risk_level: "low"(纯信息查询) / "medium"(修改数据) / "high"(不可逆操作)
4. estimated_steps: 预估需要的工具调用次数(数字)
5. strategy: "direct"(直接执行) / "plan_first"(先规划) / "clarify_first"(先澄清)

只返回 JSON，不要其他内容。` }],
    });

    let text = (response.content[0] as Anthropic.TextBlock).text.trim();
    try {
      if (text.includes("```")) {
        text = text.split("```")[1].replace("json", "").trim();
      }
      return JSON.parse(text);
    } catch {
      return { complexity: "moderate", clarity: "clear",
               risk_level: "low", estimated_steps: 3,
               strategy: "direct" };
    }
  }

  /** 策略1：直接执行（简单任务） */
  async executeDirect(task: string): Promise<string> {
    const messages: Anthropic.MessageParam[] = [{ role: "user", content: task }];
    for (let i = 0; i < 5; i++) {
      const response = await client.messages.create({
        model: "claude-sonnet-4-20250514", max_tokens: 2048,
        tools: this.tools, messages,
      });
      if (response.stop_reason === "end_turn") {
        return response.content
          .filter((b): b is Anthropic.TextBlock => b.type === "text")
          .map((b) => b.text)
          .join("");
      }
      if (response.stop_reason === "tool_use") {
        messages.push({ role: "assistant", content: response.content });
        const results: Anthropic.ToolResultBlockParam[] = [];
        for (const block of response.content) {
          if (block.type === "tool_use") {
            const func = this.toolMap[block.name];
            const result = func ? func(block.input) : { error: "未知工具" };
            results.push({
              type: "tool_result", tool_use_id: block.id,
              content: JSON.stringify(result),
            });
          }
        }
        messages.push({ role: "user", content: results });
      }
    }
    return "执行超时";
  }

  /** 策略2：先规划再执行（复杂任务） */
  async executeWithPlan(task: string): Promise<string> {
    // 规划阶段
    const planResponse = await client.messages.create({
      model: "claude-sonnet-4-20250514", max_tokens: 500,
      messages: [{ role: "user", content: `为以下任务制定 3-5 步执行计划：
任务：${task}
返回 JSON 数组：["步骤1", "步骤2", ...]` }],
    });
    let text = (planResponse.content[0] as Anthropic.TextBlock).text.trim();
    let plan: string[];
    try {
      if (text.includes("```")) {
        text = text.split("```")[1].replace("json", "").trim();
      }
      plan = JSON.parse(text);
    } catch {
      plan = [task];
    }

    // 执行阶段
    let context = "";
    for (let i = 0; i < plan.length; i++) {
      console.log(`  执行步骤 ${i + 1}/${plan.length}: ${plan[i]}`);
      const result = await this.executeDirect(`${plan[i]}\n\n已有上下文：${context}`);
      context += `\n步骤${i + 1}结果: ${result.slice(0, 200)}`;
    }

    return context;
  }

  /** 策略3：先澄清再执行（模糊任务） */
  async clarify(task: string): Promise<string> {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514", max_tokens: 300,
      messages: [{ role: "user", content: `用户的请求不够明确，请生成 2-3 个澄清问题：

用户请求：${task}

以自然语言形式提出问题，帮助理解用户的真实意图。` }],
    });
    return `我需要一些更多信息：\n${(response.content[0] as Anthropic.TextBlock).text}`;
  }

  /** 自适应执行入口 */
  async run(task: string): Promise<string> {
    // 1. 评估任务
    const assessment = await this.assessTask(task);
    const strategy = assessment.strategy ?? "direct";
    console.log(`[评估] 复杂度: ${assessment.complexity}, ` +
                `策略: ${strategy}, ` +
                `预估步骤: ${assessment.estimated_steps}`);

    // 2. 根据策略执行
    if (strategy === "clarify_first") {
      return this.clarify(task);
    } else if (strategy === "plan_first") {
      return this.executeWithPlan(task);
    } else {
      return this.executeDirect(task);
    }
  }
}
```

## 元认知机制

元认知就是"知道自己知道什么、不知道什么"。一个有元认知能力的 Agent 会：
- 承认不确定性而不是瞎编
- 主动请求更多信息
- 评估自己的回答置信度

```typescript
/** 元认知 Agent -- 知道自己的能力边界 */
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

interface CapabilityEvaluation {
  can_handle: boolean;
  confidence: number;
  reasoning: string;
  missing_info: string[];
  alternative_suggestion: string;
}

/** 具备元认知能力的 Agent */
class MetaCognitiveAgent {
  private capabilities: string[];
  private knownLimitations: string[];

  /**
   * @param capabilities - Agent 擅长的领域列表
   */
  constructor(capabilities?: string[]) {
    this.capabilities = capabilities ?? [
      "TypeScript 编程", "数据分析", "文件操作",
      "数学计算", "文本处理",
    ];
    this.knownLimitations = [
      "无法访问互联网实时数据",
      "无法执行耗时超过30秒的操作",
      "无法处理二进制文件",
      "不了解2025年之后的事件",
    ];
  }

  /** 评估自己能否胜任这个任务 */
  async evaluateCapability(task: string): Promise<CapabilityEvaluation> {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 300,
      messages: [{ role: "user", content: `评估你是否能完成以下任务：

任务：${task}

你的能力范围：${JSON.stringify(this.capabilities)}
你的已知局限：${JSON.stringify(this.knownLimitations)}

返回 JSON：
{
    "can_handle": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "评估理由",
    "missing_info": ["缺少的信息1"],
    "alternative_suggestion": "如果无法完成，替代建议"
}` }],
    });

    let text = (response.content[0] as Anthropic.TextBlock).text.trim();
    try {
      if (text.includes("```")) {
        text = text.split("```")[1].replace("json", "").trim();
      }
      return JSON.parse(text);
    } catch {
      return { can_handle: true, confidence: 0.5,
               reasoning: "评估失败", missing_info: [],
               alternative_suggestion: "" };
    }
  }

  /** 带元认知的执行 */
  async run(task: string): Promise<string> {
    // 1. 先评估自己的能力
    const evalResult = await this.evaluateCapability(task);
    console.log(`[元认知] 能否处理: ${evalResult.can_handle}, ` +
                `置信度: ${evalResult.confidence}`);

    // 2. 低置信度时提醒用户
    if (evalResult.confidence < 0.3) {
      return `坦白说，这个任务超出了我的能力范围。\n` +
             `原因: ${evalResult.reasoning}\n` +
             `建议: ${evalResult.alternative_suggestion || "请寻求专业帮助"}`;
    }

    let prefix = "";
    if (evalResult.confidence < 0.7) {
      prefix = `提醒：我对这个任务的置信度是 ` +
               `${Math.round(evalResult.confidence * 100)}%，` +
               `结果可能需要验证。\n\n`;
    }

    // 3. 如果缺少信息，先请求
    if (evalResult.missing_info?.length) {
      return "在开始之前，我需要一些额外信息：\n" +
             evalResult.missing_info.map((info) => `- ${info}`).join("\n");
    }

    // 4. 执行任务（这里简化了执行逻辑）
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514", max_tokens: 2048,
      messages: [{ role: "user", content: task }],
    });
    const result = (response.content[0] as Anthropic.TextBlock).text;
    return prefix + result;
  }
}
```

## Agent 自我改进

让 Agent 从历史执行中学习，逐步优化自己的行为模式：

```typescript
/** 自我改进 Agent -- 从历史执行中学习 */

interface Experience {
  timestamp: string;
  task: string;
  strategy: string;
  actions: string[];
  outcome: string;
  success: boolean;
  feedback: string;
}

/** 经验存储：记录成功和失败的执行历史 */
class ExperienceStore {
  experiences: Experience[] = [];

  /** 记录一次执行经验 */
  record(
    task: string, strategy: string, actions: string[],
    outcome: string, success: boolean, feedback: string = ""
  ): void {
    this.experiences.push({
      timestamp: new Date().toISOString(),
      task,
      strategy,
      actions,
      outcome,
      success,
      feedback,
    });
  }

  /** 找到与当前任务相似的历史经验 */
  getSimilarExperiences(task: string, topK: number = 3): Experience[] {
    // 简单的关键词匹配（生产环境应该用向量相似度）
    const taskWords = new Set(task.toLowerCase().split(/\s+/));
    const scored: [Experience, number][] = [];
    for (const exp of this.experiences) {
      const expWords = new Set(exp.task.toLowerCase().split(/\s+/));
      let overlap = 0;
      for (const w of taskWords) {
        if (expWords.has(w)) overlap++;
      }
      scored.push([exp, overlap]);
    }
    scored.sort((a, b) => b[1] - a[1]);
    return scored
      .filter(([, score]) => score > 0)
      .slice(0, topK)
      .map(([exp]) => exp);
  }

  /** 提取成功的执行模式 */
  getSuccessPatterns(): Experience[] {
    return this.experiences.filter((e) => e.success);
  }

  /** 提取失败的执行模式 */
  getFailurePatterns(): Experience[] {
    return this.experiences.filter((e) => !e.success);
  }
}

/** 能从历史中学习的 Agent */
class SelfImprovingAgent {
  private experienceStore = new ExperienceStore();

  /** 从历史经验中获取建议 */
  getAdviceFromHistory(task: string): string {
    const similar = this.experienceStore.getSimilarExperiences(task);
    if (similar.length === 0) {
      return "没有相关的历史经验。";
    }

    const adviceParts: string[] = [];
    for (const exp of similar) {
      const status = exp.success ? "成功" : "失败";
      adviceParts.push(
        `- 类似任务'${exp.task.slice(0, 50)}': 使用${exp.strategy}策略，` +
        `结果${status}。` +
        (exp.feedback ? `反馈: ${exp.feedback}` : "")
      );
    }

    return "历史经验参考：\n" + adviceParts.join("\n");
  }

  /** 带经验参考的执行 */
  run(task: string): string {
    // 1. 查询历史经验
    const advice = this.getAdviceFromHistory(task);
    console.log(`[经验库] ${advice}`);

    // 2. 结合经验执行任务（这里简化）
    const strategy = "direct"; // 实际中根据经验调整
    const actions: string[] = [];
    const outcome = `完成了任务: ${task}`;
    const success = true;

    // 3. 记录本次执行
    this.experienceStore.record(
      task, strategy,
      actions, outcome,
      success
    );

    return outcome;
  }

  /** 接收用户反馈，更新最近一次执行的记录 */
  receiveFeedback(feedback: string): void {
    const exps = this.experienceStore.experiences;
    if (exps.length > 0) {
      exps[exps.length - 1].feedback = feedback;
      console.log(`[学习] 已记录反馈: ${feedback}`);
    }
  }
}
```

## 错误恢复与回滚

长时间运行的 Agent 不可避免会遇到错误。设计优雅的恢复机制：

```typescript
/** 错误恢复与回滚机制 */

interface Checkpoint {
  step: number;
  state: AgentState;
  timestamp: string;
  description: string;
}

interface AgentState {
  completed_steps: { name: string; result: any; attempt: number }[];
  data: Record<string, any>;
}

interface ErrorLogEntry {
  step: string;
  attempt: number;
  error: string;
  timestamp: string;
}

/** 具备错误恢复能力的 Agent */
class ResilientAgent {
  private maxRetries: number;
  private checkpoints: Checkpoint[] = [];
  private state: AgentState = { completed_steps: [], data: {} };
  private errorLog: ErrorLogEntry[] = [];

  constructor(maxRetries: number = 3) {
    this.maxRetries = maxRetries;
  }

  /** 保存当前状态的检查点 */
  saveCheckpoint(description: string = ""): void {
    const checkpoint: Checkpoint = {
      step: this.state.completed_steps.length,
      state: structuredClone(this.state),
      timestamp: new Date().toISOString(),
      description,
    };
    this.checkpoints.push(checkpoint);
    console.log(`  [检查点] 已保存: ${description}`);
  }

  /** 回滚到之前的检查点 */
  rollback(stepsBack: number = 1): boolean {
    const targetIdx = this.checkpoints.length - stepsBack;
    if (targetIdx < 0) {
      console.log("  [回滚] 没有可用的检查点");
      return false;
    }

    const checkpoint = this.checkpoints[targetIdx];
    this.state = structuredClone(checkpoint.state);
    this.checkpoints = this.checkpoints.slice(0, targetIdx + 1);
    console.log(`  [回滚] 已回滚到: ${checkpoint.description}`);
    return true;
  }

  /** 安全执行一个步骤（带重试和回滚） */
  async executeStepSafely(
    stepName: string,
    stepFunc: (kwargs: Record<string, any>) => any | Promise<any>,
    kwargs: Record<string, any> = {}
  ): Promise<Record<string, any>> {
    this.saveCheckpoint(`开始 ${stepName}`);

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        const result = await stepFunc(kwargs);

        // 验证结果
        if (result?.error) {
          throw new Error(result.error);
        }

        // 成功，记录
        this.state.completed_steps.push({
          name: stepName,
          result,
          attempt: attempt + 1,
        });
        return result;

      } catch (e: any) {
        this.errorLog.push({
          step: stepName,
          attempt: attempt + 1,
          error: String(e),
          timestamp: new Date().toISOString(),
        });
        console.log(`  [错误] ${stepName} 第${attempt + 1}次失败: ${e}`);

        if (attempt < this.maxRetries - 1) {
          // 回滚到步骤开始前的状态
          this.rollback(1);
          console.log(`  [重试] 准备第${attempt + 2}次尝试...`);
        } else {
          console.log(`  [放弃] ${stepName} 达到最大重试次数`);
          return { error: `${stepName} 执行失败: ${e}`,
                   attempts: attempt + 1 };
        }
      }
    }
    return { error: "未知错误" };
  }

  /** 获取恢复报告 */
  getRecoveryReport(): Record<string, any> {
    return {
      completed_steps: this.state.completed_steps.length,
      checkpoints: this.checkpoints.length,
      total_errors: this.errorLog.length,
      errors_by_step: {},
      final_state: this.state,
    };
  }
}

// 使用示例
const agent = new ResilientAgent(3);

// 模拟执行步骤
function unreliableStep(kwargs: Record<string, any>): Record<string, any> {
  /** 模拟不可靠的操作（50%失败率） */
  if (Math.random() < 0.5) {
    throw new Error("网络连接超时");
  }
  return { status: "success", data: kwargs.data };
}

(async () => {
  const result = await agent.executeStepSafely(
    "获取远程数据",
    unreliableStep,
    { data: "important_info" }
  );
  console.log(`最终结果: ${JSON.stringify(result)}`);
  console.log(`恢复报告: ${JSON.stringify(agent.getRecoveryReport(), null, 2)}`);
})();
```

## 小结

- **自适应策略**：先评估任务（复杂度、清晰度、风险），再选择处理策略（直接执行/先规划/先澄清）
- **元认知**：Agent 评估自身能力，低置信度时坦诚承认、高风险时主动请求信息
- **自我改进**：记录执行历史，提取成功/失败模式，为未来任务提供经验参考
- **错误恢复**：检查点保存 + 自动回滚 + 限次重试，确保长期执行的可靠性

::: warning 高级模式的适用场景
这些模式增加了系统复杂度。不要在简单 Agent 中过度使用。
- 自适应策略：工具 > 10 个或任务类型多样时考虑
- 元认知：面向用户的产品、高可靠性要求时使用
- 自我改进：长期运行的 Agent、需要个性化的场景
- 错误恢复：涉及外部 API、文件操作、数据库写入等不可靠操作时必备
:::

## 练习

1. **自适应实验**：给 AdaptiveAgent 输入简单/中等/复杂三个不同难度的任务，验证它是否选择了合理的策略。
2. **元认知测试**：让 MetaCognitiveAgent 处理它明显不擅长的任务（如"帮我设计一个电路板"），观察它的自我评估是否准确。
3. **错误恢复**：扩展 ResilientAgent，实现"部分成功"逻辑 -- 如果 5 步中有 3 步成功 2 步失败，返回已完成的部分而非完全失败。
4. **综合挑战**：组合自适应 + 元认知 + 错误恢复，构建一个"不会崩溃的 Agent" -- 无论输入什么任务都能优雅处理。

## 参考资源

- [Reflexion (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) -- 自我反思 Agent 论文
- [Generative Agents (arXiv:2304.03442)](https://arxiv.org/abs/2304.03442) -- 生成式 Agent，包含记忆和反思机制
- [Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) -- 自我改进的 Agent，从探索中学习技能
- [Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) -- 多路径探索框架
- [Anthropic: Building Effective Agents](https://docs.anthropic.com/en/docs/build-with-claude/agentic) -- 官方最佳实践
