# 多 Agent 进阶：通信、调度与共识

::: tip 学习目标
- 掌握三种 Agent 间通信模式：消息传递、共享状态（Blackboard）、事件驱动
- 理解任务分解策略和依赖管理，能实现带优先级的任务调度器
- 掌握投票、仲裁和自我一致性三种共识机制

**学完你能做到：** 为多 Agent 系统选择合适的通信方式，实现一个带依赖管理的任务调度器，以及用分级共识策略处理 Agent 之间的分歧。
:::

## Agent 间通信

上一节我们学了怎么组织多个 Agent，但有一个关键问题没解决：**Agent 之间怎么交换信息？** Supervisor 模式里 Supervisor 把子任务分下去、收回来，这只是最简单的通信方式。当多个 Agent 需要更灵活地协作时，你需要更强大的通信机制。

主要有三种模式，各有适用场景：

| 模式 | 特点 | 适用场景 |
|------|------|---------|
| 消息传递 | 直接发送消息 | 一对一协作，请求-响应 |
| 共享状态（Blackboard） | 通过共享数据空间交互 | 多 Agent 协作，需要共享上下文 |
| 事件驱动 | 发布/订阅事件 | 松耦合系统，异步通知 |

### 消息传递模式

Agent 之间直接发送和接收消息，就像发微信一样——有明确的发送者和接收者。

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

/** Agent 间通信的消息 */
interface Message {
  sender: string;
  receiver: string;
  content: string;
  msgType: string; // "text" | "request" | "response" | "broadcast"
  timestamp: Date;
}

function createMessage(
  sender: string,
  receiver: string,
  content: string,
  msgType: string = "text"
): Message {
  return { sender, receiver, content, msgType, timestamp: new Date() };
}

type MessageHandler = (msg: Message) => string | Promise<string>;

/**
 * 消息总线：Agent 间通信的中介
 *
 * 所有 Agent 都注册到消息总线上，
 * 通过总线发送消息给指定 Agent。
 */
class MessageBus {
  private agents: Record<string, MessageHandler> = {};
  private messageLog: Message[] = [];

  /** 注册一个 Agent 的消息处理函数 */
  register(name: string, handler: MessageHandler): void {
    this.agents[name] = handler;
  }

  /** 发送消息并获取回复 */
  async send(msg: Message): Promise<string> {
    this.messageLog.push(msg);
    if (msg.receiver in this.agents) {
      const response = await this.agents[msg.receiver](msg);
      return response;
    }
    return `Agent '${msg.receiver}' 未找到`;
  }

  /** 广播消息给所有 Agent */
  async broadcast(
    sender: string,
    content: string
  ): Promise<Record<string, string>> {
    const responses: Record<string, string> = {};
    for (const [name, handler] of Object.entries(this.agents)) {
      if (name !== sender) {
        const msg = createMessage(sender, name, content, "broadcast");
        responses[name] = await handler(msg);
      }
    }
    return responses;
  }
}

// 创建 Agent 处理函数
function createAgentHandler(name: string, role: string): MessageHandler {
  return async (msg: Message): Promise<string> => {
    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 512,
      system: `你是${role}。`,
      messages: [
        { role: "user", content: `[来自${msg.sender}]: ${msg.content}` },
      ],
    });
    return response.content[0].type === "text" ? response.content[0].text : "";
  };
}

// 使用示例
const bus = new MessageBus();
bus.register("researcher", createAgentHandler("researcher", "技术研究员"));
bus.register("reviewer", createAgentHandler("reviewer", "审核专家"));

// Agent 之间通过总线通信
const researchResult = await bus.send(
  createMessage("manager", "researcher", "调研 RAG 技术的最新进展")
);

const reviewResult = await bus.send(
  createMessage(
    "manager",
    "reviewer",
    `请审核以下研究结果：\n${researchResult}`
  )
);
```

### 共享状态（Blackboard）模式

所有 Agent 通过一个共享的"黑板"交换信息。任何 Agent 都可以读写黑板，其他 Agent 可以看到更新。

这种模式特别适合多个 Agent 需要基于相同上下文协作的场景——比如协同写文档、多步骤分析。

```typescript
/**
 * 共享黑板：多 Agent 的公共数据空间
 *
 * 类似团队的共享白板：每个人都可以在上面写东西，
 * 其他人随时能看到最新内容。
 */
class Blackboard {
  private data: Record<string, unknown> = {};
  private history: {
    action: string;
    key: string;
    author: string;
    timestamp: string;
  }[] = [];

  /** 写入数据 */
  write(key: string, value: unknown, author: string): void {
    this.data[key] = value;
    this.history.push({
      action: "write",
      key,
      author,
      timestamp: new Date().toISOString(),
    });
  }

  /** 读取数据 */
  read(key: string, defaultValue: unknown = null): unknown {
    return this.data[key] ?? defaultValue;
  }

  /** 读取所有数据 */
  readAll(): Record<string, unknown> {
    return { ...this.data };
  }
}

/** 基于黑板的 Agent */
class BlackboardAgent {
  constructor(
    private name: string,
    private role: string,
    private board: Blackboard
  ) {}

  /** 读取黑板信息，执行任务，将结果写回黑板 */
  async thinkAndAct(task: string): Promise<string> {
    // 读取黑板上的所有信息作为上下文
    const context = this.board.readAll();
    const contextStr = Object.entries(context)
      .map(([k, v]) => `${k}: ${v}`)
      .join("\n");

    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      system: `你是${this.role}。参考黑板上的已有信息完成任务。`,
      messages: [
        {
          role: "user",
          content: `黑板信息：\n${contextStr}\n\n任务：${task}`,
        },
      ],
    });
    const result =
      response.content[0].type === "text" ? response.content[0].text : "";

    // 写回黑板，其他 Agent 就能看到了
    this.board.write(`${this.name}_output`, result, this.name);
    return result;
  }
}

// 使用示例
const board = new Blackboard();

const researcher = new BlackboardAgent("researcher", "研究员", board);
const analyst = new BlackboardAgent("analyst", "分析师", board);
const writer = new BlackboardAgent("writer", "写手", board);

// 多 Agent 通过黑板协作——每个 Agent 都能看到前面 Agent 的输出
board.write("task", "分析 AI Agent 市场趋势", "manager");
await researcher.thinkAndAct("收集市场数据和研究报告");
await analyst.thinkAndAct("分析研究员收集的数据，提取关键趋势");
await writer.thinkAndAct("基于分析结果撰写简报");

console.log(board.read("writer_output"));
```

### 事件驱动通信

Agent 发布事件，其他感兴趣的 Agent 订阅并响应。这是最松耦合的模式——发布者不需要知道谁在监听。

```typescript
interface Event {
  type: string;
  data: Record<string, unknown>;
  source: string;
  timestamp: string;
}

type EventHandler = (event: Event) => string | void;

/** 事件总线：发布-订阅模式 */
class EventBus {
  private subscribers: Record<string, EventHandler[]> = {};
  private eventLog: Event[] = [];

  /** 订阅某类事件 */
  subscribe(eventType: string, handler: EventHandler): void {
    if (!this.subscribers[eventType]) {
      this.subscribers[eventType] = [];
    }
    this.subscribers[eventType].push(handler);
  }

  /** 发布事件，所有订阅者都会收到 */
  publish(
    eventType: string,
    data: Record<string, unknown>,
    source: string
  ): (string | void)[] {
    const event: Event = {
      type: eventType,
      data,
      source,
      timestamp: new Date().toISOString(),
    };
    this.eventLog.push(event);

    const responses: (string | void)[] = [];
    for (const handler of this.subscribers[eventType] ?? []) {
      const result = handler(event);
      if (result) {
        responses.push(result);
      }
    }
    return responses;
  }
}

// 使用示例
const eventBus = new EventBus();

function onResearchComplete(event: Event): string {
  console.log(`[Analyst] 收到研究完成事件，开始分析...`);
  return "分析结果: ...";
}

function onResearchCompleteNotify(event: Event): void {
  console.log(`[Logger] 记录：研究已完成，来源: ${event.source}`);
}

eventBus.subscribe("research_complete", onResearchComplete);
eventBus.subscribe("research_complete", onResearchCompleteNotify);

// 研究员完成后发布事件，分析师和日志器都会响应
eventBus.publish("research_complete", { findings: "..." }, "researcher");
```

::: tip 通信模式选择
- **消息传递**：Agent 关系明确时使用（如 A 请求 B 执行任务）
- **共享状态**：多 Agent 需要共享上下文时使用（如协作写文档）
- **事件驱动**：Agent 之间松耦合时使用（如通知、触发、监控）
- 实际系统中常混合使用多种通信模式
:::

## 任务分解与调度

复杂任务通常需要拆分成多个子任务，并按正确的顺序分配给不同的 Agent。这就涉及两个核心问题：**怎么拆**和**按什么顺序执行**。

### 用 LLM 自动分解任务

```typescript
interface SubTaskDef {
  id: number;
  description: string;
  agent: string;
  dependencies: number[];
  priority: "high" | "medium" | "low";
}

/** 将任务分解为子任务并分配给 Agent */
async function decomposeTask(
  task: string,
  availableAgents: string[]
): Promise<SubTaskDef[]> {
  const agentsDesc = availableAgents.map((a) => `- ${a}`).join("\n");

  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: [
      {
        role: "user",
        content: `将以下任务分解为子任务，并分配给合适的 Agent。

任务：${task}

可用 Agent：
${agentsDesc}

要求：
1. 每个子任务应该是独立可执行的
2. 标明子任务之间的依赖关系
3. 估计每个子任务的优先级

返回 JSON：
{"subtasks": [
  {"id": 1, "description": "子任务描述", "agent": "Agent 名称",
    "dependencies": [], "priority": "high/medium/low"},
]}`,
      },
    ],
  });

  const text =
    response.content[0].type === "text" ? response.content[0].text : "{}";
  return JSON.parse(text).subtasks;
}
```

### 带依赖管理的任务调度器

关键是处理子任务之间的依赖关系——任务 C 依赖任务 A 和 B 的结果，那 C 必须等 A 和 B 都完成才能开始。

```typescript
enum TaskStatus {
  PENDING = "pending",
  RUNNING = "running",
  COMPLETED = "completed",
  FAILED = "failed",
}

/** 子任务定义 */
interface SubTask {
  id: number;
  description: string;
  agent: string;
  dependencies: number[];
  priority: string;
  status: TaskStatus;
  result: string;
}

function createSubTask(params: {
  id: number;
  description: string;
  agent: string;
  dependencies?: number[];
  priority?: string;
}): SubTask {
  return {
    ...params,
    dependencies: params.dependencies ?? [],
    priority: params.priority ?? "medium",
    status: TaskStatus.PENDING,
    result: "",
  };
}

/** 任务调度器：管理子任务的依赖和执行顺序 */
class TaskScheduler {
  private tasks: Map<number, SubTask> = new Map();
  private agentLoad: Record<string, number> = {};

  /** 批量添加子任务 */
  addTasks(
    subtasks: {
      id: number;
      description: string;
      agent: string;
      dependencies?: number[];
      priority?: string;
    }[]
  ): void {
    for (const st of subtasks) {
      const task = createSubTask(st);
      this.tasks.set(task.id, task);
      if (!(task.agent in this.agentLoad)) {
        this.agentLoad[task.agent] = 0;
      }
    }
  }

  /** 获取可以立即执行的任务（所有依赖已满足） */
  getReadyTasks(): SubTask[] {
    const ready: SubTask[] = [];
    for (const task of this.tasks.values()) {
      if (task.status !== TaskStatus.PENDING) continue;

      // 检查所有依赖是否已完成
      const depsMet = task.dependencies.every((depId) => {
        const dep = this.tasks.get(depId);
        return !dep || dep.status === TaskStatus.COMPLETED;
      });

      if (depsMet) {
        ready.push(task);
      }
    }

    // 按优先级排序
    const priorityOrder: Record<string, number> = {
      high: 0,
      medium: 1,
      low: 2,
    };
    ready.sort(
      (a, b) =>
        (priorityOrder[a.priority] ?? 1) - (priorityOrder[b.priority] ?? 1)
    );
    return ready;
  }

  /** 执行单个任务 */
  async executeTask(task: SubTask): Promise<string> {
    task.status = TaskStatus.RUNNING;
    this.agentLoad[task.agent]++;

    try {
      // 收集依赖任务的结果作为上下文
      let depContext = "";
      for (const depId of task.dependencies) {
        const depTask = this.tasks.get(depId);
        if (depTask?.result) {
          depContext += `\n[任务${depId}的结果]: ${depTask.result}`;
        }
      }

      const response = await client.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1024,
        messages: [
          {
            role: "user",
            content: `任务：${task.description}${depContext}`,
          },
        ],
      });
      task.result =
        response.content[0].type === "text" ? response.content[0].text : "";
      task.status = TaskStatus.COMPLETED;
    } catch (e) {
      task.status = TaskStatus.FAILED;
      task.result = `错误: ${e}`;
    } finally {
      this.agentLoad[task.agent]--;
    }

    return task.result;
  }

  /** 执行所有任务（尊重依赖关系） */
  async runAll(): Promise<Record<number, string>> {
    const results: Record<number, string> = {};
    const maxIterations = this.tasks.size * 2;

    for (let i = 0; i < maxIterations; i++) {
      const ready = this.getReadyTasks();
      if (ready.length === 0) {
        const allDone = [...this.tasks.values()].every(
          (t) =>
            t.status === TaskStatus.COMPLETED ||
            t.status === TaskStatus.FAILED
        );
        if (allDone) break;
        continue;
      }

      for (const task of ready) {
        console.log(
          `执行任务 ${task.id}: ${task.description.slice(0, 50)}...`
        );
        const result = await this.executeTask(task);
        results[task.id] = result;
        console.log(`  -> 完成 (状态: ${task.status})`);
      }
    }

    return results;
  }
}

// 使用示例
const scheduler = new TaskScheduler();
scheduler.addTasks([
  {
    id: 1,
    description: "调研 RAG 技术的基本概念",
    agent: "researcher",
    dependencies: [],
    priority: "high",
  },
  {
    id: 2,
    description: "调研 RAG 的最新进展",
    agent: "researcher",
    dependencies: [],
    priority: "high",
  },
  {
    id: 3,
    description: "分析 RAG 的优缺点",
    agent: "analyst",
    dependencies: [1, 2],
    priority: "medium",
  },
  {
    id: 4,
    description: "撰写技术报告",
    agent: "writer",
    dependencies: [3],
    priority: "low",
  },
]);
const results = await scheduler.runAll();
// 任务 1 和 2 没有依赖，可以先执行；
// 任务 3 等 1+2 完成；任务 4 等 3 完成
```

::: warning 并行执行的注意事项
- API 调用有速率限制，过多并行可能触发限流
- 并行任务的错误处理更复杂，需要考虑部分失败的情况
- 共享状态的并发访问需要加锁保护
:::

## 冲突解决与共识

多个 Agent 独立处理同一问题时，输出不一致是常态。原因包括：知识差异（不同上下文）、表述差异（同一结论不同措辞）、实质性分歧（不同判断）。这时候需要**共识机制**。

### 投票机制

最简单的共识方式：让多个 Agent 独立回答，取多数一致的答案。

```typescript
interface Vote {
  answer: string;
  confidence: number;
  reasoning: string;
  agent_id: number;
}

interface ConsensusResult {
  consensus_answer: string;
  consensus_ratio: number;
  avg_confidence: number;
  is_unanimous: boolean;
  all_votes: Vote[];
}

/** 投票式共识机制 */
class VotingConsensus {
  constructor(private nAgents: number = 3) {}

  /** 让多个 Agent 独立投票 */
  async getVotes(
    question: string,
    options?: string[]
  ): Promise<Vote[]> {
    const votes: Vote[] = [];
    for (let i = 0; i < this.nAgents; i++) {
      let prompt = `请回答以下问题。\n问题：${question}\n`;
      if (options) {
        prompt += `选项：${JSON.stringify(options)}\n`;
      }
      prompt +=
        '\n返回 JSON：{"answer": "你的答案", "confidence": 0.0-1.0, "reasoning": "推理过程"}';

      const response = await client.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 512,
        temperature: 0.7, // 稍高温度确保多样性
        messages: [{ role: "user", content: prompt }],
      });
      const text =
        response.content[0].type === "text" ? response.content[0].text : "{}";
      const vote: Vote = { ...JSON.parse(text), agent_id: i };
      votes.push(vote);
    }
    return votes;
  }

  /** 通过投票达成共识 */
  async reachConsensus(
    question: string,
    options?: string[]
  ): Promise<ConsensusResult> {
    const votes = await this.getVotes(question, options);

    // 统计各答案的票数
    const counter: Record<string, number> = {};
    for (const v of votes) {
      counter[v.answer] = (counter[v.answer] ?? 0) + 1;
    }

    // 找出得票最多的答案
    const [majorityAnswer, majorityCount] = Object.entries(counter).sort(
      (a, b) => b[1] - a[1]
    )[0];

    const consensusRatio = majorityCount / votes.length;
    const matchingVotes = votes.filter((v) => v.answer === majorityAnswer);
    const avgConfidence =
      matchingVotes.reduce((sum, v) => sum + v.confidence, 0) /
      Math.max(matchingVotes.length, 1);

    return {
      consensus_answer: majorityAnswer,
      consensus_ratio: consensusRatio,
      avg_confidence: avgConfidence,
      is_unanimous: consensusRatio === 1.0,
      all_votes: votes,
    };
  }
}

// 使用
const voter = new VotingConsensus(5);
const result = await voter.reachConsensus(
  "Python 和 JavaScript 哪个更适合初学者？",
  ["Python", "JavaScript", "都适合"]
);
console.log(
  `共识: ${result.consensus_answer} ` +
    `(${(result.consensus_ratio * 100).toFixed(0)}% 一致)`
);
```

### 仲裁 Agent

当投票无法达成明确共识时，引入一个独立的"仲裁者"来做最终判断。

```typescript
/** 仲裁式共识：由裁判 Agent 做最终判断 */
class ArbitrationConsensus {
  constructor(private nDebaters: number = 3) {}

  /** 收集各 Agent 的观点 */
  async collectOpinions(
    question: string
  ): Promise<{ perspective: string; opinion: string }[]> {
    const opinions: { perspective: string; opinion: string }[] = [];
    const perspectives = ["乐观主义者", "谨慎分析师", "实用主义者"];

    for (let i = 0; i < Math.min(this.nDebaters, perspectives.length); i++) {
      const response = await client.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 512,
        system: `你是一个${perspectives[i]}，请从你的角度分析问题。`,
        messages: [{ role: "user", content: question }],
      });
      opinions.push({
        perspective: perspectives[i],
        opinion:
          response.content[0].type === "text"
            ? response.content[0].text
            : "",
      });
    }
    return opinions;
  }

  /** 裁判做最终判断 */
  async arbitrate(
    question: string,
    opinions: { perspective: string; opinion: string }[]
  ): Promise<{
    opinions: { perspective: string; opinion: string }[];
    arbitration: string;
  }> {
    const opinionsText = opinions
      .map((o) => `[${o.perspective}]:\n${o.opinion}`)
      .join("\n\n");

    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 1024,
      system: "你是一个公正的裁判。综合分析各方观点，给出客观的最终结论。",
      messages: [
        {
          role: "user",
          content:
            `问题：${question}\n\n各方观点：\n${opinionsText}\n\n` +
            `请：1. 分析各方优缺点 2. 指出共识和分歧 ` +
            `3. 给出最终裁决 4. 说明理由`,
        },
      ],
    });
    return {
      opinions,
      arbitration:
        response.content[0].type === "text" ? response.content[0].text : "",
    };
  }

  async resolve(question: string) {
    const opinions = await this.collectOpinions(question);
    return this.arbitrate(question, opinions);
  }
}
```

### 综合共识框架：分级策略

将多种共识机制组合使用，从快到慢逐步升级。

```typescript
/** 综合共识框架：分级策略 */
class ConsensusFramework {
  private voting = new VotingConsensus(3);
  private arbiter = new ArbitrationConsensus();

  /** 分级共识策略 */
  async resolve(
    question: string
  ): Promise<{ method: string; answer: string; confidence: number }> {
    // Level 1: 多 Agent 投票（快速）
    const voteResult = await this.voting.reachConsensus(question);
    if (voteResult.consensus_ratio > 0.6) {
      return {
        method: "voting",
        answer: voteResult.consensus_answer,
        confidence: voteResult.avg_confidence,
      };
    }

    // Level 2: 仲裁（更可靠但更慢）
    const arbResult = await this.arbiter.resolve(question);
    return {
      method: "arbitration",
      answer: arbResult.arbitration,
      confidence: 0.7,
    };
  }
}
```

::: info 共识不等于正确
多数一致的答案不一定是正确答案，特别是当所有 Agent 共享相同的偏见时（比如 LLM 的常见错误认知）。共识机制提高了可靠性但不保证正确性。对于高风险场景，仍需引入外部验证。
:::

## 小结

- 消息传递适合直接的请求-响应交互，Blackboard 适合需要共享上下文的场景，事件驱动适合松耦合的异步协作
- 任务分解要考虑依赖关系，调度器确保子任务按正确顺序执行
- 投票是最简单的共识方法，仲裁适合复杂分歧，分级策略从快到慢逐步升级
- 实际系统中常混合使用多种通信模式和共识机制

## 练习

1. 用消息传递模式实现一个"接力翻译"：中文 -> 英文 -> 日文 -> 中文，观察信息损失。
2. 用 Blackboard 模式实现一个"协同写故事"系统：3 个 Agent 轮流在黑板上续写。
3. 实现加权投票：根据每个 Agent 的 confidence 加权，而非简单多数。

## 参考资源

- [LangGraph: Multi-Agent Communication](https://langchain-ai.github.io/langgraph/concepts/multi_agent/) -- LangGraph 多 Agent 通信文档
- [Self-Consistency Improves CoT (arXiv:2203.11171)](https://arxiv.org/abs/2203.11171) -- 自我一致性论文
- [Multi-Agent Debate (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) -- 多 Agent 辩论提升事实性
- [HuggingGPT (arXiv:2303.17580)](https://arxiv.org/abs/2303.17580) -- LLM 任务分解与调度
- [Blackboard Architecture (Wikipedia)](https://en.wikipedia.org/wiki/Blackboard_(design_pattern)) -- Blackboard 模式
- [Python asyncio Documentation](https://docs.python.org/3/library/asyncio.html) -- Python 异步编程文档
