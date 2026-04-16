# Agent 框架进阶：LangGraph、CrewAI 与 Anthropic SDK

::: tip 学习目标
- 掌握 LangGraph 的状态图模型，能够构建带条件路由和持久化的复杂工作流
- 理解 CrewAI 的 Agent-Task-Crew 抽象，能够搭建多角色协作系统
- 了解 Anthropic Agent SDK 的极简设计理念和 Handoff 机制

**学完你能做到：** 用 LangGraph 实现一个多步研究助手工作流，用 CrewAI 搭建一个内容创作团队，用 Anthropic SDK 构建一个带工具的 Agent。
:::

## LangGraph 工作流编排

LangChain 的 Chain 是线性管道，无法表达循环、分支、并行等复杂流程。LangGraph 用**状态图（StateGraph）**来定义工作流，让你精确控制每一步的执行逻辑。

### 核心概念

**State（状态）** -- 在图中所有节点间传递和共享的数据结构：

> 以下是 **Python 框架代码**，展示 LangGraph 的 TypedDict 状态定义。

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Agent 的状态定义"""
    messages: Annotated[list, add_messages]  # 对话消息（自动追加）
    current_step: str                         # 当前步骤
    results: dict                             # 中间结果
```

**等价的 TypeScript 表达** -- 用 TypeScript 接口表达同样的状态结构：

```typescript
// LangGraph 的 TypedDict 状态 → TypeScript interface
// Annotated[list, add_messages] 的含义：该字段的更新策略是"追加"而非"覆盖"
interface AgentState {
  messages: Message[]; // 对话消息（更新时自动追加到数组末尾）
  currentStep: string; // 当前步骤
  results: Record<string, unknown>; // 中间结果
}

// LangGraph 的 add_messages 是一个 reducer 函数
// 等价概念：状态更新策略
type StateReducer<T> = (current: T, update: T) => T;
const appendMessages: StateReducer<Message[]> = (current, update) => [
  ...current,
  ...update,
];
```

**Node（节点）** -- 执行具体操作的函数，接收状态、返回更新：

> 以下是 **Python 框架代码**，展示 LangGraph 的节点函数签名。

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

def chatbot_node(state: AgentState) -> dict:
    """聊天节点：调用 LLM 生成回复"""
    response = model.invoke(state["messages"])
    return {"messages": [response]}
```

**等价的 TypeScript 表达** -- 节点就是"接收状态、返回状态更新"的纯函数：

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// LangGraph 节点的本质：(state) => Partial<State>
// 接收完整状态，返回需要更新的字段
async function chatbotNode(
  state: AgentState
): Promise<Partial<AgentState>> {
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    messages: state.messages,
  });
  const text =
    response.content.find((b) => b.type === "text")?.type === "text"
      ? (response.content.find((b) => b.type === "text") as { text: string }).text
      : "";
  return { messages: [{ role: "assistant", content: text }] };
}
```

**Edge（边）** -- 定义节点之间的转换关系，可以是固定的或条件性的。

### 完整示例：ReAct Agent

> 以下是 **Python 框架代码**，展示 LangGraph 的完整 StateGraph 构建流程。

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

# 1. 定义状态
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. 定义工具
@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"搜索结果: 关于 '{query}' 的信息..."

@tool
def calculator(expression: str) -> str:
    """数学计算"""
    return str(eval(expression))

tools = [search, calculator]

# 3. 定义节点
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def agent_node(state: State) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: State) -> dict:
    last_msg = state["messages"][-1]
    tool_map = {t.name: t for t in tools}
    results = []
    for tc in last_msg.tool_calls:
        tool_fn = tool_map[tc["name"]]
        result = tool_fn.invoke(tc["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return {"messages": results}

# 4. 定义条件路由
def should_continue(state: State) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"

# 5. 构建图
graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
graph.add_edge("tools", "agent")  # 工具执行完回到 agent

# 6. 编译运行
app = graph.compile()
result = app.invoke({
    "messages": [HumanMessage(content="计算 (15 + 27) * 3 等于多少")]
})

for msg in result["messages"]:
    print(f"[{msg.type}] {str(msg.content)[:200]}")
```

**等价的 TypeScript 实现** -- 用类型系统 + 纯函数表达同样的状态图概念：

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// ===== 这是概念代码，展示 LangGraph StateGraph 的设计思路 =====

// 1. 定义状态（等价于 TypedDict）
interface State {
  messages: Anthropic.MessageParam[];
}

// 2. 定义工具
const tools: Anthropic.Tool[] = [
  {
    name: "search",
    description: "搜索信息",
    input_schema: {
      type: "object" as const,
      properties: { query: { type: "string" } },
      required: ["query"],
    },
  },
  {
    name: "calculator",
    description: "数学计算",
    input_schema: {
      type: "object" as const,
      properties: { expression: { type: "string" } },
      required: ["expression"],
    },
  },
];

function executeTool(name: string, args: Record<string, string>): string {
  if (name === "search") return `搜索结果: 关于 '${args.query}' 的信息...`;
  if (name === "calculator") return String(eval(args.expression));
  return `未知工具: ${name}`;
}

// 3. 定义节点（纯函数，接收状态返回更新）
type NodeFn = (state: State) => Promise<Partial<State>>;

const agentNode: NodeFn = async (state) => {
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 1024,
    tools,
    messages: state.messages,
  });
  return {
    messages: [
      ...state.messages,
      { role: "assistant" as const, content: response.content },
    ],
  };
};

const toolNode: NodeFn = async (state) => {
  const lastMsg = state.messages[state.messages.length - 1];
  if (lastMsg.role !== "assistant" || !Array.isArray(lastMsg.content)) {
    return state;
  }
  const toolResults: Anthropic.ToolResultBlockParam[] = [];
  for (const block of lastMsg.content) {
    if (typeof block === "object" && "type" in block && block.type === "tool_use") {
      const result = executeTool(block.name, block.input as Record<string, string>);
      toolResults.push({
        type: "tool_result",
        tool_use_id: block.id,
        content: result,
      });
    }
  }
  return {
    messages: [...state.messages, { role: "user" as const, content: toolResults }],
  };
};

// 4. 条件路由（纯函数，返回下一个节点名称）
function shouldContinue(state: State): "tools" | "end" {
  const lastMsg = state.messages[state.messages.length - 1];
  if (lastMsg.role === "assistant" && Array.isArray(lastMsg.content)) {
    const hasToolUse = lastMsg.content.some(
      (b) => typeof b === "object" && "type" in b && b.type === "tool_use"
    );
    if (hasToolUse) return "tools";
  }
  return "end";
}

// 5. 图执行引擎（等价于 StateGraph.compile()）
async function runGraph(initialState: State): Promise<State> {
  let state = initialState;

  for (let i = 0; i < 10; i++) {
    // agent 节点
    state = { ...state, ...(await agentNode(state)) };

    // 条件路由
    const next = shouldContinue(state);
    if (next === "end") break;

    // tools 节点 -> 回到 agent
    state = { ...state, ...(await toolNode(state)) };
  }
  return state;
}

// 6. 运行
const result = await runGraph({
  messages: [{ role: "user", content: "计算 (15 + 27) * 3 等于多少" }],
});
console.log(result.messages[result.messages.length - 1]);
```

### 条件路由

LangGraph 的核心能力 -- 根据状态动态选择下一个节点：

> 以下是 **Python 框架代码**，展示 LangGraph 条件路由的声明方式。

```python
def route_by_intent(state: State) -> str:
    """根据用户意图路由到不同处理节点"""
    last_msg = state["messages"][-1].content.lower()
    if "代码" in last_msg or "编程" in last_msg:
        return "code_agent"
    elif "数据" in last_msg or "分析" in last_msg:
        return "data_agent"
    else:
        return "general_agent"

graph.add_conditional_edges(
    "classifier",
    route_by_intent,
    {"code_agent": "code_agent", "data_agent": "data_agent", "general_agent": "general_agent"}
)
```

**等价的 TypeScript 表达** -- 条件路由的本质就是一个返回节点名称的纯函数：

```typescript
// 条件路由函数：根据状态决定下一步走向哪个节点
type RouteFn = (state: State) => string;

const routeByIntent: RouteFn = (state) => {
  const lastMsg = state.messages[state.messages.length - 1];
  const content =
    typeof lastMsg.content === "string" ? lastMsg.content.toLowerCase() : "";
  if (content.includes("代码") || content.includes("编程"))
    return "code_agent";
  if (content.includes("数据") || content.includes("分析"))
    return "data_agent";
  return "general_agent";
};

// 在图执行引擎中使用条件路由
const nodeMap: Record<string, NodeFn> = {
  code_agent: codeAgentNode,
  data_agent: dataAgentNode,
  general_agent: generalAgentNode,
};

const nextNodeName = routeByIntent(state);
const nextNode = nodeMap[nextNodeName];
state = { ...state, ...(await nextNode(state)) };
```

### 状态持久化

LangGraph 支持 Checkpointing，可以保存和恢复工作流状态：

> 以下是 **Python 框架代码**，展示 LangGraph 的 MemorySaver 状态持久化。

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# 第一次对话
config = {"configurable": {"thread_id": "user_123"}}
result1 = app.invoke(
    {"messages": [HumanMessage(content="我叫小明")]},
    config=config,
)

# 第二次对话（同一线程，保持上下文）
result2 = app.invoke(
    {"messages": [HumanMessage(content="我叫什么名字？")]},
    config=config,
)
# Agent 能记住"小明"
```

**等价的 TypeScript 概念** -- Checkpointing 的本质是按会话 ID 存取状态快照：

```typescript
// LangGraph MemorySaver 的等价概念：一个按 threadId 索引的状态存储
class MemorySaver<S> {
  private store = new Map<string, S>();

  save(threadId: string, state: S): void {
    this.store.set(threadId, structuredClone(state));
  }

  load(threadId: string): S | undefined {
    const saved = this.store.get(threadId);
    return saved ? structuredClone(saved) : undefined;
  }
}

// 使用：在图执行前加载状态，执行后保存状态
const memory = new MemorySaver<State>();

async function invokeWithMemory(
  threadId: string,
  newMessage: string
): Promise<State> {
  // 从上次保存的状态恢复（如果存在）
  const savedState = memory.load(threadId);
  const state: State = savedState
    ? {
        messages: [
          ...savedState.messages,
          { role: "user" as const, content: newMessage },
        ],
      }
    : { messages: [{ role: "user" as const, content: newMessage }] };

  const result = await runGraph(state);
  memory.save(threadId, result); // 保存状态快照
  return result;
}

// 第一次对话
await invokeWithMemory("user_123", "我叫小明");
// 第二次对话（同一线程，保持上下文）
await invokeWithMemory("user_123", "我叫什么名字？");
// Agent 能记住"小明"
```

### 实战：多步研究助手

> 以下是 **Python 框架代码**，展示 LangGraph 的多步工作流完整实现。

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    plan: list[str]
    current_step: int
    search_results: list[str]
    report: str

model = ChatOpenAI(model="gpt-4o-mini")

def planner(state: ResearchState) -> dict:
    """制定研究计划"""
    response = model.invoke([
        {"role": "system", "content": "你是研究规划师。为问题制定 3-5 步搜索计划。返回步骤列表。"},
        {"role": "user", "content": state["question"]}
    ])
    steps = [s.strip("0123456789. -") for s in response.content.strip().split("\n") if s.strip()]
    return {"plan": steps[:5], "current_step": 0, "search_results": []}

def researcher(state: ResearchState) -> dict:
    """执行搜索步骤"""
    step = state["plan"][state["current_step"]]
    response = model.invoke([
        {"role": "system", "content": "你是研究员。根据搜索主题提供详细信息。"},
        {"role": "user", "content": f"搜索主题: {step}"}
    ])
    results = state["search_results"] + [f"[{step}]: {response.content}"]
    return {"search_results": results, "current_step": state["current_step"] + 1}

def reporter(state: ResearchState) -> dict:
    """撰写研究报告"""
    all_results = "\n\n".join(state["search_results"])
    response = model.invoke([
        {"role": "system", "content": "你是报告撰写者。基于研究结果撰写简洁的报告。"},
        {"role": "user", "content": f"研究问题: {state['question']}\n\n研究结果:\n{all_results}"}
    ])
    return {"report": response.content}

def should_continue_research(state: ResearchState) -> str:
    if state["current_step"] < len(state["plan"]):
        return "research"
    return "report"

# 构建工作流
workflow = StateGraph(ResearchState)
workflow.add_node("planner", planner)
workflow.add_node("researcher", researcher)
workflow.add_node("reporter", reporter)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "researcher")
workflow.add_conditional_edges("researcher", should_continue_research, {
    "research": "researcher",
    "report": "reporter",
})
workflow.add_edge("reporter", END)

app = workflow.compile()

result = app.invoke({
    "messages": [], "question": "大语言模型的主要技术突破有哪些？",
    "plan": [], "current_step": 0, "search_results": [], "report": "",
})
print(result["report"])
```

**等价的 TypeScript 实现** -- 用 Anthropic SDK + 状态机思路实现同样的多步研究工作流：

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// 1. 定义状态（等价于 ResearchState TypedDict）
interface ResearchState {
  question: string;
  plan: string[];
  currentStep: number;
  searchResults: string[];
  report: string;
}

// 辅助函数：调用 LLM
async function callLLM(system: string, userMsg: string): Promise<string> {
  const response = await client.messages.create({
    model: "claude-sonnet-4-20250514",
    max_tokens: 2048,
    system,
    messages: [{ role: "user", content: userMsg }],
  });
  const textBlock = response.content.find((b) => b.type === "text");
  return textBlock?.type === "text" ? textBlock.text : "";
}

// 2. 定义节点函数
async function planner(
  state: ResearchState
): Promise<Partial<ResearchState>> {
  const content = await callLLM(
    "你是研究规划师。为问题制定 3-5 步搜索计划。返回步骤列表。",
    state.question
  );
  const steps = content
    .split("\n")
    .map((s) => s.replace(/^[\d.\-\s]+/, "").trim())
    .filter(Boolean)
    .slice(0, 5);
  return { plan: steps, currentStep: 0, searchResults: [] };
}

async function researcher(
  state: ResearchState
): Promise<Partial<ResearchState>> {
  const step = state.plan[state.currentStep];
  const content = await callLLM(
    "你是研究员。根据搜索主题提供详细信息。",
    `搜索主题: ${step}`
  );
  return {
    searchResults: [...state.searchResults, `[${step}]: ${content}`],
    currentStep: state.currentStep + 1,
  };
}

async function reporter(
  state: ResearchState
): Promise<Partial<ResearchState>> {
  const allResults = state.searchResults.join("\n\n");
  const content = await callLLM(
    "你是报告撰写者。基于研究结果撰写简洁的报告。",
    `研究问题: ${state.question}\n\n研究结果:\n${allResults}`
  );
  return { report: content };
}

// 3. 条件路由
function shouldContinueResearch(
  state: ResearchState
): "research" | "report" {
  return state.currentStep < state.plan.length ? "research" : "report";
}

// 4. 图执行引擎（等价于 StateGraph.compile() + invoke()）
async function runResearchWorkflow(question: string): Promise<string> {
  let state: ResearchState = {
    question,
    plan: [],
    currentStep: 0,
    searchResults: [],
    report: "",
  };

  // planner -> researcher (循环) -> reporter
  state = { ...state, ...(await planner(state)) };

  while (shouldContinueResearch(state) === "research") {
    state = { ...state, ...(await researcher(state)) };
  }

  state = { ...state, ...(await reporter(state)) };
  return state.report;
}

const report = await runResearchWorkflow(
  "大语言模型的主要技术突破有哪些？"
);
console.log(report);
```

::: tip LangChain Agent vs LangGraph
- 简单的"问题 -> 工具调用 -> 回答"：用 LangChain Agent
- 复杂的多步骤、有分支、需要循环的工作流：用 LangGraph
:::

## CrewAI 多角色协作

CrewAI 用直觉的隐喻来组织多 Agent 系统 -- **Agent**（团队成员）、**Task**（具体工作）、**Crew**（团队）、**Process**（工作流程）。

```bash
npm install @anthropic-ai/sdk  # CrewAI 是 Python 独占框架，以下用 TS 概念代码展示其设计模式
```

### 定义 Agent

每个 Agent 有明确的角色定位，就像你在组建一个真实团队：

> 以下是 **Python 框架代码**，展示 CrewAI 的 Agent 角色定义方式。

```python
from crewai import Agent

researcher = Agent(
    role="资深技术研究员",
    goal="深入调研指定技术主题，收集全面准确的信息",
    backstory="""你是一位有 10 年经验的技术研究员，
    擅长快速理解新技术并提取关键信息。""",
    verbose=True,
    allow_delegation=False,
)

writer = Agent(
    role="技术内容写作专家",
    goal="将技术研究转化为易懂、吸引人的文章",
    backstory="""你是一位技术博客作家，擅长将复杂的技术概念
    用通俗易懂的语言解释。""",
    verbose=True,
    allow_delegation=False,
)

editor = Agent(
    role="资深内容编辑",
    goal="确保文章质量：准确性、可读性、结构完整性",
    backstory="""你是一位严格的编辑，有丰富的技术出版经验。""",
    verbose=True,
    allow_delegation=False,
)
```

**等价的 TypeScript 概念代码** -- 展示 CrewAI Agent 的设计思路：

```typescript
// ===== 这是概念代码，展示 CrewAI Agent 的设计思路 =====
// CrewAI 的核心设计：每个 Agent = 角色(role) + 目标(goal) + 背景(backstory)
// 这三个字段会被组合成 system prompt，驱动 LLM 扮演特定角色

interface AgentConfig {
  role: string; // 角色定位
  goal: string; // 工作目标
  backstory: string; // 背景故事（影响行为风格）
  allowDelegation: boolean; // 是否允许委派任务给其他 Agent
}

const researcher: AgentConfig = {
  role: "资深技术研究员",
  goal: "深入调研指定技术主题，收集全面准确的信息",
  backstory:
    "你是一位有 10 年经验的技术研究员，擅长快速理解新技术并提取关键信息。",
  allowDelegation: false,
};

const writer: AgentConfig = {
  role: "技术内容写作专家",
  goal: "将技术研究转化为易懂、吸引人的文章",
  backstory: "你是一位技术博客作家，擅长将复杂的技术概念用通俗易懂的语言解释。",
  allowDelegation: false,
};

const editor: AgentConfig = {
  role: "资深内容编辑",
  goal: "确保文章质量：准确性、可读性、结构完整性",
  backstory: "你是一位严格的编辑，有丰富的技术出版经验。",
  allowDelegation: false,
};
```

### 定义 Task

Task 定义具体的工作内容和期望输出：

> 以下是 **Python 框架代码**，展示 CrewAI 的 Task 定义和依赖声明。

```python
from crewai import Task

research_task = Task(
    description="""深入调研以下技术主题：{topic}
    你需要：
    1. 梳理该技术的核心概念和原理
    2. 总结当前发展状况和最新进展
    3. 分析优缺点和适用场景
    4. 收集实际应用案例""",
    expected_output="一份结构化的技术研究报告",
    agent=researcher,
)

writing_task = Task(
    description="""基于研究报告撰写一篇技术博客。
    要求：标题吸引人、内容深入浅出、包含代码示例、1500字左右。""",
    expected_output="完整的技术博客文章",
    agent=writer,
    context=[research_task],  # 显式声明依赖
)

review_task = Task(
    description="""审核并改进技术文章。检查：技术准确性、语言流畅度、结构完整性。
    如果质量合格，输出 "APPROVED" + 最终稿。""",
    expected_output="审核通过的最终稿或修改建议",
    agent=editor,
    context=[writing_task],
)
```

**等价的 TypeScript 概念代码** -- 展示 Task 的设计思路：

```typescript
// ===== 这是概念代码，展示 CrewAI Task 的设计思路 =====
// Task = 任务描述 + 期望输出 + 执行者 + 依赖任务

interface TaskConfig {
  description: string; // 任务描述（支持模板变量）
  expectedOutput: string; // 期望输出格式
  agent: AgentConfig; // 负责执行的 Agent
  context?: TaskConfig[]; // 依赖的前置任务（其输出会作为本任务的上下文）
}

const researchTask: TaskConfig = {
  description: `深入调研以下技术主题：{topic}
    你需要：
    1. 梳理该技术的核心概念和原理
    2. 总结当前发展状况和最新进展
    3. 分析优缺点和适用场景
    4. 收集实际应用案例`,
  expectedOutput: "一份结构化的技术研究报告",
  agent: researcher,
};

const writingTask: TaskConfig = {
  description:
    "基于研究报告撰写一篇技术博客。要求：标题吸引人、内容深入浅出、包含代码示例、1500字左右。",
  expectedOutput: "完整的技术博客文章",
  agent: writer,
  context: [researchTask], // 显式声明依赖
};

const reviewTask: TaskConfig = {
  description: `审核并改进技术文章。检查：技术准确性、语言流畅度、结构完整性。
    如果质量合格，输出 "APPROVED" + 最终稿。`,
  expectedOutput: "审核通过的最终稿或修改建议",
  agent: editor,
  context: [writingTask],
};
```

### 组建 Crew 并运行

> 以下是 **Python 框架代码**，展示 CrewAI 的 Crew 组装和执行方式。

```python
from crewai import Crew, Process

# Sequential 流程：研究员 -> 写作者 -> 编辑
content_crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,
    verbose=True,
)

result = content_crew.kickoff(inputs={"topic": "RAG（检索增强生成）技术"})
print(result)
```

**等价的 TypeScript 实现** -- 用 Anthropic SDK 实现 Sequential Crew 的核心逻辑：

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

// Crew 的 Sequential 流程本质：按顺序执行任务，上一个输出传给下一个
async function runSequentialCrew(
  tasks: TaskConfig[],
  inputs: Record<string, string>
): Promise<string> {
  let previousOutput = "";

  for (const task of tasks) {
    // 将 Agent 的 role + goal + backstory 组合成 system prompt
    const systemPrompt = `你的角色: ${task.agent.role}
你的目标: ${task.agent.goal}
你的背景: ${task.agent.backstory}

期望输出格式: ${task.expectedOutput}`;

    // 将描述中的模板变量替换为实际值
    let description = task.description;
    for (const [key, value] of Object.entries(inputs)) {
      description = description.replace(`{${key}}`, value);
    }

    // 如果有前置任务的输出，作为上下文传入
    const userContent = previousOutput
      ? `前置任务的输出:\n${previousOutput}\n\n当前任务:\n${description}`
      : description;

    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 4096,
      system: systemPrompt,
      messages: [{ role: "user", content: userContent }],
    });

    const textBlock = response.content.find((b) => b.type === "text");
    previousOutput =
      textBlock?.type === "text" ? textBlock.text : "";
    console.log(`[${task.agent.role}] 完成`);
  }

  return previousOutput;
}

const result = await runSequentialCrew(
  [researchTask, writingTask, reviewTask],
  { topic: "RAG（检索增强生成）技术" }
);
console.log(result);
```

::: info Sequential vs Hierarchical
- **Sequential（顺序流程）**：任务按定义顺序执行，上一个输出传给下一个。简单明确。
- **Hierarchical（层级流程）**：有一个"管理者"Agent 动态分配任务。更灵活，适合复杂协作。

> 以下是 **Python 框架代码**，展示 CrewAI 的 Hierarchical 流程配置。

```python
# Hierarchical 用法
from langchain_openai import ChatOpenAI
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, review_task],
    process=Process.hierarchical,
    manager_llm=ChatOpenAI(model="gpt-4o"),
    verbose=True,
)
```
:::

### 自定义工具

> 以下是 **Python 框架代码**，展示 CrewAI 的 BaseTool 自定义工具模式。

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="搜索查询")

class CustomSearchTool(BaseTool):
    name: str = "knowledge_base_search"
    description: str = "在内部知识库中搜索技术文档"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        return f"知识库搜索结果: {query}"

# 给 Agent 配备工具
researcher_with_tools = Agent(
    role="研究员",
    goal="深入调研技术主题",
    backstory="你是经验丰富的技术研究员。",
    tools=[CustomSearchTool()],
    verbose=True,
)
```

**等价的 TypeScript 概念代码** -- 展示工具定义的设计模式：

```typescript
// ===== 这是概念代码，展示 CrewAI 工具定义的设计思路 =====
// CrewAI 的 BaseTool 本质：name + description + input schema + 执行函数

interface ToolConfig {
  name: string;
  description: string;
  inputSchema: {
    type: "object";
    properties: Record<string, { type: string; description: string }>;
    required: string[];
  };
  run: (args: Record<string, string>) => string | Promise<string>;
}

const customSearchTool: ToolConfig = {
  name: "knowledge_base_search",
  description: "在内部知识库中搜索技术文档",
  inputSchema: {
    type: "object",
    properties: {
      query: { type: "string", description: "搜索查询" },
    },
    required: ["query"],
  },
  run: (args) => `知识库搜索结果: ${args.query}`,
};

// 给 Agent 配备工具
const researcherWithTools: AgentConfig & { tools: ToolConfig[] } = {
  role: "研究员",
  goal: "深入调研技术主题",
  backstory: "你是经验丰富的技术研究员。",
  allowDelegation: false,
  tools: [customSearchTool],
};
```

## Anthropic Agent SDK

Anthropic SDK 秉持**轻量和灵活**的设计哲学 -- 只封装真正重复的模式，让你保持对代码的完全控制。

```bash
npm install @anthropic-ai/sdk
```

### 构建可复用的 Agent 类

```typescript
import Anthropic from "@anthropic-ai/sdk";

// 工具定义
interface Tool {
  name: string;
  description: string;
  inputSchema: Anthropic.Tool["input_schema"];
  handler: (args: Record<string, unknown>) => string | Promise<string>;
}

// 轻量级 Agent
class Agent {
  name: string;
  systemPrompt: string;
  model: string;
  tools: Tool[];
  maxSteps: number;
  client: Anthropic;

  constructor(config: {
    name: string;
    systemPrompt: string;
    model?: string;
    tools?: Tool[];
    maxSteps?: number;
  }) {
    this.name = config.name;
    this.systemPrompt = config.systemPrompt;
    this.model = config.model ?? "claude-sonnet-4-20250514";
    this.tools = config.tools ?? [];
    this.maxSteps = config.maxSteps ?? 10;
    this.client = new Anthropic();
  }

  private getToolDefinitions(): Anthropic.Tool[] {
    return this.tools.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.inputSchema,
    }));
  }

  private async executeTool(
    name: string,
    inputData: Record<string, unknown>
  ): Promise<string> {
    const tool = this.tools.find((t) => t.name === name);
    if (!tool) return `未知工具: ${name}`;
    return tool.handler(inputData);
  }

  async run(userMessage: string): Promise<string> {
    // 运行 Agent 核心循环
    const messages: Anthropic.MessageParam[] = [
      { role: "user", content: userMessage },
    ];
    const toolDefs = this.getToolDefinitions();

    for (let i = 0; i < this.maxSteps; i++) {
      const createParams: Anthropic.MessageCreateParams = {
        model: this.model,
        max_tokens: 2048,
        system: this.systemPrompt,
        messages,
      };
      if (toolDefs.length > 0) {
        createParams.tools = toolDefs;
      }

      const response = await this.client.messages.create(createParams);

      if (response.stop_reason === "end_turn") {
        const textBlock = response.content.find((b) => b.type === "text");
        return textBlock?.type === "text" ? textBlock.text : "";
      }

      // 处理工具调用
      const toolResults: Anthropic.ToolResultBlockParam[] = [];
      for (const block of response.content) {
        if (block.type === "tool_use") {
          const result = await this.executeTool(
            block.name,
            block.input as Record<string, unknown>
          );
          toolResults.push({
            type: "tool_result",
            tool_use_id: block.id,
            content: String(result),
          });
        }
      }

      messages.push({ role: "assistant", content: response.content });
      messages.push({ role: "user", content: toolResults });
    }

    return "达到最大步骤数";
  }
}
```

### Handoff 机制

Handoff 是 Anthropic SDK 中优雅的多 Agent 协作模式：当一个 Agent 遇到超出能力范围的问题时，将对话"交接"给更专业的 Agent。

```typescript
import Anthropic from "@anthropic-ai/sdk";

// Handoff 目标定义
interface HandoffTarget {
  agent: Agent;
  description: string;
}

// 带 Handoff 能力的 Agent
class AgentWithHandoff extends Agent {
  handoffs: HandoffTarget[];

  constructor(
    config: ConstructorParameters<typeof Agent>[0] & {
      handoffs?: HandoffTarget[];
    }
  ) {
    super(config);
    this.handoffs = config.handoffs ?? [];
  }

  private getToolDefinitions(): Anthropic.Tool[] {
    const baseDefs = this.tools.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.inputSchema,
    }));

    // 为每个 Handoff 目标创建一个虚拟工具
    for (const target of this.handoffs) {
      baseDefs.push({
        name: `handoff_to_${target.agent.name}`,
        description: `将对话交接给 ${target.agent.name}: ${target.description}`,
        input_schema: {
          type: "object" as const,
          properties: {
            reason: { type: "string", description: "交接原因" },
            context: { type: "string", description: "需要传递的上下文" },
          },
          required: ["reason"],
        },
      });
    }

    return baseDefs;
  }

  protected async executeTool(
    name: string,
    inputData: Record<string, unknown>
  ): Promise<string> {
    // 检查是否是 Handoff 调用
    if (name.startsWith("handoff_to_")) {
      const targetName = name.replace("handoff_to_", "");
      const target = this.handoffs.find(
        (h) => h.agent.name === targetName
      );
      if (target) {
        console.log(`\n[Handoff] ${this.name} -> ${targetName}`);
        return target.agent.run(
          (inputData.context as string) ?? ""
        );
      }
    }
    // 退回到普通工具执行
    const tool = this.tools.find((t) => t.name === name);
    if (!tool) return `未知工具: ${name}`;
    return tool.handler(inputData);
  }
}

// 构建 Handoff 系统
const techAgent = new Agent({
  name: "tech_support",
  systemPrompt: "你是技术支持专家。",
});

const billingAgent = new Agent({
  name: "billing_support",
  systemPrompt: "你是账单专家。",
});

const receptionist = new AgentWithHandoff({
  name: "receptionist",
  systemPrompt:
    "你是客服前台。技术问题交给 tech_support，账单问题交给 billing_support。",
  handoffs: [
    { agent: techAgent, description: "处理技术问题、故障排查" },
    { agent: billingAgent, description: "处理账单、付款、退款问题" },
  ],
});

const answer = await receptionist.run("我的订单付款失败了，怎么办？");
```

## 三大框架对比

| 维度 | LangGraph | CrewAI | Anthropic SDK |
|------|-----------|--------|--------------|
| 抽象层次 | 中 | 中 | 最低 |
| 学习曲线 | 中等 | 平缓 | 平缓 |
| 灵活性 | 高（显式状态图） | 中（固定 Agent-Task 模式） | 最高（完全手控） |
| 多 Agent | 通过图组合 | 原生支持（Crew） | 通过 Handoff |
| 状态管理 | Checkpointing | 框架内置 | 需自行实现 |
| 生态 | LangChain 生态 | 独立生态 | Claude 专属 |
| 适合场景 | 复杂工作流 | 多角色协作 | 极简深度集成 |

## 小结

- **LangGraph** 用状态图定义工作流：State 共享数据、Node 执行逻辑、条件 Edge 动态分支，适合多步骤复杂流程
- **CrewAI** 用 Agent-Task-Crew 三层抽象：Agent 定角色、Task 定任务、Crew 定流程，是最简单的多 Agent 协作框架
- **Anthropic SDK** 走极简路线：自己写 Agent 循环，Handoff 实现优雅的 Agent 间交接
- 三者不是互斥的 -- 你可以在 LangGraph 里嵌入 CrewAI 子系统，或用 Anthropic SDK 作为底层 Model Provider

## 练习题

1. 用 LangGraph 实现一个 Plan-and-Execute Agent：先规划步骤，再逐步执行，在关键步骤暂停等待人类确认。
2. 用 CrewAI 构建一个"产品分析团队"：市场调研员 + 竞品分析师 + 报告撰写者，对比 Sequential 和 Hierarchical 的输出差异。
3. 用 Anthropic SDK 构建一个 3 层 Handoff 系统：前台 -> 部门 -> 专家，测试问题能否正确路由。

## 参考资源

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) -- LangGraph 官方文档
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/) -- 官方教程合集
- [CrewAI Documentation](https://docs.crewai.com/) -- CrewAI 官方文档
- [CrewAI GitHub](https://github.com/crewAIInc/crewAI) -- CrewAI 源码
- [Anthropic: Building effective agents](https://docs.anthropic.com/en/docs/build-with-claude/agentic) -- Agent 构建指南
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) -- Anthropic 官方示例集
- [Building Agents with LangGraph (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph/) -- LangGraph 课程
- [Multi AI Agent Systems with crewAI (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) -- CrewAI 课程
