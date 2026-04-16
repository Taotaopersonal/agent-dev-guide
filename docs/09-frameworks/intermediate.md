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

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Agent 的状态定义"""
    messages: Annotated[list, add_messages]  # 对话消息（自动追加）
    current_step: str                         # 当前步骤
    results: dict                             # 中间结果
```

**Node（节点）** -- 执行具体操作的函数，接收状态、返回更新：

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

def chatbot_node(state: AgentState) -> dict:
    """聊天节点：调用 LLM 生成回复"""
    response = model.invoke(state["messages"])
    return {"messages": [response]}
```

**Edge（边）** -- 定义节点之间的转换关系，可以是固定的或条件性的。

### 完整示例：ReAct Agent

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

### 条件路由

LangGraph 的核心能力 -- 根据状态动态选择下一个节点：

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

### 状态持久化

LangGraph 支持 Checkpointing，可以保存和恢复工作流状态：

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

### 实战：多步研究助手

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

::: tip LangChain Agent vs LangGraph
- 简单的"问题 -> 工具调用 -> 回答"：用 LangChain Agent
- 复杂的多步骤、有分支、需要循环的工作流：用 LangGraph
:::

## CrewAI 多角色协作

CrewAI 用直觉的隐喻来组织多 Agent 系统 -- **Agent**（团队成员）、**Task**（具体工作）、**Crew**（团队）、**Process**（工作流程）。

```bash
pip install crewai crewai-tools
```

### 定义 Agent

每个 Agent 有明确的角色定位，就像你在组建一个真实团队：

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

### 定义 Task

Task 定义具体的工作内容和期望输出：

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

### 组建 Crew 并运行

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

::: info Sequential vs Hierarchical
- **Sequential（顺序流程）**：任务按定义顺序执行，上一个输出传给下一个。简单明确。
- **Hierarchical（层级流程）**：有一个"管理者"Agent 动态分配任务。更灵活，适合复杂协作。

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

## Anthropic Agent SDK

Anthropic SDK 秉持**轻量和灵活**的设计哲学 -- 只封装真正重复的模式，让你保持对代码的完全控制。

```bash
pip install anthropic
```

### 构建可复用的 Agent 类

```python
import anthropic
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    input_schema: dict
    handler: Callable

@dataclass
class Agent:
    """轻量级 Agent"""
    name: str
    system_prompt: str
    model: str = "claude-sonnet-4-20250514"
    tools: list[Tool] = field(default_factory=list)
    max_steps: int = 10
    client: anthropic.Anthropic = field(default_factory=anthropic.Anthropic)

    def _get_tool_definitions(self) -> list[dict]:
        return [{"name": t.name, "description": t.description,
                 "input_schema": t.input_schema} for t in self.tools]

    def _execute_tool(self, name: str, input_data: dict) -> str:
        for tool in self.tools:
            if tool.name == name:
                return tool.handler(**input_data)
        return f"未知工具: {name}"

    def run(self, user_message: str) -> str:
        """运行 Agent 核心循环"""
        messages = [{"role": "user", "content": user_message}]
        tool_defs = self._get_tool_definitions()

        for _ in range(self.max_steps):
            kwargs = {"model": self.model, "max_tokens": 2048,
                      "system": self.system_prompt, "messages": messages}
            if tool_defs:
                kwargs["tools"] = tool_defs

            response = self.client.messages.create(**kwargs)

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return ""

            # 处理工具调用
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return "达到最大步骤数"
```

### Handoff 机制

Handoff 是 Anthropic SDK 中优雅的多 Agent 协作模式：当一个 Agent 遇到超出能力范围的问题时，将对话"交接"给更专业的 Agent。

```python
@dataclass
class HandoffTarget:
    agent: "Agent"
    description: str

@dataclass
class AgentWithHandoff(Agent):
    handoffs: list[HandoffTarget] = field(default_factory=list)

    def _get_tool_definitions(self) -> list[dict]:
        defs = super()._get_tool_definitions()
        for target in self.handoffs:
            defs.append({
                "name": f"handoff_to_{target.agent.name}",
                "description": f"将对话交接给 {target.agent.name}: {target.description}",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "交接原因"},
                        "context": {"type": "string", "description": "需要传递的上下文"}
                    },
                    "required": ["reason"]
                }
            })
        return defs

    def _execute_tool(self, name: str, input_data: dict) -> str:
        if name.startswith("handoff_to_"):
            target_name = name.replace("handoff_to_", "")
            for target in self.handoffs:
                if target.agent.name == target_name:
                    print(f"\n[Handoff] {self.name} -> {target_name}")
                    return target.agent.run(input_data.get("context", ""))
        return super()._execute_tool(name, input_data)

# 构建 Handoff 系统
tech_agent = Agent(name="tech_support", system_prompt="你是技术支持专家。")
billing_agent = Agent(name="billing_support", system_prompt="你是账单专家。")

receptionist = AgentWithHandoff(
    name="receptionist",
    system_prompt="你是客服前台。技术问题交给 tech_support，账单问题交给 billing_support。",
    handoffs=[
        HandoffTarget(tech_agent, "处理技术问题、故障排查"),
        HandoffTarget(billing_agent, "处理账单、付款、退款问题"),
    ],
)

answer = receptionist.run("我的订单付款失败了，怎么办？")
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
