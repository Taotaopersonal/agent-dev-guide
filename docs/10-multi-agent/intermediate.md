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

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable
import anthropic

client = anthropic.Anthropic()

@dataclass
class Message:
    """Agent 间通信的消息"""
    sender: str
    receiver: str
    content: str
    msg_type: str = "text"  # text, request, response, broadcast
    timestamp: datetime = field(default_factory=datetime.now)

class MessageBus:
    """消息总线：Agent 间通信的中介

    所有 Agent 都注册到消息总线上，
    通过总线发送消息给指定 Agent。
    """

    def __init__(self):
        self.agents: dict[str, Callable] = {}
        self.message_log: list[Message] = []

    def register(self, name: str, handler: Callable):
        """注册一个 Agent 的消息处理函数"""
        self.agents[name] = handler

    def send(self, msg: Message) -> str:
        """发送消息并获取回复"""
        self.message_log.append(msg)
        if msg.receiver in self.agents:
            response = self.agents[msg.receiver](msg)
            return response
        return f"Agent '{msg.receiver}' 未找到"

    def broadcast(self, sender: str, content: str) -> dict[str, str]:
        """广播消息给所有 Agent"""
        responses = {}
        for name, handler in self.agents.items():
            if name != sender:
                msg = Message(sender=sender, receiver=name,
                            content=content, msg_type="broadcast")
                responses[name] = handler(msg)
        return responses


# 创建 Agent 处理函数
def create_agent_handler(name: str, role: str):
    def handler(msg: Message) -> str:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=f"你是{role}。",
            messages=[{"role": "user",
                       "content": f"[来自{msg.sender}]: {msg.content}"}]
        )
        return response.content[0].text
    return handler


# 使用示例
bus = MessageBus()
bus.register("researcher", create_agent_handler("researcher", "技术研究员"))
bus.register("reviewer", create_agent_handler("reviewer", "审核专家"))

# Agent 之间通过总线通信
research_result = bus.send(Message(
    sender="manager",
    receiver="researcher",
    content="调研 RAG 技术的最新进展",
))

review_result = bus.send(Message(
    sender="manager",
    receiver="reviewer",
    content=f"请审核以下研究结果：\n{research_result}",
))
```

### 共享状态（Blackboard）模式

所有 Agent 通过一个共享的"黑板"交换信息。任何 Agent 都可以读写黑板，其他 Agent 可以看到更新。

这种模式特别适合多个 Agent 需要基于相同上下文协作的场景——比如协同写文档、多步骤分析。

```python
from threading import Lock
from typing import Any

class Blackboard:
    """共享黑板：多 Agent 的公共数据空间

    类似团队的共享白板：每个人都可以在上面写东西，
    其他人随时能看到最新内容。
    """

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._history: list[dict] = []
        self._lock = Lock()

    def write(self, key: str, value: Any, author: str):
        """写入数据"""
        with self._lock:
            self._data[key] = value
            self._history.append({
                "action": "write",
                "key": key,
                "author": author,
                "timestamp": datetime.now().isoformat(),
            })

    def read(self, key: str, default: Any = None) -> Any:
        """读取数据"""
        return self._data.get(key, default)

    def read_all(self) -> dict:
        """读取所有数据"""
        return self._data.copy()


class BlackboardAgent:
    """基于黑板的 Agent"""

    def __init__(self, name: str, role: str, blackboard: Blackboard):
        self.name = name
        self.role = role
        self.board = blackboard

    def think_and_act(self, task: str) -> str:
        """读取黑板信息，执行任务，将结果写回黑板"""
        # 读取黑板上的所有信息作为上下文
        context = self.board.read_all()
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=f"你是{self.role}。参考黑板上的已有信息完成任务。",
            messages=[{
                "role": "user",
                "content": f"黑板信息：\n{context_str}\n\n任务：{task}"
            }]
        )
        result = response.content[0].text

        # 写回黑板，其他 Agent 就能看到了
        self.board.write(f"{self.name}_output", result, self.name)
        return result


# 使用示例
board = Blackboard()

researcher = BlackboardAgent("researcher", "研究员", board)
analyst = BlackboardAgent("analyst", "分析师", board)
writer = BlackboardAgent("writer", "写手", board)

# 多 Agent 通过黑板协作——每个 Agent 都能看到前面 Agent 的输出
board.write("task", "分析 AI Agent 市场趋势", "manager")
researcher.think_and_act("收集市场数据和研究报告")
analyst.think_and_act("分析研究员收集的数据，提取关键趋势")
writer.think_and_act("基于分析结果撰写简报")

print(board.read("writer_output"))
```

### 事件驱动通信

Agent 发布事件，其他感兴趣的 Agent 订阅并响应。这是最松耦合的模式——发布者不需要知道谁在监听。

```python
class EventBus:
    """事件总线：发布-订阅模式"""

    def __init__(self):
        self.subscribers: dict[str, list[Callable]] = {}
        self.event_log: list[dict] = []

    def subscribe(self, event_type: str, handler: Callable):
        """订阅某类事件"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    def publish(self, event_type: str, data: dict, source: str):
        """发布事件，所有订阅者都会收到"""
        event = {
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": datetime.now().isoformat(),
        }
        self.event_log.append(event)

        responses = []
        for handler in self.subscribers.get(event_type, []):
            result = handler(event)
            if result:
                responses.append(result)
        return responses


# 使用示例
event_bus = EventBus()

def on_research_complete(event):
    print(f"[Analyst] 收到研究完成事件，开始分析...")
    return "分析结果: ..."

def on_research_complete_notify(event):
    print(f"[Logger] 记录：研究已完成，来源: {event['source']}")

event_bus.subscribe("research_complete", on_research_complete)
event_bus.subscribe("research_complete", on_research_complete_notify)

# 研究员完成后发布事件，分析师和日志器都会响应
event_bus.publish("research_complete", {"findings": "..."}, source="researcher")
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

```python
import json

def decompose_task(task: str, available_agents: list[str]) -> list[dict]:
    """将任务分解为子任务并分配给 Agent"""
    agents_desc = "\n".join([f"- {a}" for a in available_agents])

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""将以下任务分解为子任务，并分配给合适的 Agent。

任务：{task}

可用 Agent：
{agents_desc}

要求：
1. 每个子任务应该是独立可执行的
2. 标明子任务之间的依赖关系
3. 估计每个子任务的优先级

返回 JSON：
{{"subtasks": [
  {{"id": 1, "description": "子任务描述", "agent": "Agent 名称",
    "dependencies": [], "priority": "high/medium/low"}},
]}}"""
        }]
    )
    return json.loads(response.content[0].text)["subtasks"]
```

### 带依赖管理的任务调度器

关键是处理子任务之间的依赖关系——任务 C 依赖任务 A 和 B 的结果，那 C 必须等 A 和 B 都完成才能开始。

```python
from dataclasses import dataclass, field
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class SubTask:
    """子任务定义"""
    id: int
    description: str
    agent: str
    dependencies: list[int] = field(default_factory=list)
    priority: str = "medium"
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""

class TaskScheduler:
    """任务调度器：管理子任务的依赖和执行顺序"""

    def __init__(self):
        self.tasks: dict[int, SubTask] = {}
        self.agent_load: dict[str, int] = {}

    def add_tasks(self, subtasks: list[dict]):
        """批量添加子任务"""
        for st in subtasks:
            task = SubTask(**st)
            self.tasks[task.id] = task
            self.agent_load.setdefault(task.agent, 0)

    def get_ready_tasks(self) -> list[SubTask]:
        """获取可以立即执行的任务（所有依赖已满足）"""
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            # 检查所有依赖是否已完成
            deps_met = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
                if dep_id in self.tasks
            )
            if deps_met:
                ready.append(task)

        # 按优先级排序
        priority_order = {"high": 0, "medium": 1, "low": 2}
        ready.sort(key=lambda t: priority_order.get(t.priority, 1))
        return ready

    def execute_task(self, task: SubTask) -> str:
        """执行单个任务"""
        task.status = TaskStatus.RUNNING
        self.agent_load[task.agent] += 1

        try:
            # 收集依赖任务的结果作为上下文
            dep_context = ""
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                if dep_task and dep_task.result:
                    dep_context += f"\n[任务{dep_id}的结果]: {dep_task.result}"

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": f"任务：{task.description}{dep_context}"
                }]
            )
            task.result = response.content[0].text
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = f"错误: {e}"
        finally:
            self.agent_load[task.agent] -= 1

        return task.result

    def run_all(self) -> dict[int, str]:
        """执行所有任务（尊重依赖关系）"""
        results = {}
        max_iterations = len(self.tasks) * 2

        for _ in range(max_iterations):
            ready = self.get_ready_tasks()
            if not ready:
                all_done = all(
                    t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                    for t in self.tasks.values()
                )
                if all_done:
                    break
                continue

            for task in ready:
                print(f"执行任务 {task.id}: {task.description[:50]}...")
                result = self.execute_task(task)
                results[task.id] = result
                print(f"  -> 完成 (状态: {task.status.value})")

        return results


# 使用示例
scheduler = TaskScheduler()
scheduler.add_tasks([
    {"id": 1, "description": "调研 RAG 技术的基本概念",
     "agent": "researcher", "dependencies": [], "priority": "high"},
    {"id": 2, "description": "调研 RAG 的最新进展",
     "agent": "researcher", "dependencies": [], "priority": "high"},
    {"id": 3, "description": "分析 RAG 的优缺点",
     "agent": "analyst", "dependencies": [1, 2], "priority": "medium"},
    {"id": 4, "description": "撰写技术报告",
     "agent": "writer", "dependencies": [3], "priority": "low"},
])
results = scheduler.run_all()
# 任务 1 和 2 没有依赖，可以先执行；
# 任务 3 等 1+2 完成；任务 4 等 3 完成
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

```python
from collections import Counter

class VotingConsensus:
    """投票式共识机制"""

    def __init__(self, n_agents: int = 3):
        self.n_agents = n_agents

    def get_votes(self, question: str,
                  options: list[str] = None) -> list[dict]:
        """让多个 Agent 独立投票"""
        votes = []
        for i in range(self.n_agents):
            prompt = f"请回答以下问题。\n问题：{question}\n"
            if options:
                prompt += f"选项：{json.dumps(options, ensure_ascii=False)}\n"
            prompt += '\n返回 JSON：{"answer": "你的答案", "confidence": 0.0-1.0, "reasoning": "推理过程"}'

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                temperature=0.7,  # 稍高温度确保多样性
                messages=[{"role": "user", "content": prompt}]
            )
            vote = json.loads(response.content[0].text)
            vote["agent_id"] = i
            votes.append(vote)
        return votes

    def reach_consensus(self, question: str,
                        options: list[str] = None) -> dict:
        """通过投票达成共识"""
        votes = self.get_votes(question, options)

        answers = [v["answer"] for v in votes]
        counter = Counter(answers)
        majority_answer, majority_count = counter.most_common(1)[0]

        consensus_ratio = majority_count / len(votes)
        avg_confidence = sum(
            v["confidence"] for v in votes if v["answer"] == majority_answer
        ) / max(majority_count, 1)

        return {
            "consensus_answer": majority_answer,
            "consensus_ratio": consensus_ratio,
            "avg_confidence": avg_confidence,
            "is_unanimous": consensus_ratio == 1.0,
            "all_votes": votes,
        }


# 使用
voter = VotingConsensus(n_agents=5)
result = voter.reach_consensus(
    "Python 和 JavaScript 哪个更适合初学者？",
    options=["Python", "JavaScript", "都适合"]
)
print(f"共识: {result['consensus_answer']} "
      f"({result['consensus_ratio']*100:.0f}% 一致)")
```

### 仲裁 Agent

当投票无法达成明确共识时，引入一个独立的"仲裁者"来做最终判断。

```python
class ArbitrationConsensus:
    """仲裁式共识：由裁判 Agent 做最终判断"""

    def __init__(self, n_debaters: int = 3):
        self.n_debaters = n_debaters

    def collect_opinions(self, question: str) -> list[dict]:
        """收集各 Agent 的观点"""
        opinions = []
        perspectives = ["乐观主义者", "谨慎分析师", "实用主义者"]

        for i in range(min(self.n_debaters, len(perspectives))):
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                system=f"你是一个{perspectives[i]}，请从你的角度分析问题。",
                messages=[{"role": "user", "content": question}]
            )
            opinions.append({
                "perspective": perspectives[i],
                "opinion": response.content[0].text,
            })
        return opinions

    def arbitrate(self, question: str, opinions: list[dict]) -> dict:
        """裁判做最终判断"""
        opinions_text = "\n\n".join([
            f"[{o['perspective']}]:\n{o['opinion']}" for o in opinions
        ])

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system="你是一个公正的裁判。综合分析各方观点，给出客观的最终结论。",
            messages=[{
                "role": "user",
                "content": f"问题：{question}\n\n各方观点：\n{opinions_text}\n\n"
                           f"请：1. 分析各方优缺点 2. 指出共识和分歧 "
                           f"3. 给出最终裁决 4. 说明理由"
            }]
        )
        return {
            "opinions": opinions,
            "arbitration": response.content[0].text,
        }

    def resolve(self, question: str) -> dict:
        opinions = self.collect_opinions(question)
        return self.arbitrate(question, opinions)
```

### 综合共识框架：分级策略

将多种共识机制组合使用，从快到慢逐步升级。

```python
class ConsensusFramework:
    """综合共识框架：分级策略"""

    def __init__(self):
        self.voting = VotingConsensus(n_agents=3)
        self.arbiter = ArbitrationConsensus()

    def resolve(self, question: str) -> dict:
        """分级共识策略"""
        # Level 1: 多 Agent 投票（快速）
        vote_result = self.voting.reach_consensus(question)
        if vote_result["consensus_ratio"] > 0.6:
            return {
                "method": "voting",
                "answer": vote_result["consensus_answer"],
                "confidence": vote_result["avg_confidence"],
            }

        # Level 2: 仲裁（更可靠但更慢）
        arb_result = self.arbiter.resolve(question)
        return {
            "method": "arbitration",
            "answer": arb_result["arbitration"],
            "confidence": 0.7,
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
