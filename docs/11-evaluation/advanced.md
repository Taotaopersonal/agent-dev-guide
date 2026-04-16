# 评估高级：可观测性与生产监控

::: tip 学习目标
- 理解 Agent 可观测性的三大支柱：Traces、Metrics、Logs
- 掌握自建 Tracing 方案的实现（Span/Trace 数据结构、TracedAgent）
- 学会使用 LangSmith 和 Phoenix 两个主流可观测性平台
- 能够搭建生产级的监控面板和回归测试追踪器

**学完你能做到：** 为你的 Agent 系统添加完整的执行轨迹追踪，搭建一个包含延迟/Token/错误率的监控面板，以及一个能检测性能退化的回归测试系统。
:::

## 为什么需要可观测性

Agent 的执行过程就像一个黑盒：输入一个问题，输出一个答案，但中间发生了什么？调用了几次 LLM？用了哪些工具？每步花了多少时间和 Token？

没有可观测性，你无法：
- **定位性能瓶颈**：哪一步最慢？
- **排查质量问题**：哪一步出了错？
- **优化成本**：哪些调用是冗余的？
- **理解决策过程**：Agent 为什么选择调用这个工具而不是那个？

## 自建 Tracing 方案

我们先从零搭一个 Tracing 系统，理解核心概念后再看第三方工具。

### Span 和 Trace 数据结构

```python
from dataclasses import dataclass, field
import time
import json

@dataclass
class Span:
    """一个追踪 Span（执行步骤）

    Span 是 Tracing 的基本单位，表示一次操作（LLM 调用、工具执行等）。
    Span 可以嵌套——一个 Agent 运行的 root span 包含多个 LLM 调用和工具调用的子 span。
    """
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: float = 0
    input_data: dict = field(default_factory=dict)
    output_data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)
    error: str = None

    def end(self):
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 1),
            "input": self.input_data,
            "output": {k: str(v)[:200] for k, v in self.output_data.items()},
            "metadata": self.metadata,
            "error": self.error,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class Trace:
    """一次完整的 Agent 执行追踪"""
    trace_id: str
    root_span: Span = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "metadata": self.metadata,
            "root": self.root_span.to_dict() if self.root_span else None,
            "total_duration_ms": (self.root_span.duration_ms
                                  if self.root_span else 0),
        }
```

### 带追踪的 Agent

```python
import uuid
import anthropic

client = anthropic.Anthropic()

class TracedAgent:
    """带追踪的 Agent：每次执行都记录完整的 Trace"""

    def __init__(self, name: str):
        self.name = name
        self.traces: list[Trace] = []

    def run(self, question: str) -> tuple[str, Trace]:
        """运行并追踪"""
        trace = Trace(
            trace_id=str(uuid.uuid4())[:8],
            metadata={"agent": self.name, "question": question[:100]},
        )

        root = Span(
            name="agent_run",
            input_data={"question": question},
        )
        trace.root_span = root

        messages = [{"role": "user", "content": question}]
        tools = [{
            "name": "search",
            "description": "搜索信息",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        }]

        answer = ""
        step = 0
        while step < 5:
            step += 1

            # 追踪 LLM 调用
            llm_span = Span(
                name=f"llm_call_{step}",
                input_data={"messages_count": len(messages)},
            )

            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    tools=tools,
                    messages=messages,
                )
                llm_span.output_data = {
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
                llm_span.metadata = {
                    "model": "claude-sonnet-4-20250514",
                    "total_tokens": (response.usage.input_tokens +
                                    response.usage.output_tokens),
                }
            except Exception as e:
                llm_span.error = str(e)
            finally:
                llm_span.end()
                root.children.append(llm_span)

            if response.stop_reason == "end_turn":
                answer = response.content[0].text
                root.output_data = {"answer": answer}
                break

            # 追踪工具调用
            for block in response.content:
                if block.type == "tool_use":
                    tool_span = Span(
                        name=f"tool_{block.name}",
                        input_data=block.input,
                    )
                    try:
                        result = f"搜索 '{block.input.get('query', '')}' 的结果..."
                        tool_span.output_data = {"result": result}
                    except Exception as e:
                        tool_span.error = str(e)
                        result = f"工具错误: {e}"
                    finally:
                        tool_span.end()
                        root.children.append(tool_span)

                    messages.append({
                        "role": "assistant", "content": response.content
                    })
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }]
                    })

        root.end()
        self.traces.append(trace)
        return answer, trace

    def print_trace(self, trace: Trace):
        """打印追踪信息——直观展示每步的耗时和 Token 消耗"""
        data = trace.to_dict()
        print(f"\n{'='*60}")
        print(f"Trace ID: {data['trace_id']}")
        print(f"Total Duration: {data['total_duration_ms']:.0f}ms")
        print(f"{'='*60}")

        if data["root"]:
            self._print_span(data["root"], indent=0)

    def _print_span(self, span: dict, indent: int):
        prefix = "  " * indent
        duration = span["duration_ms"]
        error = " [ERROR]" if span["error"] else ""
        tokens = span.get("metadata", {}).get("total_tokens", "")
        tokens_str = f" ({tokens} tokens)" if tokens else ""
        print(f"{prefix}|- {span['name']}: {duration:.0f}ms"
              f"{tokens_str}{error}")
        for child in span.get("children", []):
            self._print_span(child, indent + 1)
```

输出示例：

```
============================================================
Trace ID: a1b2c3d4
Total Duration: 2340ms
============================================================
|- agent_run: 2340ms
  |- llm_call_1: 1200ms (580 tokens)
  |- tool_search: 150ms
  |- llm_call_2: 990ms (720 tokens)
```

一眼就能看出：LLM 调用是性能瓶颈，工具调用很快。这就是可观测性的价值。

## 使用 LangSmith

LangSmith 是 LangChain 官方的追踪和评估平台，也是目前最成熟的 LLM 可观测性工具之一。

```python
# 安装和配置
# pip install langsmith
# export LANGCHAIN_TRACING_V2=true
# export LANGCHAIN_API_KEY=your_key
# export LANGCHAIN_PROJECT=my_agent_project

from langsmith import traceable
from langchain_openai import ChatOpenAI

@traceable(name="my_agent")
def my_agent(question: str) -> str:
    """LangSmith 会自动追踪这个函数的输入输出"""
    model = ChatOpenAI(model="gpt-4o-mini")
    response = model.invoke(question)
    return response.content

@traceable(name="retrieval")
def retrieve_docs(query: str) -> list[str]:
    """子步骤也会被追踪"""
    return [f"Document about {query}"]

@traceable(name="rag_pipeline")
def rag_query(question: str) -> str:
    docs = retrieve_docs(question)
    context = "\n".join(docs)
    return my_agent(f"Context: {context}\nQuestion: {question}")

# 调用后，追踪数据自动上传到 LangSmith 平台
result = rag_query("什么是向量数据库？")
```

LangSmith 的核心价值在于它的 Web UI——你可以在浏览器中看到每次 Agent 运行的完整执行树，包括每步的输入输出、耗时、Token 消耗，还能按项目分组管理和对比不同版本。

## 使用 Phoenix (Arize)

Phoenix 是一个开源的 LLM 可观测性工具，可以本地部署，不需要将数据发送到第三方。

```python
# pip install arize-phoenix openinference-instrumentation-openai

import phoenix as px
from openinference.instrumentation.openai import OpenAIInstrumentor

# 启动 Phoenix 本地服务
session = px.launch_app()

# 自动注入追踪——之后所有 OpenAI 调用都会被追踪
OpenAIInstrumentor().instrument()

from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
)

# 打开 Phoenix UI 查看追踪
print(f"Phoenix UI: {session.url}")
```

::: tip LangSmith vs Phoenix 怎么选
- **LangSmith**：云服务，功能最全，适合团队使用，有免费额度
- **Phoenix**：开源，本地部署，数据不出域，适合隐私要求高的场景
- 两者都支持 OpenTelemetry 标准，可以互相迁移
:::

## 关键指标监控面板

```python
class MetricsDashboard:
    """指标监控面板：聚合多次 Trace 数据"""

    def __init__(self):
        self.traces: list[dict] = []

    def add_trace(self, trace: Trace):
        self.traces.append(trace.to_dict())

    def get_metrics(self) -> dict:
        """计算关键指标"""
        if not self.traces:
            return {}

        durations = [t["total_duration_ms"] for t in self.traces]
        all_tokens = []
        llm_calls = []
        tool_calls = []
        errors = 0

        for trace in self.traces:
            root = trace.get("root", {})
            for child in root.get("children", []):
                if child["name"].startswith("llm_call"):
                    llm_calls.append(child["duration_ms"])
                    tokens = child.get("metadata", {}).get("total_tokens", 0)
                    if tokens:
                        all_tokens.append(tokens)
                elif child["name"].startswith("tool_"):
                    tool_calls.append(child["duration_ms"])
                if child.get("error"):
                    errors += 1

        return {
            "total_traces": len(self.traces),
            "latency": {
                "avg_ms": sum(durations) / len(durations),
                "p50_ms": sorted(durations)[len(durations) // 2],
                "p95_ms": (sorted(durations)[int(len(durations) * 0.95)]
                          if len(durations) > 20 else max(durations)),
                "max_ms": max(durations),
            },
            "tokens": {
                "avg_per_trace": (sum(all_tokens) / len(all_tokens)
                                  if all_tokens else 0),
                "total": sum(all_tokens),
            },
            "calls": {
                "avg_llm_calls": len(llm_calls) / len(self.traces),
                "avg_tool_calls": len(tool_calls) / len(self.traces),
            },
            "errors": {
                "total": errors,
                "rate": errors / max(len(llm_calls) + len(tool_calls), 1),
            },
        }
```

## 回归测试追踪

每次修改 Agent 后运行回归测试，确保改进不会引入退化。

```python
class RegressionTracker:
    """回归测试追踪器：检测版本间的性能变化"""

    def __init__(self, history_path: str = "./eval_history.json"):
        self.history_path = history_path
        self.history = self._load_history()

    def _load_history(self) -> list:
        try:
            with open(self.history_path) as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def record(self, version: str, report: dict):
        """记录一次评估结果"""
        import time
        entry = {
            "version": version,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": report["summary"],
        }
        self.history.append(entry)
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def compare(self) -> dict:
        """与上一版本对比"""
        if len(self.history) < 2:
            return {"message": "不够历史数据进行对比"}

        current = self.history[-1]
        previous = self.history[-2]

        comparison = {}
        for metric in ["avg_auto_score", "avg_llm_score",
                       "avg_latency_ms", "error_rate"]:
            curr_val = current["summary"].get(metric, 0)
            prev_val = previous["summary"].get(metric, 0)
            delta = curr_val - prev_val
            # 分数上升=改进，延迟和错误率下降=改进
            is_better = (delta > 0 if metric not in ("avg_latency_ms", "error_rate")
                        else delta < 0)
            comparison[metric] = {
                "current": curr_val,
                "previous": prev_val,
                "delta": round(delta, 4),
                "improved": is_better,
            }

        return comparison
```

::: tip 可观测性的三大支柱
1. **Traces**：完整的执行路径追踪——调试单次执行
2. **Metrics**：聚合的性能指标（P50/P95/P99、Token、错误率）——发现系统性问题
3. **Logs**：详细的运行时日志——深入排查具体问题

Agent 的可观测性需要三者结合才能全面覆盖。
:::

## 小结

- 可观测性是 Agent 生产化的必备能力，没有它你就是在盲飞
- 自建 Tracing 适合简单场景和学习目的，LangSmith/Phoenix 适合生产环境
- 关键指标：延迟（P50/P95/P99）、Token 消耗、错误率、工具调用效率
- 回归测试追踪每次迭代的效果变化，防止改进引入退化
- 建议在 CI/CD 中集成自动评估和回归测试

## 练习

1. 为你的 Agent 添加完整的 Tracing：记录每次 LLM 调用和工具调用的输入输出、耗时和 Token 消耗。
2. 搭建一个 LangSmith 或 Phoenix 项目，观察 Agent 的执行轨迹。
3. 实现一个告警系统：当 P95 延迟超过阈值或错误率超过 5% 时打印告警信息。

## 参考资源

- [LangSmith Documentation](https://docs.smith.langchain.com/) -- LangSmith 官方文档
- [Phoenix (Arize) Documentation](https://docs.arize.com/phoenix/) -- Phoenix 可观测性平台
- [OpenTelemetry for LLM Applications](https://opentelemetry.io/) -- OpenTelemetry 标准
- [Langfuse: Open Source LLM Observability](https://langfuse.com/docs) -- 开源可观测性平台
- [Braintrust: LLM Evaluation and Observability](https://www.braintrust.dev/docs) -- 评估和观测平台
- [Harrison Chase: Observability for LLM Apps](https://www.youtube.com/watch?v=Uv8Y8GgYTuA) -- LangChain 创始人讲解
