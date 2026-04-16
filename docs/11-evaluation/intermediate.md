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

```python
import anthropic
import json

client = anthropic.Anthropic()

JUDGE_PROMPT = """你是一个严格公正的 AI 评估专家。请根据以下标准评估回答的质量。

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
{{
  "scores": {{
    "accuracy": 1-5,
    "completeness": 1-5,
    "clarity": 1-5,
    "relevance": 1-5
  }},
  "overall_score": 1-5,
  "strengths": ["优点1", "优点2"],
  "weaknesses": ["不足1", "不足2"],
  "reasoning": "评分理由"
}}"""
```

### LLM Judge 实现

```python
class LLMJudge:
    """LLM-as-Judge 评估器"""

    def __init__(self, judge_model: str = "claude-sonnet-4-20250514"):
        self.judge_model = judge_model

    def evaluate(self, question: str, answer: str,
                 reference: str = "无参考答案") -> dict:
        """评估单个回答"""
        prompt = JUDGE_PROMPT.format(
            question=question,
            reference=reference,
            answer=answer,
        )
        response = client.messages.create(
            model=self.judge_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.content[0].text)

    def compare(self, question: str,
                answer_a: str, answer_b: str) -> dict:
        """对比评估两个回答"""
        response = client.messages.create(
            model=self.judge_model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"""对比以下两个回答，判断哪个更好。

问题：{question}

回答 A：{answer_a}

回答 B：{answer_b}

返回 JSON：
{{
  "winner": "A" 或 "B" 或 "TIE",
  "score_a": 1-5,
  "score_b": 1-5,
  "reasoning": "判断理由"
}}"""
            }]
        )
        return json.loads(response.content[0].text)


# 使用
judge = LLMJudge()

result = judge.evaluate(
    question="解释什么是 RAG",
    answer="RAG 就是一种搜索技术",
    reference="RAG（检索增强生成）是将外部知识检索与 LLM 生成结合的技术",
)
print(f"总分: {result['overall_score']}/5")
print(f"优点: {result['strengths']}")
print(f"不足: {result['weaknesses']}")
```

### 针对特定场景的评估

不同场景需要不同的评估标准。RAG 系统要评估"忠实度"（是否忠于检索到的文档），代码生成要评估"正确性"和"效率"。

```python
RAG_JUDGE_PROMPT = """评估以下 RAG 系统的回答质量。

问题：{question}
检索到的文档：{retrieved_docs}
系统回答：{answer}

评估维度：
1. **忠实度** (Faithfulness)：回答是否忠实于文档（不编造信息）
2. **相关性** (Relevance)：回答是否与问题相关
3. **信息完整性** (Completeness)：是否利用了文档中的关键信息
4. **无幻觉** (No Hallucination)：是否包含文档中没有的错误信息

返回 JSON：
{{
  "faithfulness": 1-5,
  "relevance": 1-5,
  "completeness": 1-5,
  "hallucination_check": true/false,
  "hallucinated_content": ["幻觉内容1"],
  "overall_score": 1-5,
  "reasoning": "评估理由"
}}"""
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

```python
class RobustLLMJudge:
    """带偏差缓解的 LLM Judge"""

    def __init__(self):
        self.judge = LLMJudge()

    def compare_robust(self, question: str,
                       answer_a: str, answer_b: str) -> dict:
        """位置交换法：两次评估，交换 A/B 位置"""
        # 第一次：A 在前
        result_1 = self.judge.compare(question, answer_a, answer_b)
        # 第二次：B 在前
        result_2 = self.judge.compare(question, answer_b, answer_a)

        # 综合判断——如果交换后结果一致，说明没有位置偏差
        winner_1 = result_1["winner"]
        winner_2 = ("A" if result_2["winner"] == "B"
                    else ("B" if result_2["winner"] == "A" else "TIE"))

        if winner_1 == winner_2:
            return {"winner": winner_1, "confidence": "high",
                    "method": "position_swap"}
        else:
            return {"winner": "TIE", "confidence": "low",
                    "method": "position_swap",
                    "note": "两次评估结果不一致，可能存在位置偏差"}

    def evaluate_multi_judge(self, question: str, answer: str,
                             reference: str = "",
                             n_judges: int = 3) -> dict:
        """多 Judge 评估：多次评估取平均"""
        scores = []
        for _ in range(n_judges):
            result = self.judge.evaluate(question, answer, reference)
            scores.append(result["overall_score"])

        avg_score = sum(scores) / len(scores)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)

        return {
            "avg_score": round(avg_score, 2),
            "variance": round(variance, 2),
            "all_scores": scores,
            "is_reliable": variance < 1.0,
        }
```

## 构建测试集

好的测试集是评估的基础。每条用例都应该有明确的分类、难度标注和评估标准。

```python
from dataclasses import dataclass, field, asdict
from enum import Enum

class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class Category(Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    CREATIVE = "creative"
    TOOL_USE = "tool_use"
    MULTI_STEP = "multi_step"

@dataclass
class TestCase:
    """测试用例"""
    id: str
    question: str
    expected_answer: str
    category: str = "factual"
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    evaluation_criteria: list[str] = field(default_factory=list)

class TestSuiteBuilder:
    """测试集构建器"""

    def __init__(self, name: str):
        self.name = name
        self.cases: list[TestCase] = []

    def add_case(self, **kwargs) -> "TestSuiteBuilder":
        case = TestCase(**kwargs)
        self.cases.append(case)
        return self

    def save(self, path: str):
        data = {
            "name": self.name,
            "total_cases": len(self.cases),
            "cases": [asdict(c) for c in self.cases],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "TestSuiteBuilder":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        builder = cls(data["name"])
        for c in data["cases"]:
            builder.add_case(**c)
        return builder

    def stats(self) -> dict:
        """测试集统计"""
        from collections import Counter
        return {
            "total": len(self.cases),
            "by_category": dict(Counter(c.category for c in self.cases)),
            "by_difficulty": dict(Counter(c.difficulty for c in self.cases)),
        }


# 构建示例测试集
suite = TestSuiteBuilder("RAG Agent 评估集")

suite.add_case(
    id="fact_001",
    question="Python 是什么类型的编程语言？",
    expected_answer="解释型",
    category="factual",
    difficulty="easy",
    tags=["python", "basic"],
    evaluation_criteria=["提到'解释型'", "提到'面向对象'（加分）"],
)

suite.add_case(
    id="reason_001",
    question="如果一个 API 的 P99 延迟是 200ms，P50 是 50ms，说明了什么？",
    expected_answer="尾部延迟较高，有少量请求耗时显著增大",
    category="reasoning",
    difficulty="medium",
    tags=["performance"],
    evaluation_criteria=["提到尾部延迟", "解释 P99 和 P50 的含义"],
)

suite.save("./test_suite.json")
print(suite.stats())
```

## 自动化评估流水线

把上面的组件串起来：自动运行测试集、用规则检查 + LLM Judge 评估、生成结构化报告。

```python
import time
import json
import anthropic

client = anthropic.Anthropic()

class EvaluationPipeline:
    """自动化评估流水线"""

    def __init__(self, agent_fn, test_suite_path: str):
        self.agent_fn = agent_fn
        self.suite = TestSuiteBuilder.load(test_suite_path)

    def run(self, use_llm_judge: bool = True) -> dict:
        """运行完整评估"""
        results = []

        for case in self.suite.cases:
            print(f"评估 [{case.id}] {case.question[:40]}...", end=" ")
            start = time.time()

            try:
                actual_answer = self.agent_fn(case.question)
                latency = (time.time() - start) * 1000

                auto_score = self._auto_check(actual_answer, case)

                llm_score = None
                if use_llm_judge:
                    llm_score = self._llm_judge(case, actual_answer)

                result = {
                    "case_id": case.id,
                    "category": case.category,
                    "difficulty": case.difficulty,
                    "auto_score": auto_score,
                    "llm_score": llm_score,
                    "latency_ms": round(latency, 1),
                    "actual_answer": actual_answer[:300],
                    "error": None,
                }
                print(f"auto={auto_score:.1f} llm={llm_score}")

            except Exception as e:
                result = {
                    "case_id": case.id,
                    "category": case.category,
                    "auto_score": 0,
                    "llm_score": 0,
                    "error": str(e),
                }
                print(f"ERROR: {e}")

            results.append(result)

        return self._generate_report(results)

    def _auto_check(self, actual: str, case: TestCase) -> float:
        """基于规则的自动检查"""
        criteria = case.evaluation_criteria
        if not criteria:
            return 1.0 if case.expected_answer.lower() in actual.lower() else 0.0

        score = 0
        for criterion in criteria:
            key_terms = (criterion.replace("提到", "")
                        .replace("'", "").replace("（加分）", "").strip())
            if key_terms.lower() in actual.lower():
                score += 1
        return score / len(criteria) if criteria else 0

    def _llm_judge(self, case: TestCase, actual: str) -> float:
        """用 LLM 评估答案质量（0-1 分）"""
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{
                "role": "user",
                "content": f"""评估以下回答的质量（0-1 分）。

问题：{case.question}
预期答案要点：{case.expected_answer}
评估标准：{json.dumps(case.evaluation_criteria, ensure_ascii=False)}
实际回答：{actual[:500]}

只返回一个 0 到 1 之间的数字："""
            }]
        )
        try:
            return float(response.content[0].text.strip())
        except ValueError:
            return 0.5

    def _generate_report(self, results: list[dict]) -> dict:
        """生成评估报告"""
        total = len(results)
        auto_scores = [r["auto_score"] for r in results
                       if r["auto_score"] is not None]
        llm_scores = [r["llm_score"] for r in results
                      if r["llm_score"] is not None]
        latencies = [r["latency_ms"] for r in results
                     if r.get("latency_ms")]
        errors = [r for r in results if r.get("error")]

        report = {
            "summary": {
                "total_cases": total,
                "avg_auto_score": (sum(auto_scores) / len(auto_scores)
                                   if auto_scores else 0),
                "avg_llm_score": (sum(llm_scores) / len(llm_scores)
                                  if llm_scores else 0),
                "avg_latency_ms": (sum(latencies) / len(latencies)
                                   if latencies else 0),
                "error_rate": len(errors) / total if total else 0,
            },
            "by_category": {},
            "worst_cases": sorted(
                results, key=lambda r: r.get("llm_score", 0)
            )[:5],
            "details": results,
        }

        # 按类别统计
        for category in set(r["category"] for r in results):
            cat_results = [r for r in results if r["category"] == category]
            cat_scores = [r["llm_score"] for r in cat_results
                         if r["llm_score"] is not None]
            report["by_category"][category] = {
                "count": len(cat_results),
                "avg_score": (sum(cat_scores) / len(cat_scores)
                             if cat_scores else 0),
            }

        return report
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
