# Tool Use 高级

> **学习目标**：掌握动态工具生成、工具结果缓存、工具优先级路由和安全沙箱等生产级技巧，将 Tool Use 从"能用"提升到"好用"。

学完本节，你将能够：
- 根据上下文动态生成和注册工具
- 用缓存避免重复的工具调用
- 设计工具路由策略优化大型工具集的使用
- 构建安全沙箱保护工具执行环境

## 动态工具生成

入门和进阶篇中，工具列表都是静态定义的。但在实际场景中，可用的工具可能随上下文变化——比如用户连接了数据库后才有 SQL 工具，上传了文件后才有文件分析工具。

### 根据上下文加载工具

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

interface ToolEntry {
  schema: Anthropic.Tool;
  func: (...args: any[]) => any;
}

class DynamicToolAgent {
  /** 支持动态工具注册的 Agent */
  private systemPrompt: string;
  private baseTools: Map<string, ToolEntry> = new Map();       // 始终可用的基础工具
  private dynamicTools: Map<string, ToolEntry> = new Map();    // 动态加载的工具
  private messages: Anthropic.MessageParam[] = [];

  constructor(systemPrompt: string = "") {
    this.systemPrompt = systemPrompt;
  }

  registerBaseTool(
    name: string, description: string,
    parameters: Anthropic.Tool.InputSchema, func: (...args: any[]) => any
  ): void {
    /** 注册始终可用的基础工具 */
    this.baseTools.set(name, {
      schema: { name, description, input_schema: parameters },
      func,
    });
  }

  addDynamicTool(
    name: string, description: string,
    parameters: Anthropic.Tool.InputSchema, func: (...args: any[]) => any
  ): void {
    /** 动态添加工具（上下文触发） */
    this.dynamicTools.set(name, {
      schema: { name, description, input_schema: parameters },
      func,
    });
    console.log(`[动态工具] 已加载: ${name}`);
  }

  removeDynamicTool(name: string): void {
    /** 移除动态工具 */
    if (this.dynamicTools.has(name)) {
      this.dynamicTools.delete(name);
      console.log(`[动态工具] 已卸载: ${name}`);
    }
  }

  get allTools(): Map<string, ToolEntry> {
    /** 合并基础工具和动态工具 */
    return new Map([...this.baseTools, ...this.dynamicTools]);
  }

  get toolSchemas(): Anthropic.Tool[] {
    return [...this.allTools.values()].map((t) => t.schema);
  }

  async run(userInput: string): Promise<string> {
    this.messages.push({ role: "user", content: userInput });
    for (let i = 0; i < 10; i++) {
      const response = await client.messages.create({
        model: "claude-sonnet-4-20250514",
        max_tokens: 4096,
        system: this.systemPrompt,
        tools: this.toolSchemas,
        messages: this.messages,
      });
      if (response.stop_reason === "end_turn") {
        this.messages.push({ role: "assistant", content: response.content });
        return response.content
          .filter((b): b is Anthropic.TextBlock => b.type === "text")
          .map((b) => b.text)
          .join("");
      }

      if (response.stop_reason === "tool_use") {
        this.messages.push({ role: "assistant", content: response.content });
        const results: Anthropic.ToolResultBlockParam[] = [];
        for (const block of response.content) {
          if (block.type === "tool_use") {
            const tool = this.allTools.get(block.name);
            let result: any;
            if (tool) {
              result = tool.func(block.input);
            } else {
              result = { error: `工具不存在: ${block.name}` };
            }
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
    return "达到最大迭代次数";
  }
}
```

使用场景示例：用户上传 CSV 文件后，动态添加数据分析工具：

```typescript
const agent = new DynamicToolAgent("你是一个数据分析助手。");

// 始终可用的工具
agent.registerBaseTool(
  "list_files", "列出可用的数据文件",
  { type: "object" as const, properties: {} },
  () => ({ files: ["sales.csv", "users.csv"] })
);

// 用户选择文件后动态加载分析工具
function onFileSelected(filename: string): void {
  function analyze(input: { column: string }): Record<string, any> {
    // 实际实现：读取 CSV 并分析指定列
    return { file: filename, column: input.column, mean: 42.5, count: 100 };
  }

  agent.addDynamicTool(
    "analyze_column",
    `分析 ${filename} 的指定列，返回统计信息（均值、计数等）`,
    {
      type: "object" as const,
      properties: { column: { type: "string", description: "列名" } },
      required: ["column"],
    },
    analyze
  );
}

// 模拟用户操作
onFileSelected("sales.csv");
const result = await agent.run("分析 sales.csv 的 revenue 列");
```

## 工具结果缓存

如果模型多次调用同一个工具、传相同的参数，每次都真正执行一遍就很浪费。比如在多轮对话中反复查询同一个城市的天气，或者链式调用中多次获取同一个用户的信息。

### 基于参数的结果缓存

```typescript
import { createHash } from "crypto";

interface CacheEntry {
  result: string;
  timestamp: number;
}

class ToolCache {
  /** 工具结果缓存 */
  private cache: Map<string, CacheEntry> = new Map();
  private ttl: number;

  /**
   * @param ttl 缓存过期时间（秒），默认 5 分钟
   */
  constructor(ttl: number = 300) {
    this.ttl = ttl;
  }

  private makeKey(toolName: string, params: Record<string, any>): string {
    /** 生成缓存键 */
    const raw = `${toolName}:${JSON.stringify(params, Object.keys(params).sort())}`;
    return createHash("md5").update(raw).digest("hex");
  }

  get(toolName: string, params: Record<string, any>): string | null {
    /** 获取缓存结果，过期返回 null */
    const key = this.makeKey(toolName, params);
    const entry = this.cache.get(key);
    if (entry) {
      if (Date.now() / 1000 - entry.timestamp < this.ttl) {
        console.log(`  [缓存命中] ${toolName}`);
        return entry.result;
      } else {
        this.cache.delete(key);
      }
    }
    return null;
  }

  set(toolName: string, params: Record<string, any>, result: string): void {
    /** 存入缓存 */
    const key = this.makeKey(toolName, params);
    this.cache.set(key, { result, timestamp: Date.now() / 1000 });
  }

  invalidate(toolName?: string): void {
    /** 清除缓存（可指定工具名） */
    // 简化处理：清除所有缓存
    this.cache.clear();
  }
}

function executeWithCache(
  cache: ToolCache, toolName: string,
  params: Record<string, any>, func: (...args: any[]) => any
): string {
  /** 带缓存的工具执行 */
  // 先查缓存
  const cached = cache.get(toolName, params);
  if (cached !== null) {
    return cached;
  }

  // 执行工具
  const result = func(params);
  const resultStr = JSON.stringify(result);

  // 存入缓存
  cache.set(toolName, params, resultStr);
  return resultStr;
}
```

::: warning 哪些工具适合缓存
- **适合缓存**：天气查询、用户信息、搜索结果（短 TTL）、文件元信息
- **不适合缓存**：写入操作、有副作用的操作、实时数据（股价、传感器）、随机结果
- 原则：**幂等的只读操作**可以缓存，有副作用的操作不要缓存
:::

## 工具优先级与路由策略

当工具数量较多时，直接把所有工具塞给模型会导致选择准确率下降（进阶篇讨论过）。更成熟的做法是建立一个路由层。

### 基于意图的工具路由

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

interface ToolWithFunc {
  schema: Anthropic.Tool;
  func: (...args: any[]) => any;
}

interface ToolGroup {
  description: string;
  tools: ToolWithFunc[];
}

class ToolRouter {
  /** 工具路由器：根据用户意图选择工具子集 */
  private toolGroups: Map<string, ToolGroup> = new Map();
  private toolFuncs: Map<string, (...args: any[]) => any> = new Map();

  registerGroup(groupName: string, description: string, tools: ToolWithFunc[]): void {
    /** 注册一组工具 */
    this.toolGroups.set(groupName, { description, tools });
    for (const tool of tools) {
      this.toolFuncs.set(tool.schema.name, tool.func);
    }
  }

  async selectTools(query: string): Promise<Anthropic.Tool[]> {
    /** 让 LLM 选择合适的工具组 */
    const groupDescriptions = [...this.toolGroups.entries()]
      .map(([name, info]) => `- ${name}: ${info.description}`)
      .join("\n");

    const response = await client.messages.create({
      model: "claude-sonnet-4-20250514",
      max_tokens: 100,
      messages: [{
        role: "user",
        content: `根据用户请求选择最合适的工具组。

可用工具组：
${groupDescriptions}

用户请求：${query}

只返回工具组名称，多个用逗号分隔。`,
      }],
    });

    const text = response.content[0].type === "text" ? response.content[0].text : "";
    const selected = text.trim().split(",").map((s) => s.trim());

    // 合并选中的工具组
    const schemas: Anthropic.Tool[] = [];
    for (const groupName of selected) {
      const group = this.toolGroups.get(groupName);
      if (group) {
        schemas.push(...group.tools.map((t) => t.schema));
      }
    }
    return schemas;
  }

  getFunc(toolName: string): ((...args: any[]) => any) | undefined {
    return this.toolFuncs.get(toolName);
  }
}
```

### 基于规则的快速路由

对于结构化的请求，规则匹配比 LLM 路由更快更便宜：

```typescript
class RuleBasedRouter {
  /** 基于规则的快速工具路由 */
  private rules: Array<{ keywords: string[]; group: string }> = [];

  addRule(keywords: string[], group: string): void {
    /** 添加关键词匹配规则 */
    this.rules.push({ keywords, group });
  }

  route(query: string): string {
    /** 匹配规则返回工具组名 */
    const queryLower = query.toLowerCase();
    for (const { keywords, group } of this.rules) {
      if (keywords.some((kw) => queryLower.includes(kw))) {
        return group;
      }
    }
    return "general"; // 兜底
  }
}

// 使用
const router = new RuleBasedRouter();
router.addRule(["文件", "目录", "读取", "写入"], "file_tools");
router.addRule(["搜索", "查找", "搜一下"], "search_tools");
router.addRule(["计算", "求解", "算一下"], "math_tools");

const group = router.route("帮我读取 config.json 文件"); // -> "file_tools"
```

::: tip 生产环境推荐混合路由
规则匹配（最快、最确定） -> 语义匹配（快、无需 LLM） -> LLM 分类（最灵活、兜底）。三层逐级降级，兼顾速度和准确率。
:::

## 安全沙箱设计

工具执行涉及外部操作（文件读写、命令执行、网络请求），安全问题不容忽视。

### 路径安全：防止目录穿越

```typescript
import path from "path";

class SafePathResolver {
  /** 安全路径解析器 */
  private workspace: string;

  constructor(workspace: string) {
    this.workspace = path.resolve(workspace);
  }

  resolve(filePath: string): string {
    /** 解析路径，确保不超出工作目录 */
    const fullPath = path.isAbsolute(filePath)
      ? path.resolve(filePath)
      : path.resolve(this.workspace, filePath);

    if (!fullPath.startsWith(this.workspace)) {
      throw new Error(`路径 '${filePath}' 超出工作目录范围`);
    }
    return fullPath;
  }
}

// 使用
const resolver = new SafePathResolver("/tmp/workspace");
resolver.resolve("data.txt");             // OK: /tmp/workspace/data.txt
resolver.resolve("../../../etc/passwd");   // Error!
```

### 执行超时：防止工具挂起

```typescript
function withTimeout<T>(fn: (...args: any[]) => Promise<T>, seconds: number = 30) {
  /** 包装器：限制异步函数执行时间 */
  return async (...args: any[]): Promise<T> => {
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`工具执行超时（${seconds}秒）`));
      }, seconds * 1000);

      fn(...args)
        .then((result) => {
          clearTimeout(timer);
          resolve(result);
        })
        .catch((err) => {
          clearTimeout(timer);
          reject(err);
        });
    });
  };
}

// 使用
async function slowSearch(query: string): Promise<Record<string, any>> {
  /** 可能很慢的搜索操作 */
  await new Promise((resolve) => setTimeout(resolve, 5000)); // 模拟慢操作
  return { results: ["..."] };
}

const safeSearch = withTimeout(slowSearch, 10);
```

### 权限控制：分级管理

```typescript
enum ToolPermission {
  READ = "read",       // 只读操作
  WRITE = "write",     // 写入操作
  EXECUTE = "execute", // 执行代码
  NETWORK = "network", // 网络请求
}

class PermissionManager {
  /** 工具权限管理器 */
  private allowed: Set<ToolPermission>;

  constructor(allowed: Set<ToolPermission>) {
    this.allowed = allowed;
  }

  check(toolName: string, required: ToolPermission): boolean {
    /** 检查工具是否有执行权限 */
    if (!this.allowed.has(required)) {
      throw new Error(
        `工具 '${toolName}' 需要 ${required} 权限，当前未授权`
      );
    }
    return true;
  }
}

// 使用：只允许读取和网络操作，不允许写入和执行
const manager = new PermissionManager(
  new Set([ToolPermission.READ, ToolPermission.NETWORK])
);
manager.check("read_file", ToolPermission.READ);      // OK
manager.check("run_code", ToolPermission.EXECUTE);     // Error!
```

## 综合实践：生产级工具执行器

把上面所有概念整合到一个工具执行器中：

```typescript
interface ExecutionLogEntry {
  tool: string;
  params: Record<string, any>;
  status: string;
  timestamp: number;
}

class ProductionToolExecutor {
  /** 生产级工具执行器：缓存 + 超时 + 权限 + 日志 */
  private cache: ToolCache;
  private timeout: number;
  private executionLog: ExecutionLogEntry[] = [];

  constructor(cacheTtl: number = 300, timeout: number = 30) {
    this.cache = new ToolCache(cacheTtl);
    this.timeout = timeout;
  }

  execute(
    toolName: string, params: Record<string, any>,
    func: (params: Record<string, any>) => any,
    permissionManager?: PermissionManager, cacheable: boolean = true
  ): string {
    /** 执行工具（带完整保护） */
    // 1. 权限检查
    if (permissionManager) {
      // 根据工具类型检查权限（简化示例）
    }

    // 2. 查缓存
    if (cacheable) {
      const cached = this.cache.get(toolName, params);
      if (cached !== null) {
        this.log(toolName, params, "cache_hit");
        return cached;
      }
    }

    // 3. 执行（带超时）
    let resultStr: string;
    try {
      const result = func(params);
      resultStr = JSON.stringify(result);
    } catch (e) {
      if (e instanceof Error && e.message.includes("超时")) {
        resultStr = JSON.stringify({ error: "执行超时" });
      } else {
        resultStr = JSON.stringify({ error: String(e) });
      }
    }

    // 4. 存缓存
    if (cacheable) {
      this.cache.set(toolName, params, resultStr);
    }

    // 5. 记录日志
    this.log(toolName, params, "executed");
    return resultStr;
  }

  private log(toolName: string, params: Record<string, any>, status: string): void {
    this.executionLog.push({
      tool: toolName,
      params,
      status,
      timestamp: Date.now() / 1000,
    });
  }
}
```

## 小结

- **动态工具生成**：根据上下文加载/卸载工具，避免一次性传入过多工具
- **结果缓存**：幂等只读操作用 TTL 缓存避免重复调用，注意不缓存有副作用的操作
- **工具路由**：规则匹配（快） + 语义匹配 + LLM 分类（准），三层降级
- **安全沙箱**：路径限制、执行超时、权限分级是生产环境的必备措施

## 练习

1. **动态工具**：实现一个 Agent，用户说"连接数据库 X"时自动加载 SQL 查询工具，说"断开连接"时卸载。
2. **缓存策略**：为天气查询工具实现一个智能缓存——同城市 10 分钟内用缓存，超过 10 分钟重新查询。
3. **安全测试**：尝试让 Agent 读取 `/etc/passwd`，验证路径安全机制是否生效。
4. **综合挑战**：用本节的 `ProductionToolExecutor` 替换入门篇天气 Agent 的工具执行逻辑，对比有无缓存时的 API 调用次数。

## 参考资源

- [Anthropic: Building Effective Agents](https://docs.anthropic.com/en/docs/build-with-claude/agentic) -- 官方 Agent 构建指南
- [LangChain: Tool Use](https://python.langchain.com/docs/concepts/tools/) -- LangChain 工具使用文档
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/) -- LLM 应用安全风险
- [Anthropic: Prompt Injection](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) -- 提示注入防护
