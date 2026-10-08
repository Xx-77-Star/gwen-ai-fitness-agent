# Gwen AI Fitness Companion 架构说明

本文档补充 `README.md` 中的发布级架构概览，聚焦 Agent 编排边界、模块职责和扩展点。代码中的 Agent、Memory、RAG 和工具逻辑保持不变。

## 运行时分层

```text
Web UI / Avatar
      |
      v
FastAPI routes
      |
      +--> LangGraph workflow --> LLM Gateway
      |          |
      |          +--> Memory Service
      |          +--> RAG Knowledge Service
      |          +--> Tool Registry / Executor
      |
      +--> SQLite repositories
      +--> Agent Trace / SSE events
```

## LangGraph 编排

请求级 `AgentState` 在节点之间传递。默认工作流：

```text
load_profile -> memory_retrieve -> rag_retrieve -> intent -> tool_decision
tool_decision --no tools--> response
tool_decision --tool_calls--> tool_execution -> tool_decision
response -> memory_update -> conversation_persistence -> END
```

工具流程受最大轮次和最大调用次数约束。工具决策只接受 Tool Registry 注册且通过 Pydantic 校验的参数；`user_id` 由运行时注入，不能由模型伪造。

## Memory

`MemoryServiceInterface` 是 Agent 与记忆实现之间的稳定契约：

```text
retrieve(user_id, short_term) -> MemoryContext
remember(user_id, memory_key, memory_value) -> UserMemory
```

当前实现将短期对话上下文与长期 `UserMemory` 合并为 `MemoryContext`。会话历史由 `ConversationRepository` 持久化，长期记忆支持查看、更新和重置 API。

## RAG Knowledge Base

`knowledge/` 中的 Markdown 文档经过加载和分块后进入向量检索。`KnowledgeRetrievalService` 负责查询 Embedding、Top-K 检索和带来源的上下文格式化；`VectorStore` 使用 FAISS 保存索引和 chunk metadata。RAG 结果以 `source`、`content`、`score` 三元组进入 `AgentState.rag_context`。

## Agent Trace

`AgentTraceCallback` 观察 LangGraph chain/node 生命周期，生成：

- `AgentRunTrace`：运行 ID、输入摘要、状态、耗时和停止原因。
- `AgentNodeTrace`：节点名称、开始/结束时间、耗时、工具摘要和错误。
- 可注入的 `AgentTraceSink`：内存、结构化日志或 JSON Lines。
- `AgentNodeStreamingSink`：将 start/end 事件发布到前端 SSE 状态流。

Trace 是可观测性层，不修改业务节点和状态语义。

## 扩展边界

- **Memory**：记忆置信度、生命周期、遗忘策略和用户授权。
- **RAG**：增量索引、引用来源、检索评估和更多文档格式。
- **Tools**：训练计划、饮食记录、周期化推荐和外部服务。
- **MCP**：将外部工具以标准协议接入 Agent。
- **Observability**：OpenTelemetry、成本/延迟指标和运行回放。
- **Production**：认证、授权、数据库迁移、备份和部署配置。

## 安全边界

- API Key 只存在于本地 `.env`；发布包只包含 `.env.example`。
- SQLite、日志、缓存和临时文件由 `.gitignore` 排除。
- Trace 默认只暴露工作流阶段、计数和安全摘要，不直接暴露模型提示词或秘密信息。
- 健身建议不是医疗建议；涉及伤病或疾病时应咨询专业人员。
