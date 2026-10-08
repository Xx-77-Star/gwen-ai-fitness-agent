# Gwen AI Fitness Companion

Gwen 是一个面向健身与个人训练场景的 AI Fitness Companion，展示如何把 LLM、LangGraph 工作流、长期记忆、RAG 知识库和工具调用组合成一个可观察、可扩展的 AI Agent 应用。仓库同时提供 FastAPI 服务、静态前端和 Gwen 3D Avatar 体验。

## 核心能力

### LLM Agent

Gwen 将用户输入转换为结构化 Agent 状态，通过提示词编排用户画像、记忆上下文、知识检索结果、工具结果和对话历史，最终生成面向训练、营养与恢复场景的回答。LLM 通过 OpenAI-compatible gateway 接入阿里云百炼 / DashScope。

### LangGraph Workflow

每次对话由 LangGraph 图驱动，按以下阶段执行：

```text
START
  -> load_profile
  -> memory_retrieve
  -> rag_retrieve
  -> intent
  -> tool_decision
       |                         |
       | 需要工具               | 不需要工具
       v                         v
  tool_execution -> tool_decision   response
       |                         |
       +-------------------------+
                  |
                  v
              memory_update
                  |
                  v
        conversation_persistence
                  |
                  v
                 END
```

工具决策节点根据用户意图选择只读训练工具或直接生成回答；工具执行节点支持训练记录、训练推荐和天气等工具，并通过轮次和调用数量限制保证流程可控。

### Long-term Memory

Memory 模块以稳定的 `MemoryServiceInterface` 与 Agent 解耦，结合短期对话上下文和长期 `UserMemory` 记忆，将用户目标、训练偏好和经验等级等持久化信息注入后续对话。Memory Center API 支持查看、更新和重置长期记忆。

### RAG Knowledge Base

`knowledge/` 目录提供训练、营养和恢复主题的 Markdown 知识文档。RAG 层支持文档分块、Embedding、FAISS 向量检索、Top-K 排序和来源标注，将相关知识安全地格式化到 Agent 上下文中。

### Agent Trace

LangGraph Callback 记录每次运行的节点、耗时、工具调用、工具结果、Memory 使用、RAG 命中和停止原因。Trace 既可通过结构化日志输出，也会通过 SSE 事件流驱动前端执行状态，并在 Chat API 响应中返回可视化所需的结构化数据。

## 系统架构

```text
                       +----------------------+
                       |  Frontend / Avatar   |
                       +----------+-----------+
                                  | HTTP / SSE
                                  v
+----------------+       +--------+---------+       +----------------+
| Knowledge Base | ----> | FastAPI / Chat   | ----> | LangGraph Agent|
| Markdown + FAISS|      | API + SSE Trace  |       | Workflow       |
+----------------+       +--------+---------+       +--------+-------+
                                   |                         |
                           +-------+--------+        +-------+--------+
                           | SQLite          |        | LLM Gateway   |
                           | Profiles /      |        | OpenAI-compatible|
                           | Memory / Turns  |        | DashScope      |
                           | Training Records|        +----------------+
                           +-----------------+
```

主要模块：

| 模块 | 路径 | 职责 |
| --- | --- | --- |
| Agent | `app/agent/` | LangGraph 状态、节点、路由和工具调用编排 |
| Memory | `app/memory/` | 用户画像、长期记忆、短期会话和记忆提取 |
| RAG | `app/rag/` | 文档加载、分块、Embedding、FAISS 检索和来源格式化 |
| Tools | `app/tools/` | 训练记录、训练推荐和天气等受控工具 |
| Observability | `app/observability/` | Agent Run Trace、节点事件流和 JSON 日志 |
| API | `app/api/routes/` | Chat、Memory、Profile、Training 和 Health API |
| Frontend | `frontend/` | Gwen Web UI、Agent Trace Drawer 和 3D Avatar |
| Knowledge | `knowledge/` | 训练、营养、恢复知识库文档 |

## 技术栈

- **Agent orchestration**：Python、LangGraph、LangChain Callbacks
- **LLM integration**：OpenAI-compatible SDK、阿里云百炼 / DashScope
- **RAG**：FAISS、NumPy、Embedding API、Markdown knowledge base
- **Backend**：FastAPI、Uvicorn、Pydantic、SQLAlchemy
- **Persistence**：SQLite（开发默认）
- **Frontend**：原生 JavaScript、ES Modules、Three.js、GLB Avatar
- **Quality**：pytest、pytest-asyncio、Ruff

## 运行方式

### 环境要求

- Python 3.11+
- 可访问 OpenAI-compatible LLM endpoint 的 API Key
- Windows PowerShell 或兼容终端

### 安装

```powershell
cd "D:\AI agent2\fitlife-ai"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 配置环境变量

复制模板并填写本地配置：

```powershell
Copy-Item .env.example .env
```

`.env` 只放在本地，真实 API Key、数据库文件和运行日志不会被 Git 提交。请将 `LLM_API_KEY` 替换为自己的密钥。

### 启动服务

```powershell
.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

打开：

- [Web UI](http://127.0.0.1:8000/)
- [Swagger UI](http://127.0.0.1:8000/docs)

启动时应用会校验配置并初始化 SQLite schema。

### 运行测试与检查

```powershell
.venv\Scripts\Activate.ps1
python -m pytest
python -m ruff check app tests
```

默认 pytest 会跳过 `real_llm` 测试，不会调用真实 LLM API。前端 Avatar 和交互测试可用 Node.js 运行对应测试脚本。

## API 示例

### 发送对话

```http
POST /chat
Content-Type: application/json

{
  "user_id": "user-001",
  "message": "今天应该怎么训练？",
  "conversation_id": "conversation-001"
}
```

响应包含 `response`、`conversation_id` 和结构化 `trace`。可订阅执行状态：

```http
GET /chat/{conversation_id}/events?user_id=user-001
Accept: text/event-stream
```

### 管理长期记忆

```http
GET    /memory/{user_id}
PUT    /memory/{user_id}/{memory_key}
DELETE /memory/{user_id}
```

支持的 `memory_key` 由 `app/schemas/memory.py` 中的枚举定义。

## 项目结构

```text
fitlife-ai/
├── app/
│   ├── agent/          # LangGraph workflow、AgentState、工具决策与执行
│   ├── api/routes/     # Chat、Memory、Profile、Training、Health API
│   ├── config/         # 环境变量与应用配置
│   ├── database/       # SQLAlchemy models、session、repositories
│   ├── llm/            # OpenAI-compatible LLM gateway
│   ├── memory/         # Long-term memory、conversation persistence
│   ├── observability/  # Agent trace、streaming events
│   ├── rag/            # RAG loading、chunking、embedding、FAISS
│   ├── schemas/        # Pydantic API schemas
│   └── tools/          # Tool registry、executor、business tools
├── frontend/           # Gwen Web UI、Avatar、Three.js runtime
├── knowledge/          # Markdown RAG knowledge base
├── tests/              # API、Agent、Memory、RAG、Tool、Trace tests
├── .env.example
├── .gitignore
├── ARCHITECTURE.md
├── pyproject.toml
└── README.md
```

## 安全与边界

- `.env`、数据库文件、日志、缓存、Python 临时文件和 IDE 配置均被忽略。
- 不要将真实 API Key 写入代码、README、测试夹具或 Git 提交记录。
- Agent 输出用于一般健身信息与训练协作，不构成医疗诊断或治疗方案；涉及持续疼痛、伤病或疾病时，请咨询医生或合格专业人员。
- 当前本地默认使用 SQLite，生产环境应替换为具备备份、访问控制和监控能力的数据库。

## 未来规划

- 强化 Long-term Memory：记忆置信度、过期策略、遗忘机制和用户可控编辑。
- 扩展 RAG Knowledge Base：增量索引、来源版本管理、检索评估和引用式回答。
- 丰富 Agent Tools：饮食记录、训练计划生成、周期化推荐和更多外部服务。
- 引入 MCP：将外部工具和数据源以统一协议接入 LangGraph。
- 完善 Agent Trace：OpenTelemetry、Trace 导出、运行回放、成本与延迟指标。
- 增加认证授权、多用户数据隔离、数据库迁移、部署配置和可观测性告警。
- 持续优化 Avatar、移动端体验和无脚本回退体验。

## License

如需公开发布，请在仓库根目录补充适合项目的 LICENSE 文件，并确认第三方组件许可证与资源授权。
