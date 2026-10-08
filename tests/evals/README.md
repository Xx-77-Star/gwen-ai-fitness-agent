# Agent Evaluation

Phase E-3 的本地评估目录，包含 Tool Selection Evaluation 和 Response Quality Evaluation。
所有评估均使用 Fake LLM，不构造真实 LLM Client，也不访问网络或百炼 API。

## 运行

```powershell
pytest -q -W error
```

完整数据集评估会输出：

```text
Tool Selection Accuracy: 100.00% (21/21)
Response Quality Score: 100.00% (14/14 pass)
```

## Tool Selection 数据集

`datasets/tool_selection.jsonl` 每行包含：

- `id`：评估样本 ID。
- `question`：用户自然语言问题。
- `expected_tool`：期望选择的训练工具。
- `arguments`：该工具调用使用的固定参数。

数据集覆盖 `get_training_summary`、`get_last_training_record` 和 `list_training_records`。
`FakeToolSelectionLLM` 通过真实的 `tool_decision_node` 返回确定性工具调用。

## Response Quality 数据集

`datasets/response_quality.jsonl` 覆盖三类回答场景：

- `direct_generated`：Agent 直接生成回答。
- `direct_draft`：Agent 复用 Tool Decision 的回答草稿。
- `tool_result`：Agent 根据实际工具结果生成最终回答。

每条数据包含候选回答和可复现的质量量规：

- `required_terms`：必须包含的关键信息。
- `required_any_groups`：每组至少命中一个表达。
- `forbidden_terms`：禁止编造或不安全表达。
- `allowed_numbers`：允许出现在回答中的数字。
- `safety_required`：是否必须包含就医或专业人员提醒。
- `must_recommend_action`：是否必须提供具体行动建议。

`FakeResponseEvaluationLLM` 扩展 Tool Selection Fake LLM，通过真实的 `response_node` 生成最终回答。
最终质量分按以下维度加权：

- Instruction Following：30%
- Groundedness：25%
- Usefulness：20%
- Safety：15%
- Clarity：10%

单条样本达到 90% 计为通过。## Real Model Evaluation

`real_llm/` 包含真实百炼 qwen-plus 评估，默认由 `addopts` 中的 `not real_llm` 跳过。
只有显式执行以下命令才会调用真实 API：

```powershell
pytest -q -W error -m real_llm
```

真实评估覆盖：

- Tool Selection Accuracy 与稳定返回 `tool_calls`。
- 参数格式、重复调用和工具误选统计。
- Chest 60分钟、Leg 70分钟、Back 50分钟固定数据上的 Response Groundedness。

结果保存到 `tests/evals/results/`，仅记录模型名、行为指标、脱敏结果和固定训练事实，不记录 API Key。