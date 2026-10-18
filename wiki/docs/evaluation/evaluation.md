---
sources:
  - scripts/run_eval.py
  - app/services/eval_service.py
  - eval/eval_set.json
---
# 评测集与评测方式

## 评测集模板

`eval/eval_set.json` 提供 **20 条**汽车售后场景问题，覆盖：

保养周期、油液规格、易损件、故障码、电气/蓄电池、轮胎胎压、季节性检查、
使用与应急、故障诊断、保修条款与索赔流程，以及 **1 条负样本**（用于检验系统是否拒答）。

### 字段含义

| 字段 | 说明 |
| --- | --- |
| `id` | 唯一编号 |
| `category` | 分类，便于按维度看结果 |
| `question` | 用户问题 |
| `ground_truth` | 标准答案，**需要你按真实资料填写**（`[待补充]`）；只用于人工复核，不参与自动打分 |
| `expected_sources` | 期望命中的来源文件（`data/documents` 下的相对路径），用于"来源命中"统计 |
| `expected_keywords` | 答案中必须出现的字面串（数值、部件名、故障码），用于关键词覆盖率 |
| `must_refuse` | `true` 表示知识库无依据，期望系统拒答 |
| `notes` | 该条想考察什么，便于面试时讲设计意图 |

!!! warning "填齐全 20 条需要你有真实资料"
    这一步**必须由你完成**，项目不替你编造答案。

---

## 怎么跑评测

=== "方式一：命令行脚本（推荐）"

    纯标准库，同时产出 Markdown 报告便于人工复核。

    ```bash
    # 完整问答评测（消耗 token）
    python scripts/run_eval.py

    # 只测召回，不调用大模型，零 token 消耗
    python scripts/run_eval.py --mode retrieve

    # 只跑前 5 条做快速验证
    python scripts/run_eval.py --limit 5
    ```

=== "方式二：HTTP 接口"

    ```bash
    curl -X POST http://127.0.0.1:8000/api/v1/eval/run ^
      -H "Content-Type: application/json" ^
      -d "{}"
    ```

### 产出

- `eval/results/local_report_<时间戳>.json`：逐条问题、召回来源、引用编号、耗时、token 用量；
- `eval/results/local_report_<时间戳>.md`：汇总表 + 逐条回答原文，适合贴进面试材料。

---

## 报告里的数字是什么，不是什么

| 指标 | 计算方式 | 请注意 |
| --- | --- | --- |
| `source_hit_rate` | 期望来源是否出现在召回结果中 | 衡量**召回**，与回答质量无关 |
| `avg_keyword_coverage` | 期望关键词在回答里的覆盖率 | 只是字面覆盖，同义改写会判为未覆盖 |
| `refusal_success` | 负样本是否明确拒答 | 只看是否拒答，不看话术质量 |
| `citation_out_of_range_items` | 回答里出现越界 `[n]` 的条目 | 用于发现幻觉引用 |
| `elapsed_ms` / `latency` | 该次运行的真实墙钟耗时 | 受你本机与网络影响，**不是性能基准** |

!!! danger "这些都不是准确率"
    也没有与任何基线系统对比。
    要做人工评测，请按 `ground_truth` 逐条人工核对，
    并把结论记为"人工评测结果（N 条，评测人：你）"。

---

## 相关页面

- [实测记录（诚实版）](results.md) —— 三轮真实执行结果，含未解决的失败样例
- [排查手册](../ops/troubleshooting.md) —— 检索不准时按什么顺序排查
