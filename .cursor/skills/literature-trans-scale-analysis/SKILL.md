---
name: literature-trans-scale-analysis
description: >-
  Reads PDF scientific articles and records structured answers about mesoscale-to-microscale
  offline nesting or coupling (e.g. WRF/COSMO to PALM/OpenFOAM/Fluent), urban micrometeorology
  turbulence modelling emphasis (LES/RANS/URANS/mixed), stated contributions, and acknowledged
  limitations. After analysis, appends one JSON line to docs/reference-candidate/literature-trans-scale-ledger.jsonl.
  Use when the user references this SKILL.md with a PDF path, or asks for trans-scale / offline nesting
  literature extraction into the project ledger.
disable-model-invocation: true
---

# 文献：中尺度–微尺度跨尺度（coupling / offline nesting）分析

当用户使用 **「根据 `@.../SKILL.md` 分析 `@.../xxx.pdf`」** 或等价表述时：

1. **先完整阅读本 `SKILL.md`**（含下方判据与落盘格式）。
2. **再阅读目标 PDF**（优先可读文本层；图/表以图题、表题与正文互证）。
3. **在聊天中给出简明结论**（便于人类快速浏览）。
4. **必须**在本回合内将 **恰好一行** JSON 追加到台账文件（见「强制落盘」）。不得只做对话分析而不写文件。

**忽略路径**：不要在任何 `**/archive/**` 目录内存放 PDF 或修改台账（与用户仓库约定一致）。

---

## 台账路径（唯一）

- 追加目标：仓库根下 [`docs/reference-candidate/literature-trans-scale-ledger.jsonl`](docs/reference-candidate/literature-trans-scale-ledger.jsonl)。

每行 **一个完整 JSON 对象**，UTF-8，**无**尾逗号，**无**多行 prettify（整对象占一行，便于 `>>` 式追加与版本控制）。

---

## 问题树（必须逐项作答）

### 1. 是否研究了「中尺度（如 WRF / COSMO / 其他）– 微尺度（如 PALM / OpenFOAM / Ansys Fluent / STAR-CCM+ / 其他）」的 **coupling** 和/或 **offline nesting**？

- 输出到 `content.q1_meso_micro_coupling`：`answer`（boolean）与 `evidence`（短句：摘要/方法/节号定位，可英文）。

### 1.A.（仅当 Q1 为 true）城市微气象 **主要方法**

- 选项：`LES` | `RANS` | `URANS` | `mixed`。
- **定义（重要）**：以 **Results / 案例 / 数值实验配置** 中实际运行的微尺度模式为准。**Introduction 仅提及**某方法、但结果部分未使用该方法 → **不得**将该方法记为主要方法。
- 输出到 `content.q1a_main_method`：
  - `value`：上述枚举之一（与选项同名的字符串）。
  - `mixed_specify`：仅当 `value` 为 `mixed` 时必填，例如 `"LES+RANS"`。
  - `rationale`：一两句说明为何判定（可英文）。

若 Q1 为 false：将 `q1a_main_method` 设为 `null`，并在 `content.notes` 或单独字段标明 `skipped_due_to_q1_false`（见 schema）。

### 1.B.（仅当 Q1 为 true）对 trans-scale coupling / offline nesting 的 **独特贡献**

- 输出到 `content.q1b_unique_contribution`：一段连贯文字（中英皆可，建议中文便于检索）。
- 若 Q1 为 false：该字段为 `null` 或字面 `"skipped"`。

### 1.C.（仅当 Q1 为 true）是否 **承认** 该方法在若干方面 **未能改进 / 仍有限制**

- 输出到 `content.q1c_acknowledges_limitations`：`answer`（boolean）与 `points`（字符串数组，每条一句）。
- 若 Q1 为 false：`answer` 可为 `false` 且 `points` 为 `[]`，或整体 `null` 并注明 skipped。

### 自我评分

- 输出到 `content.self_assessment`：`score_0_10`（0–10 数字）、`caveats`（本分析的不确定性，如扫描版 PDF、未重读图坐标等）。

---

## 顶层记录 schema（每行 JSON 对象）

| 字段 | 类型 | 说明 |
|------|------|------|
| `ledger_schema_version` | number | 固定 `1`，便于日后迁移 |
| `id` | string | **推荐**：有 DOI 时用 **小写** `10.xxxx/...`；无 DOI 时用 `slugified_title` + `#` + `year` |
| `title` | string | 文章题目 |
| `year` | number | 出版年 |
| `first_author` | string | 第一作者 **姓氏**（拉丁字母论文用姓氏；中文论文用第一作者中文名或拼音，全文一致即可） |
| `doi` | string \| null | DOI；无则 `null` |
| `source` | object | `pdf_path`（相对仓库根）、可选 `journal_pages` |
| `analyzed_at` | string | ISO-8601 UTC，如 `2026-05-11T12:34:56Z` |
| `content` | object | 见上；可额外含 `notes`（string，可选） |

`content` 内建议结构：

```json
{
  "q1_meso_micro_coupling": { "answer": true, "evidence": "..." },
  "q1a_main_method": { "value": "LES", "mixed_specify": null, "rationale": "..." },
  "q1b_unique_contribution": "...",
  "q1c_acknowledges_limitations": { "answer": true, "points": ["...", "..."] },
  "self_assessment": { "score_0_10": 8.0, "caveats": "..." },
  "notes": "optional"
}
```

当 `q1_meso_micro_coupling.answer` 为 `false` 时：`q1a_main_method`、`q1b_unique_contribution`、`q1c_acknowledges_limitations` 设为 `null`，并在 `content.notes` 写 `"skipped_1a_1b_1c_due_to_q1_false": true`（或等价短句）。

更完整的示例见同目录 [`reference.md`](reference.md)。

---

## 强制落盘（追加 JSONL）

1. 构造 **单行** JSON：对上述字段赋值；字符串内换行与引号必须按 JSON 转义。
2. 打开 [`docs/reference-candidate/literature-trans-scale-ledger.jsonl`](docs/reference-candidate/literature-trans-scale-ledger.jsonl)，在文件 **末尾** 追加 **一行**（文件末尾原有最后一行则先换行再写新对象行）。
3. **重复记录冲突**：追加前先检查 ledger 是否已有相同 DOI（或用户指定的唯一键）。若已存在：**保留已有记录，放弃当前落盘任务**（在回复中说明冲突并摘要已有行）。仅当用户明确要求「替换旧记录」时，才整文件编辑删除对应旧行后写入新行。
4. 回合结束前 **自检**：确认该文件已保存且新增行可被 `Read` 工具读出。

可选：若用户明确要求维护人类可读的 Markdown 副本，再维护 `docs/reference-candidate/literature-trans-scale-ledger.md`；**默认不创建/不更新**该 md。

---

## 调用示例

```text
根据 @.cursor/skills/literature-trans-scale-analysis/SKILL.md 分析 @docs/reference-candidate/某论文.pdf
```

---

## 分析流程提示

- 先定位：标题、摘要、方法、案例/结果、讨论与结论。
- 对「是否耦合/嵌套」：区分 **双向在线耦合** 与 **单向离线嵌套**；二者任一且以中尺度驱动微尺度为主题即 Q1 可为 true。
- 对图/表：优先提取 **caption** 与正文交叉引用；数值结论避免超出文本所给。
