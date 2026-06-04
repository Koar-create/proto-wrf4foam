---
name: literature-obs-sim-comparison-analysis
description: >-
  Reads scientific articles (PDF, Marker markdown output, or other readable text)
  and records structured answers about whether the study compares simulations with
  observations, the spatial type of observation data (single-layer stations, multi-layer
  lidar/towers, or wind-tunnel scale models), or—when no observation comparison exists—
  the authors' rationale for the study's significance. After analysis, appends one JSON
  line to docs/reference-candidate/literature-obs-sim-comparison-ledger.jsonl. Use when
  the user references this SKILL.md with a PDF path, a marker_out directory, a .md path,
  or asks for observation–simulation comparison / validation literature extraction.
disable-model-invocation: true
---

# 文献：观测–模拟对比分析

当用户使用 **「根据 `@.../SKILL.md` 分析 `@.../xxx.pdf`」**、指向 **marker 输出目录** 或 **`.md` 文件**，或等价表述时：

1. **先完整阅读本 `SKILL.md`**（含下方判据与落盘格式）。
2. **再阅读目标文章**（见「文章来源」；图/表以图题、表题与正文互证，必要时读 marker 目录内图片）。
3. **在聊天中给出简明结论**（便于人类快速浏览）。
4. **必须**在本回合内将 **恰好一行** JSON 追加到台账文件（见「强制落盘」）。不得只做对话分析而不写文件。

**忽略路径**：不要在任何 `**/archive/**` 目录内存放源文件或修改台账（与用户仓库约定一致）。

---

## 文章来源（阅读优先级）

用户 `@` 的路径决定阅读方式；**优先使用用户明确指定的路径**，其次按下列规则自动选择：

| 用户给定 | 阅读方式 |
|----------|----------|
| **PDF 文件**（`.pdf`） | 优先 PDF 可读文本层；若扫描版/乱码严重，且存在对应 marker 输出，则改读 marker Markdown（见下行）。 |
| **marker 输出目录**（如 `docs/reference-candidate/marker_out/<stem>/`） | 读目录内 **主 Markdown**：`<stem>/<stem>.md`（`<stem>` 为目录名）；可选读 `<stem>_meta.json`（目录/节号）；图以 `.md` 内 `![](_page_*.jpeg)` 相对引用为准，验证/对比图必要时 **Read 图片**。 |
| **Markdown 文件**（`.md`） | 直接读该文件；若为 marker 产物，同目录图片按相对路径读取。 |

**marker 目录约定**（本仓库 `scripts/marker.ps1`）：PDF 经 marker 转换后输出至 `docs/reference-candidate/marker_out/<pdf-stem>/`，含 `<pdf-stem>.md`、`<pdf-stem>_meta.json` 及 `_page_*` 图片。

**台账 `source` 字段**：实际阅读路径必须落盘——读 PDF 时填 `pdf_path`；读 marker 时填 `markdown_path`（及可选 `marker_out_dir`），并 **尽量** 同时填写对应原始 `pdf_path`（若已知）；仅 `.md` 且无 PDF 时 `pdf_path` 可省略，`source_format` 标明 `"marker_md"` 或 `"markdown"`。

---

## 台账路径（唯一）

- 追加目标：仓库根下 [`docs/reference-candidate/literature-obs-sim-comparison-ledger.jsonl`](docs/reference-candidate/literature-obs-sim-comparison-ledger.jsonl)。

每行 **一个完整 JSON 对象**，UTF-8，**无**尾逗号，**无**多行 prettify（整对象占一行，便于 `>>` 式追加与版本控制）。

---

## 问题树（必须逐项作答）

### Q1. 本研究是否有「观测–模拟对比」？

- **判定标准**：论文是否将 **现场/实验室观测数据** 与 **本研究的数值模拟（或模型链）输出** 进行 **定量或定性** 对比（含散点图、时间序列、统计指标、空间分布对比等）。以下情形 **不算** Q1=true：
  - 仅用 **其他文献/第三方模型/再分析** 作参考，未与本研究模拟直接对比；
  - 仅做 **模型间互比**（如 RANS vs LES、不同方案 sensitivity），无独立观测；
  - 仅用风洞/理想化实验 **验证 CFD 设置**，但 **案例/应用部分** 不再与任何观测对比 → Q1 仍可为 true（风洞验证本身即观测–模拟对比），但 Q2 应记为 `wind_tunnel`（见下）。
- 输出到 `content.q1_observation_simulation_comparison`：`answer`（boolean）与 `evidence`（短句：摘要/验证节/表图定位，可英文）。

### Q2.（仅当 Q1 为 true）观测数据的空间类型

- 选项（枚举，与选项同名字符串）：
  - `single_layer` — **单层/近单高度** 观测，例如地面或屋顶 **气象站**、单个高度 sonic、单点 mast 某一高度层；空间上多为 **点位** 而非垂直廓线。
  - `multi_layer` — **多层/廓线** 观测，例如 **激光雷达**、多高度塔、无线电探空、无人机垂直剖面、风廓线雷达等；同一位置或路径上 **≥2 个显著不同高度** 的同步或序列测量。
  - `wind_tunnel` — **风洞（缩尺模型）** 或可控实验室流场数据，作为观测基准与 CFD/模型对比。
- **若同一论文含多种类型**：选 **本研究主要验证/结论所依赖** 的那一类；在 `details` 中列出次要类型。
- 输出到 `content.q2_observation_spatial_type`：
  - `value`：上述枚举之一。
  - `instruments`：字符串数组，如 `["10 m mast", "rooftop AWS"]` 或 `["WindCube lidar", "925 m tower"]`。
  - `details`：一两句说明站点/实验设置（可英文）。
  - `rationale`：为何归入该类型（可英文）。

若 Q1 为 false：将 `q2_observation_spatial_type` 设为 `null`，并在 `content.notes` 标明 `skipped_q2_due_to_q1_false`。

### Q3.（仅当 Q1 为 false）缺乏与观测的对比，作者通过什么逻辑论证本研究很有意义？

- 从 **Introduction / Motivation / Discussion / Conclusions** 提取作者 **未依赖现场验证** 时仍主张贡献的逻辑链，例如：
  - 方法/工具链 **首次提出** 或 **降低门槛**（开源、自动化、算力可承受）；
  - **模型间对比**、敏感性分析、理想化/解析解对照；
  - **风洞或 LES 高保真基准** 间接支撑（注意：若全文仅有风洞验证而无其他观测，Q1 应为 true、走 Q2）；
  - 填补 **研究空白**、服务 **工程/政策/设计** 需求；
  - 与 **再分析/NWP/已有文献结果** 的间接一致性；
  - **物理机制** 解释、参数化改进、可重复算例等。
- 输出到 `content.q3_meaning_without_observation`：
  - `summary`：一段连贯中文（或英文）概括；
  - `arguments`：字符串数组，每条一句独立论点，尽量 **引用论文章节**。

若 Q1 为 true：将 `q3_meaning_without_observation` 设为 `null`，并在 `content.notes` 标明 `skipped_q3_due_to_q1_true`（或与 Q2 skipped 合并写一句即可）。

### 自我评分

- 输出到 `content.self_assessment`：`score_0_10`（0–10 数字）、`caveats`（本分析的不确定性，如扫描版 PDF、验证节与补充材料分离等）。

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
| `source` | object | 见下表；至少一项可读路径 |
| `analyzed_at` | string | ISO-8601 UTC，如 `2026-06-02T12:34:56Z` |
| `content` | object | 见上；可额外含 `notes`（string，可选） |

`source` 建议字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `source_format` | string | `"pdf"` \| `"marker_md"` \| `"markdown"` — 实际主要阅读格式 |
| `pdf_path` | string \| null | 原始 PDF（相对仓库根）；无则 `null` |
| `markdown_path` | string \| null | marker 或独立 Markdown 路径；仅读 PDF 时可省略或 `null` |
| `marker_out_dir` | string \| null | marker 输出目录；非 marker 源可省略 |
| `journal_pages` | string | 可选，刊面信息 |

示例（marker 源）：

```json
"source": {
  "source_format": "marker_md",
  "pdf_path": "docs/reference-candidate/direction1-rans-cpratio/1-s2.0-S0360132318302671-main.pdf",
  "markdown_path": "docs/reference-candidate/marker_out/1-s2.0-S0360132318302671-main/1-s2.0-S0360132318302671-main.md",
  "marker_out_dir": "docs/reference-candidate/marker_out/1-s2.0-S0360132318302671-main",
  "journal_pages": "Build. Environ. 139 (2018) 146-156"
}
```

`content` 内建议结构（Q1=true 示例）：

```json
{
  "q1_observation_simulation_comparison": { "answer": true, "evidence": "..." },
  "q2_observation_spatial_type": {
    "value": "multi_layer",
    "instruments": ["WindCube lidar"],
    "details": "...",
    "rationale": "..."
  },
  "q3_meaning_without_observation": null,
  "self_assessment": { "score_0_10": 8.0, "caveats": "..." },
  "notes": "skipped_q3_due_to_q1_true"
}
```

Q1=false 时：`q2_observation_spatial_type` 为 `null`，`q3_meaning_without_observation` 填 `{ "summary": "...", "arguments": ["..."] }`。

更完整的示例见同目录 [`reference.md`](reference.md)。

---

## 强制落盘（追加 JSONL）

1. 构造 **单行** JSON：对上述字段赋值；字符串内换行与引号必须按 JSON 转义。
2. 打开 [`docs/reference-candidate/literature-obs-sim-comparison-ledger.jsonl`](docs/reference-candidate/literature-obs-sim-comparison-ledger.jsonl)，在文件 **末尾** 追加 **一行**（文件末尾原有最后一行则先换行再写新对象行）。
3. **重复记录冲突**：追加前先检查 ledger 是否已有相同 DOI（或用户指定的唯一键）。若已存在：**保留已有记录，放弃当前落盘任务**（在回复中说明冲突并摘要已有行）。仅当用户明确要求「替换旧记录」时，才整文件编辑删除对应旧行后写入新行。
4. 回合结束前 **自检**：确认该文件已保存且新增行可被 `Read` 工具读出。

可选：若用户明确要求维护人类可读的 Markdown 副本，再维护 `docs/reference-candidate/literature-obs-sim-comparison-ledger.md`；**默认不创建/不更新**该 md。

---

## 调用示例

```text
根据 @.cursor/skills/literature-obs-sim-comparison-analysis/SKILL.md 分析 @docs/reference-candidate/某论文.pdf
```

```text
根据 @.cursor/skills/literature-obs-sim-comparison-analysis/SKILL.md 分析 @docs/reference-candidate/marker_out/1-s2.0-S0360132318302671-main
```

```text
根据 @.cursor/skills/literature-obs-sim-comparison-analysis/SKILL.md 分析 @docs/reference-candidate/marker_out/某论文/某论文.md
```

可与 trans-scale skill **独立或串联** 使用：同一文献可分别追加到 `literature-trans-scale-ledger.jsonl` 与本台账（两 skill 均支持 PDF 与 marker 目录）。

---

## 分析流程提示

- 先定位：**Validation / Model evaluation / Comparison with measurements / Wind tunnel experiment** 等节；摘要与结论中的 *validated against*、*compared with observations* 等表述。
- **Q1 边界**：WRF 与站点对比但 **未** 与 CFD 对比 → 若论文主旨包含对 **整条模型链或微尺度结果** 的观测验证，仍可为 true；若观测仅用于驱动 WRF、未与任何模拟输出对比，则为 false。
- **Q2 边界**：屋顶单点风速计 → `single_layer`；塔式多层 sonic → `multi_layer`；Ju2003 风洞 PIV → `wind_tunnel`。城市 CFD 论文常见 **风洞先验验证 + 现场个例**：Q2 取 **现场个例** 类型；`details` 注明风洞用于 setup validation。
- **Q3**：避免把「未来将与观测对比」当作已有意义；应记录作者 **当下** 给出的论证逻辑。
- **marker 源**：表格在 `.md` 中可能排版失真，以正文叙述与图题为主；`_meta.json` 的 `table_of_contents` 可辅助定位 Validation 节。
