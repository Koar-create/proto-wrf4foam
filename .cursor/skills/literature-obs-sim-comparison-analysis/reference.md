# literature-obs-sim-comparison-ledger.jsonl — reference

## Source formats

- **PDF only**: `"source": { "source_format": "pdf", "pdf_path": "...", "journal_pages": "..." }`
- **Marker directory / md** (prefer filling both markdown and original pdf when known):

```json
"source": {
  "source_format": "marker_md",
  "pdf_path": "docs/reference-candidate/direction1-rans-cpratio/1-s2.0-S0360132318302671-main.pdf",
  "markdown_path": "docs/reference-candidate/marker_out/1-s2.0-S0360132318302671-main/1-s2.0-S0360132318302671-main.md",
  "marker_out_dir": "docs/reference-candidate/marker_out/1-s2.0-S0360132318302671-main",
  "journal_pages": "Build. Environ. 139 (2018) 146-156"
}
```

## Q1 true + multi_layer (pretty-printed; file stores one line)

```json
{
  "ledger_schema_version": 1,
  "id": "10.1016/j.uclim.2023.101569",
  "title": "Sensitivity analysis of WRF-CFD-based downscaling methods for evaluation of urban pedestrian-level wind",
  "year": 2023,
  "first_author": "Huang",
  "doi": "10.1016/j.uclim.2023.101569",
  "source": {
    "pdf_path": "docs/reference-candidate/Huang-2023-WRF-CFD-Sensitivity analysis of WRF-CFD-based downscaling methods for evaluation of urban pedestrian-level wind.pdf",
    "journal_pages": "Urban Climate (2023) 101569"
  },
  "analyzed_at": "2026-06-02T12:00:00Z",
  "content": {
    "q1_observation_simulation_comparison": {
      "answer": true,
      "evidence": "Sect. 2.1 + 3: long-term tower measurements QC'd and compared with WRF/CFD pedestrian-level wind statistics (R2, spatial means)."
    },
    "q2_observation_spatial_type": {
      "value": "multi_layer",
      "instruments": ["15 m rooftop tower", "pedestrian-height sensors"],
      "details": "On-site mast with multiple reporting heights for mean wind speed/direction over May–Jul 2018.",
      "rationale": "Tower provides vertical structure above roof and links to pedestrian-level network; not a single-height point measurement."
    },
    "q3_meaning_without_observation": null,
    "self_assessment": { "score_0_10": 8.5, "caveats": "Pedestrian sensor layout interpreted from text/captions." },
    "notes": "skipped_q3_due_to_q1_true"
  }
}
```

## Q1 true + wind_tunnel

```json
{
  "ledger_schema_version": 1,
  "id": "10.1007/s00376-013-2234-9",
  "title": "Simulating Urban Flow and Dispersion in Beijing by coupling a CFD model with the WRF model",
  "year": 2013,
  "first_author": "Miao",
  "doi": "10.1007/s00376-013-2234-9",
  "source": { "pdf_path": "docs/reference-candidate/WRF-OpenFoam-MiaoYucong-2013-Simulating Urban Flow and Dispersion in Beijing by coupling a CFD with WRF.pdf" },
  "analyzed_at": "2026-06-02T12:00:00Z",
  "content": {
    "q1_observation_simulation_comparison": {
      "answer": true,
      "evidence": "Sect. 4.1: OpenFOAM vs Hamburg wind-tunnel single-building experiment; Sect. 4.1.2: WRF vs Beijing met stations."
    },
    "q2_observation_spatial_type": {
      "value": "wind_tunnel",
      "instruments": ["Hamburg Univ. wind tunnel (1:200 scale building)"],
      "details": "Primary CFD validation uses laboratory PIV/pressure; WRF also compared to surface stations but coupling case lacks building-scale field validation.",
      "rationale": "Paper's explicit OpenFOAM validation benchmark is wind-tunnel; field obs used mainly for WRF, not street-scale CFD."
    },
    "q3_meaning_without_observation": null,
    "self_assessment": { "score_0_10": 8.0, "caveats": "Mixed obs types; Q2 follows main CFD validation source per SKILL boundary note." },
    "notes": "skipped_q3_due_to_q1_true"
  }
}
```

## Q1 false stub

```json
{
  "ledger_schema_version": 1,
  "id": "10.1016/j.scs.2025.106841",
  "title": "Assessing scale-dependent urban wind loss patterns: A CFD analysis of realistic urban segments from 1980 to 2020",
  "year": 2025,
  "first_author": "Wang",
  "doi": "10.1016/j.scs.2025.106841",
  "source": { "pdf_path": "docs/reference-candidate/2025-Assessing scale-dependent urban wind loss patterns -- A CFD analysis of realistic urban segments from 1980 to 2020.pdf" },
  "analyzed_at": "2026-06-02T12:00:00Z",
  "content": {
    "q1_observation_simulation_comparison": {
      "answer": false,
      "evidence": "Sect. 2–4: prescribed power-law inflow RANS over reconstructed urban morphologies; no measurement comparison reported."
    },
    "q2_observation_spatial_type": null,
    "q3_meaning_without_observation": {
      "summary": "作者以 1980–2020 真实城市片段 CFD 揭示尺度依赖的风损格局，论证长时段形态变化对通风的结构性影响，无需现场验证即可支持规划讨论。",
      "arguments": [
        "Intro: 填补多年代真实城市形态与风环境响应之间证据空白（Sect. 1）。",
        "Methods: 基于 GIS 重建的历史建成区几何保证情景可重复（Sect. 2.1）。",
        "Results: 跨尺度 wind-loss 指标对比不同年代片段，提供政策相关定性结论（Sect. 3–4）。"
      ]
    },
    "self_assessment": { "score_0_10": 8.5, "caveats": "Q3 arguments paraphrased from abstract/intro/conclusions." },
    "notes": "skipped_q2_due_to_q1_false"
  }
}
```
