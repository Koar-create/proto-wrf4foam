# literature-trans-scale-ledger.jsonl — reference

## One-line record example (pretty-printed here for reading only; file stores a single line)

```json
{
  "ledger_schema_version": 1,
  "id": "10.5194/gmd-14-2503-2021",
  "title": "WRF4PALM v1.0: a mesoscale dynamical driver for the microscale PALM model system 6.0",
  "year": 2021,
  "first_author": "Lin",
  "doi": "10.5194/gmd-14-2503-2021",
  "source": {
    "pdf_path": "docs/reference-candidate/2021-WRF4PALMv1.0--A-mesoscale-Dynamical-Driver-for-the-Microscale-PALM-Model-System-6.0.pdf",
    "journal_pages": "GMD 14, 2503–2524"
  },
  "analyzed_at": "2026-05-11T00:00:00Z",
  "content": {
    "q1_meso_micro_coupling": { "answer": true, "evidence": "See abstract and Sect. 2–3." },
    "q1a_main_method": {
      "value": "LES",
      "mixed_specify": null,
      "rationale": "Case-study PALM configuration uses LES with STG; RANS not used in results."
    },
    "q1b_unique_contribution": "…",
    "q1c_acknowledges_limitations": { "answer": true, "points": ["…"] },
    "self_assessment": { "score_0_10": 8.5, "caveats": "…" },
    "notes": "optional"
  }
}
```

## Q1 false stub

```json
{
  "ledger_schema_version": 1,
  "id": "no-doi-slugified-title#2020",
  "title": "…",
  "year": 2020,
  "first_author": "…",
  "doi": null,
  "source": { "pdf_path": "docs/reference-candidate/example.pdf" },
  "analyzed_at": "2026-05-11T00:00:00Z",
  "content": {
    "q1_meso_micro_coupling": { "answer": false, "evidence": "…" },
    "q1a_main_method": null,
    "q1b_unique_contribution": null,
    "q1c_acknowledges_limitations": null,
    "self_assessment": { "score_0_10": 7.0, "caveats": "…" },
    "notes": "skipped_1a_1b_1c_due_to_q1_false"
  }
}
```
