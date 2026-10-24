# LES 2000 s analysis（`LES/analysis/2000s`）

Based on `LES/docs/inbox/LES_2000s_ANALYSIS_PLAN_v1.md`.

```
python run_p0.py          # Probe stitching, route backfill, LiDAR, S1 roof, sampling matching
python tests/test_p0.py   # Route 2 缺口、高度对齐、方差可加性
python run_p1.py          # M1 sampleability report, F01, M2 leading edge (F07)
python run_p2.py          # M3 layer validation (F02/F03/F08), F05, virtual DBS at site
python run_p3.py          # M5 transient Λ (requires cached slices)
```

Cache written to `cache/`, tables and reports written to `out/`, figures written to `results/les/2000s/`.

B 级量一律带置信区间；禁止把 1400–2000 s 窗口统计写成“平均场”或“已收敛 TI”。
