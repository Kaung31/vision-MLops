# Promotion decision: PROMOTE
- champion v1  →  challenger v2

| clause | result | detail |
|---|---|---|
| C1_in_distribution | PASS | in-dist mAP50-95 0.630 vs champion 0.601; CI-overlap ok; point >= champion-eps |
| C2_gated_cross_dataset | PASS | gated aggregate 0.151 vs champion 0.137; worst single-suite drop 0.000 on none (ok) |
| C3_large_slice_regression | PASS | 0 large-slice regression(s); 0 small-slice warning(s) |
| C4_corruption | PASS | corruption AUC 0.584 vs champion 0.533 (tol 0.02) |
| C5_slo | PASS | STUBBED — SLO not yet measured — placeholder pass until Phase 6 serving stack exists |
