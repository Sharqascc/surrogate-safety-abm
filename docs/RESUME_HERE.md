# Resume Point - 2026-09-15

## Done

- ABM framework: 32 modules, ~230 tests, 99% coverage
- Multi-city simulation: Vadodara / Surat / Ahmedabad (Table 1)
- Video pipeline: UVH-26 YOLOv11-S + ByteTrack + homography
- 116 clean trajectories from Sama_video.mp4 (Vaidya intersection, 62 s)
- Calibrated evasive model: Sim120-Cal = 0.65x episodes, 1.12x TTC
- Paper draft: docs/paper_draft.md
- Cover letter: docs/cover_letter.md

## Tomorrow's Priorities

1. Add accuracy metrics to Table 2 (RMSE, MAE, Pearson r, KS test, 95% CI)
2. Reframe title: drop 'multi-city study' claim
3. Qualify Surat/Ahmedabad as simulation-only (no video validation)
4. Expand limitations section (n=1 video, 62 s duration)

## Key Numbers

| Metric | Observed | Sim120-Cal | Ratio |
|---|---|---|---|
| Episodes | 96 | 62 | 0.65x |
| TTC mean (s) | 1.462 | 1.642 | 1.12x |
| DRAC median | 1.846 | 1.397 | 0.76x |
| Serious % | 75.0% | 41.9% | - |

## Session Start Commands

    %cd /content/drive/MyDrive/surrogate-safety-abm
    !pip install -q -e '.[dev]'
    import sys
    for m in list(sys.modules):
        if m.startswith('surrogate_safety_abm'):
            del sys.modules[m]
