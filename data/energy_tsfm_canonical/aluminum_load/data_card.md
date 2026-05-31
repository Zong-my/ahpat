# Data Card: aluminum_load

- Rows: 62620
- Series count: 5
- Target: `P_AL_rt` mapped to canonical `target`
- Time span: 2024-09-21 18:15:00 -> 2025-02-21 11:30:00
- Split counts: `{"test": 9395, "train": 43832, "validation": 9393}`
- Source paths:
  - `aluminum_load_line_N.csv (industrial plant records)`
  - `aluminum_load_line_N.csv (industrial plant records)`
  - `aluminum_load_line_N.csv (industrial plant records)`
  - `aluminum_load_line_N.csv (industrial plant records)`
  - `aluminum_load_line_N.csv (industrial plant records)`

## Notes
- Five sibling electrolytic-aluminum CSV files are retained as separate series rather than summed, avoiding an unsupported plant-wide aggregation claim.
- Native samples are approximately one minute; canonical series are resampled to 15 minutes by mean.
- Array-valued fields are excluded from first-pass canonical covariates.
