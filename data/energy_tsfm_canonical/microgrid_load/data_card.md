# Data Card: microgrid_load

- Rows: 105120
- Series count: 1
- Target: `IUT_load.Total` mapped to canonical `target`
- Time span: 2021-01-01 00:00:00+04:00 -> 2022-12-31 23:50:00+04:00
- Split counts: `{"test": 15768, "train": 73584, "validation": 15768}`
- Source paths:
  - `(original source path removed)`

## Notes
- First-pass target is `Total` from `IUT_load.txt`.
- PV and meteorological files are joined by timestamp and retained as covariates.
- Native 10-minute resolution is preserved.
