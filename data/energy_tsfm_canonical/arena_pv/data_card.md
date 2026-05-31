# Data Card: arena_pv

- Rows: 20777
- Series count: 1
- Target: `pvexport_data_power_real` mapped to canonical `target`
- Time span: 2020-05-01 00:00:00 -> 2020-12-07 23:45:00
- Split counts: `{"test": 3117, "train": 14543, "validation": 3117}`
- Source paths:
  - `(original source path removed)`
  - `(original source path removed)`

## Notes
- First-pass site is GANNSF1 because validation showed the lowest target missingness among inspected sites.
- CSV.GZ is used instead of parquet because it preserves explicit `datetime`.
- Native 1-minute data are resampled to 15 minutes by mean.
