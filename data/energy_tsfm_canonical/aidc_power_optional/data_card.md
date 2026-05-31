# Data Card: aidc_power_optional

- Rows: 10013
- Series count: 1
- Target: `gpu_power_sum_W_mean` mapped to canonical `target`
- Time span: 2023-05-16 09:30:00+00:00 -> 2023-08-31 00:00:00+00:00
- Split counts: `{"test": 1502, "train": 7009, "validation": 1502}`
- Source paths:
  - `(original source path removed)`

## Notes
- AIDC remains optional and should be retained only if it supports H1/H2/H3.
- This canonical source uses aggregate GPU power, not token demand or job demand as measured energy.
- Collapsed 518 duplicate rows by mean over identical domain_id/series_id/timestamp keys.
