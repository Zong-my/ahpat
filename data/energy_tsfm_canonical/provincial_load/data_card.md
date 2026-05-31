# Data Card: provincial_load

- Rows: 140928
- Series count: 2
- Target: `load` mapped to canonical `target`
- Time span: 2020-04-01 00:00:00 -> 2022-04-28 23:45:00
- Split counts: `{"test": 21140, "train": 98648, "validation": 21140}`
- Source paths:
  - `(original source path removed)`
  - `(original source path removed)`

## Notes
- Manuscript-facing text must describe this as an anonymized real provincial grid-load dataset from northern China.
- Concrete province and child identifiers are not written to canonical outputs.
- Weather join uses timestamp intersection, so the current canonical span follows the weather table.
