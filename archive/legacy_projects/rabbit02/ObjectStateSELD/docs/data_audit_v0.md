# Object-State SELD v0 Data Audit

Status: **PASS**

| Split | Samples | Recordings | Tracks | Missing audio |
| --- | --- | --- | --- | --- |
| train | 63569 | 388 | 963 | 0 |
| val | 15024 | 98 | 252 | 0 |
| test | 14765 | 99 | 231 | 0 |

## Track summary

- Continuous segments: 7143
- Segments >= 2.5 s: 2498

## Geometry and leakage

- train: max |norm(p)-1|=7.271e-09; max |p dot v_tan|=9.551e-09; max future norm error=7.271e-09.
- val: max |norm(p)-1|=7.271e-09; max |p dot v_tan|=8.971e-09; max future norm error=7.271e-09.
- test: max |norm(p)-1|=7.271e-09; max |p dot v_tan|=1.248e-08; max future norm error=7.271e-09.
- Recording overlaps: {'test-train': [], 'test-val': [], 'train-val': []}

## Errors

- None
