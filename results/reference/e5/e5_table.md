| Arm | GoodWiki nDCG@10 | LoCo macro | DAPFAM nDCG@100 | MTEB |
|---|---:|---:|---:|---|
| ThreeWayCosine distractors bs18 (published recipe) | 66.99 | 68.26 | 31.60 | ArguAna 50.94, FiQA2018 33.62 |
| ThreeWayCosine distractors bs48 | 65.42 | 66.77 | 31.11 | ArguAna 50.46, FiQA2018 33.47 |
| InfoNCE t=.07 a=0.5 distractors bs18 | 60.16 | 63.15 | 29.48 | ArguAna 48.55, FiQA2018 32.06 |
| InfoNCE t=.07 a=0.5 distractors bs48 | 64.52 | 65.25 | 30.46 | ArguAna 49.96, FiQA2018 33.31 |
| InfoNCE t=.07 a=0 in-batch only bs48 | 65.91 | 65.94 | 30.71 | ArguAna 49.74, FiQA2018 32.55 |
| InfoNCE t=.07 a=0.5 distractors bs48, warm-start | 65.74 | 65.88 | 30.87 | ArguAna 48.37, FiQA2018 31.72 |
| InfoNCE t=.10 a=0.5 distractors bs48 | 64.25 | 64.98 | 30.19 | -- |
| InfoNCE t=.10 a=0 in-batch only bs48 | 64.76 | 64.95 | 30.64 | -- |
| InfoNCE t=.07 a=0 in-batch only bs48, 50 epochs | 66.37 | 66.20 | 30.49 | -- |
