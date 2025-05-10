# Results

We evaluate DPE against robust baselines across nine benchmarks:

- WATERBIRDS, CELEBA, METASHIFT
- CHEXPERT, IMAGENETBG, LIVING17
- MULTINLI, CIVILCOMMENTS, NICO++

## Worst-Group Accuracy (No Attribute)

| Method       | Avg WGA |
|--------------|---------|
| ERM          | 57.7    |
| DFR          | 65.2    |
| RWY          | 67.5    |
| **DPE**      | **73.9**|

## Worst-Group Accuracy (With Attribute)

| Method       | Avg WGA |
|--------------|---------|
| CRT          | 73.0    |
| DFR*         | 77.1    |
| GAP (All)    | 83.9    |
| **DPE**      | **83.0**|

## Accuracy vs. Ensemble Size

DPE gains saturate around 15 prototypes:

- ∆5 = 2.4%
- ∆10 = 3.3%
- ∆15 = 3.7%
- ∆25 = 3.7%

## Runtime (RTX6000, batch=1)

| Prototypes | Time (s) | Memory (GB) |
|------------|----------|-------------|
| 15         | 0.0031   | 0.2032      |
| 100        | 0.0045   | 0.8517      |


