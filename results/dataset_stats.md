# SPC Dataset Statistics

## Effective Loader Counts (Duplicate-Label Images Excluded)

Matches `cft_engine/utils/datasets.py` (`LoadImagesAndLabels.cache_labels`), which drops images containing exact duplicate label rows.

| Split / Subset | Images | Total Instances | People | Car | Bus | Motorcycle | Lamp | Truck |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train_all | 7148 | 54176 | 17297 | 33566 | 525 | 345 | 1714 | 729 |
| train_m3fd | 3024 | 24593 | 8314 | 12966 | 525 | 345 | 1714 | 729 |
| train_flir | 4124 | 29583 | 8983 | 20600 | 0 | 0 | 0 | 0 |
| sel_m3fd | 336 | 2731 | 908 | 1485 | 46 | 45 | 177 | 70 |
| test_m3fd | 840 | 7083 | 2255 | 3845 | 129 | 131 | 514 | 209 |
| test_flir | 1011 | 8214 | 4104 | 4110 | 0 | 0 | 0 | 0 |
| m3fd_all | 840 | 7083 | 2255 | 3845 | 129 | 131 | 514 | 209 |
| m3fd_day | 782 | 6663 | 2085 | 3617 | 114 | 125 | 513 | 209 |
| m3fd_night | 58 | 420 | 170 | 228 | 15 | 6 | 1 | 0 |
| flir_all | 1011 | 8214 | 4104 | 4110 | 0 | 0 | 0 | 0 |
| flir_day | 991 | 8045 | 4085 | 3960 | 0 | 0 | 0 | 0 |
| flir_night | 20 | 169 | 19 | 150 | 0 | 0 | 0 | 0 |

## Nominal Split Counts (All Listed Pairs)

| Split / Subset | Images | Total Instances | People | Car | Bus | Motorcycle | Lamp | Truck |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train_all | 7150 | 54188 | 17301 | 33574 | 525 | 345 | 1714 | 729 |
| train_m3fd | 3024 | 24593 | 8314 | 12966 | 525 | 345 | 1714 | 729 |
| train_flir | 4126 | 29595 | 8987 | 20608 | 0 | 0 | 0 | 0 |
| sel_m3fd | 336 | 2731 | 908 | 1485 | 46 | 45 | 177 | 70 |
| test_m3fd | 840 | 7083 | 2255 | 3845 | 129 | 131 | 514 | 209 |
| test_flir | 1013 | 8231 | 4107 | 4124 | 0 | 0 | 0 | 0 |
| m3fd_all | 840 | 7083 | 2255 | 3845 | 129 | 131 | 514 | 209 |
| m3fd_day | 782 | 6663 | 2085 | 3617 | 114 | 125 | 513 | 209 |
| m3fd_night | 58 | 420 | 170 | 228 | 15 | 6 | 1 | 0 |
| flir_all | 1013 | 8231 | 4107 | 4124 | 0 | 0 | 0 | 0 |
| flir_day | 993 | 8062 | 4088 | 3974 | 0 | 0 | 0 | 0 |
| flir_night | 20 | 169 | 19 | 150 | 0 | 0 | 0 | 0 |

## Excluded Duplicate-Label Stems

- **train_all** (2): flir_train_FLIR_07226, flir_train_FLIR_07525
- **train_flir** (2): flir_train_FLIR_07226, flir_train_FLIR_07525
- **test_flir** (2): flir_val_FLIR_09055, flir_val_FLIR_09653
- **flir_all** (2): flir_val_FLIR_09055, flir_val_FLIR_09653
- **flir_day** (2): flir_val_FLIR_09055, flir_val_FLIR_09653
