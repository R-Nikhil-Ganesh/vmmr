# Ground-Truth Color Validation Report (`E-09`)

## 1. Overview & Ground-Truth Source
- **Human Ground-Truth Dataset**: CompCars Surveillance (`sv_data/color_list.mat`), matched via `external_metadata_51make.csv` against [`annotation_runs/run2/annotations.csv`](../annotation_runs/run2/run_config.json).
- **Matched Human-Labeled Rows in Run 5**: **19,479** images (`train`: `13,753`, `val`: `2,859`, `test`: `2,867`).
- **Stratified Manual Review Sheet**: Generated [`annotation_runs/color_human_review_600.csv`](../annotation_runs/color_human_review_600.csv) (`598` rows stratified across all 13 Qwen colors, with 30% drawn from `color_conf < 0.70`).

## 2. Qwen3-VL-8B Agreement with CompCars Human Color Labels

> [!NOTE]
> CompCars uses a 10-color taxonomy (`black, white, red, yellow, blue, green, purple, brown, champagne, silver`) where **gray and silver are merged into class `9: silver`** and **champagne/beige/gold are merged into class `8: champagne`**. We report both:
> 1. **Strict Agreement**: Treats Qwen `gray` on a CompCars `9: silver` car as an error.
> 2. **Taxonomy-Aligned (Grouped) Agreement**: Aligns Qwen `{silver, gray}` $\to$ `silver`, `{beige, gold}` $\to$ `champagne`, `{yellow, orange}` $\to$ `yellow`, and `other` $\to$ `purple`.

| Subset | Rows | Share | Strict Agreement | Taxonomy-Aligned Agreement |
| :--- | ---: | ---: | ---: | ---: |
| **All Matched Rows** | `19,479` | `100.00%` | `70.21%` | **`78.44%`** |
| **High Confidence (`color_conf >= 0.70`)** | `18,433` | `94.63%` | `71.93%` | **`79.70%`** |
| **Low Confidence (`color_conf < 0.70`)** | `1,046` | `5.37%` | `39.87%` | **`56.21%`** |
| **Train Split** | `13,753` | `70.60%` | `69.95%` | **`78.08%`** |
| **Val Split** | `2,859` | `14.68%` | `70.23%` | **`78.91%`** |
| **Test Split** | `2,867` | `14.72%` | `71.43%` | **`79.67%`** |

## 3. Recall by CompCars Human Ground-Truth Color

| CompCars ID | Human Color | Samples | Strict Recall | Aligned Recall | Top-3 Qwen3-VL Predictions |
| :---: | :--- | ---: | ---: | ---: | :--- |
| `0` | **black** | `7,004` | `91.98%` | **`91.98%`** | `black` (6,442), `gray` (421), `silver` (99) |
| `1` | **white** | `3,567` | `72.92%` | **`72.92%`** | `white` (2,601), `silver` (903), `gray` (44) |
| `2` | **red** | `1,448` | `78.59%` | **`78.59%`** | `red` (1,138), `brown` (219), `black` (73) |
| `3` | **yellow** | `454` | `28.63%` | **`38.55%`** | `yellow` (130), `brown` (56), `gray` (56) |
| `4` | **blue** | `531` | `52.54%` | **`52.54%`** | `blue` (279), `black` (118), `gray` (77) |
| `5` | **green** | `163` | `14.11%` | **`14.11%`** | `black` (94), `gray` (41), `green` (23) |
| `6` | **purple** | `76` | `0.00%` | **`88.16%`** | `other` (67), `black` (9) |
| `7` | **brown** | `585` | `10.60%` | **`10.60%`** | `black` (298), `gray` (194), `brown` (62) |
| `8` | **champagne** | `444` | `3.60%` | **`4.73%`** | `gray` (214), `silver` (126), `black` (81) |
| `9` | **silver** | `5,207` | `57.33%` | **`85.87%`** | `silver` (2,985), `gray` (1,486), `black` (703) |

## 4. Precision & Confidence Calibration by Qwen3-VL Predicted Color

| Qwen Predicted Color | Samples | Mean `color_conf` | `% < 0.70` | Strict Precision | Aligned Precision | Top-3 Human Ground-Truth Labels |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **black** | `7,861` | `0.9850` | `1.91%` | `81.95%` | **`81.95%`** | `black` (6,442), `silver` (703), `brown` (298) |
| **silver** | `4,235` | `0.9377` | `8.24%` | `70.48%` | **`70.48%`** | `silver` (2,985), `white` (903), `champagne` (126) |
| **white** | `2,628` | `0.9594` | `5.02%` | `98.97%` | **`98.97%`** | `white` (2,601), `black` (15), `silver` (8) |
| **gray** | `2,540` | `0.8993` | `12.17%` | `0.00%` | **`58.50%`** | `silver` (1,486), `black` (421), `champagne` (214) |
| **red** | `1,173` | `0.9803` | `2.39%` | `97.02%` | **`97.02%`** | `red` (1,138), `black` (11), `yellow` (10) |
| **brown** | `352` | `0.9287` | `9.66%` | `17.61%` | **`17.61%`** | `red` (219), `brown` (62), `yellow` (56) |
| **blue** | `291` | `0.9749` | `3.09%` | `95.88%` | **`95.88%`** | `blue` (279), `black` (9), `yellow` (1) |
| **yellow** | `131` | `0.9729` | `3.82%` | `99.24%` | **`99.24%`** | `yellow` (130), `black` (1) |
| **other** | `72` | `0.9873` | `1.39%` | `0.00%` | **`93.06%`** | `purple` (67), `red` (3), `yellow` (1) |
| **beige** | `62` | `0.8294` | `24.19%` | `25.81%` | **`25.81%`** | `yellow` (23), `champagne` (16), `silver` (12) |
| **orange** | `55` | `0.9176` | `9.09%` | `0.00%` | **`81.82%`** | `yellow` (45), `brown` (8), `red` (2) |
| **gold** | `51` | `0.8869` | `15.69%` | `0.00%` | **`9.80%`** | `yellow` (42), `champagne` (5), `silver` (3) |
| **green** | `28` | `0.9389` | `3.57%` | `82.14%` | **`82.14%`** | `green` (23), `blue` (3), `yellow` (2) |
