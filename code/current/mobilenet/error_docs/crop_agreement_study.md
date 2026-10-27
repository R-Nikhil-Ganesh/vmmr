# Vehicle Cropping & Color Agreement Study (`E-01` & `E-02`)

## 1. Source Image Audit on the Research PC
Direct inspection of `/home/researchadmin/Econ/resized_512` (PlatesMania) and `/home/researchadmin/Econ/external_resized_512` (External) revealed that **PlatesMania and 4 of the 5 External sources were already cropped to vehicle bounding boxes on disk before Run 4/5**:

| Dataset Source | Disk Resolution | Pre-Cropped to BBox on Disk? | Bounding-Box Source |
| :--- | :---: | :---: | :--- |
| **PlatesMania (`resized_512`)** | `512×512` | **Yes (`100%`)** | Pre-cropped to vehicle bbox + white letterboxed |
| **External: `stanford_cars`** | `512×512` | **Yes (`100%`)** | `cars_train_annos.mat` / `cars_test_annos_withlabels.mat` (`L1 = 0.04`) |
| **External: `compcars_web`** | `512×512` | **Yes (`100%`)** | `data/label/<make>/<model>/<year>/<id>.txt` (`L1 = 0.04`) |
| **External: `compcars_cctv`** | `512×512` | **Yes (`100%`)** | YOLOv8s vehicle detector (`+2%` margin) |
| **External: `resized_cupra`** | `512×512` | **Yes (`100%`)** | Copied from `resized_512/Cupra` |
| **External: `boxcars116k`** | `512×512` | **No (`0%` — fixed in manifest)** | Raw `BoxCars116k` images have a 30px road border around `inst['2DBB']`; all `18,520` `2DBB` boxes are now populated in `boxcars116k_bboxes_512.json` and `splits_1235models/*.csv` |

## 2. 300-Sample Crop Sensitivity Experiment (`E-02`)
- **Sample Size**: `300` images (`150` PlatesMania, `150` `BoxCars116k` with recovered `2DBB` bounding boxes).
- **Overall Color Prediction Agreement (Legacy Crop vs. Tight Vehicle BBox Crop)**: **`71.33%`**
  - **PlatesMania (`150` images, already pre-cropped on disk)**: **`76.00%`** agreement.
  - **External `BoxCars116k` (`150` images, removing the 30px road/pavement border via `2DBB`)**: **`66.67%`** agreement (`33.33%` of `BoxCars116k` images change predicted color when the surrounding gray/black asphalt border is removed!).
