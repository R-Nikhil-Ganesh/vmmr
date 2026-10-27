# Error Audits, Verification Studies & Implementation Plans

This folder consolidates all audit logs, empirical probe reports, and execution plans produced during the resolution of items `E-01` through `E-21`.

---

## Index of Documents

| File | Description | Related Error IDs |
| :--- | :--- | :---: |
| [`errors.md`](errors.md) | **Primary Error & Resolution Register**: Catalogs all 21 items, empirical findings, and final resolution status. | `E-01` – `E-21` |
| [`checkpoint.md`](checkpoint.md) | **Execution Checkpoint**: Phase-by-phase log tracking the resolution of all errors from start to completion. | `E-01` – `E-21` |
| [`errors_master_implementation_plan.md`](errors_master_implementation_plan.md) | **Master Plan**: The 5-phase engineering implementation plan used to execute all fixes. | `E-01` – `E-21` |
| [`crop_agreement_study.md`](crop_agreement_study.md) | **Vehicle Bounding-Box Sensitivity Study**: 300-sample probe evaluating full-frame vs tight vehicle crops on color predictions. | `E-01`, `E-02` |
| [`color_ground_truth_validation.md`](color_ground_truth_validation.md) | **Ground-Truth Color Validation**: Validation against 19,479 CompCars CCTV human-labeled images. | `E-09` |
| [`why_external_share_is_low.md`](why_external_share_is_low.md) | **External Dataset Share Analysis**: Investigation into class overlaps, 51-make expansion, and < 50 sample pruning. | `E-03` |
| [`documentation_collisions_plan.md`](documentation_collisions_plan.md) | **Documentation Alignment Plan**: Resolution map for collisions across README, run docs, and paths. | `E-11` – `E-16` |
| [`e01_e02_e03_plan.md`](e01_e02_e03_plan.md) | **Initial Data Defect Plan**: Strategy for resolving initial cropping and dataset share defects. | `E-01`, `E-02`, `E-03` |
