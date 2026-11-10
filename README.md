# Integrating Computer Vision and sUAS Imagery for Sweetpotato Storage Root Phenotyping

This repository contains the curated, anonymized datasets and analysis scripts supporting the publication:

> **Integrating Computer Vision and sUAS Imagery for Sweetpotato Storage Root Phenotyping**  
> Alexis Suero, Simon Fraher, G. Craig Yencho, Kenneth Pecota, Jerome Maleski, Michael Kudenov, and Jing Zhang*  
> *Department of Horticultural Science & Department of Electrical and Computer Engineering, North Carolina State University*

---

## Repository Structure

```text
├── data/
│   ├── 2024_sorter_CLI.csv       # Commercial optical sorter reference plot records (Clinton, NC; n = 144 plots)
│   ├── 2024_sorter_CUN.csv       # Commercial optical sorter reference plot records (Kinston/Cunningham, NC; n = 142 plots)
│   ├── 2024_uav_plots.csv        # Aggregated sUAS plot-level phenotypes across both 2024 field trials (n = 288 plots)
│   ├── 2024_uav_roots.csv        # Individual root-level sUAS phenotypes across both 2024 field trials (n = 17,793 roots)
│   └── 2026_uav_test.csv         # Controlled 70-root physical water-displacement & multi-orientation sUAS dataset (n = 420 observations)
├── scripts/
│   ├── utils.py                  # Data loaders, calibration functions, and evaluation metric helpers
│   ├── reproduce_tables.py       # Reproduces Tables 3, 4, 5, 6, and 7 + in-text statistical tests and selection metrics
│   ├── reproduce_figures.py      # Generates analytical figures (Figures 7, 8, 9, 10, and 11 bar panels)
│   └── statistical_models.R      # Fits repeated-measures LMM ANOVAs, EMMs, ICCs, and broad-sense heritability (H^2)
├── figures/                      # Output directory for generated publication figures
├── requirements.txt              # Python package dependencies
└── README.md                     # Repository documentation and data dictionary
```

---

## Data Curation & Anonymization Policy

In accordance with breeding program confidentiality and the Data Policy of ASA, CSSA, and SSSA:
1. **Genotype Anonymization:** All experimental clones and cultivars across the 2024 multi-environment field trials are anonymized as **`G1`–`G137`** (`G1`–`G136` across the $n = 286$ paired sUAS–sorter plots, plus `G137` for a single plot lacking a paired optical sorter record at Kinston). The seven cultivars evaluated in the 2026 controlled physical validation experiment are anonymized as **`C1`–`C7`** (with individual roots designated `C1-01` through `C7-10`). All plot identifiers (`plot_name`), root identifiers (`sp_id`), and image filenames (`filename`) have been scrubbed of original breeding line codes and cultivar names.
2. **Curated Scope:** Only variables and metrics directly used or referenced in the manuscript are retained in the datasets.

---

## Data Dictionary (Metadata & Units)

### 1. `data/2024_sorter_CLI.csv` & `data/2024_sorter_CUN.csv`
Plot-level reference records from the commercial Exeter AccuVision Grading System at the Horticultural Crops Research Station in Clinton, NC (`CLI`, $n = 144$ plots) and the Cunningham Research Station in Kinston, NC (`CUN`, $n = 142$ plots; two plots had missing sorter records).

| Column | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `plot_number` | Integer | — | Standardized field plot number within location |
| `plot_name` | String | — | Anonymized unique plot identifier (`<Trial>_<Plot>_<Rep>_<Genotype>`) |
| `clone_name` | String | — | Anonymized genotype identifier (`G1`–`G136`) |
| `count_all` | Integer | count | Total number of storage roots recorded by the optical sorter in the plot |
| `weight_all_kg` | Float | kg | Total weight of all storage roots in the plot estimated by the optical sorter |
| `mean_ld_all` | Float | ratio | Mean length-to-diameter (LD) ratio across all roots in the plot |

---

### 2. `data/2024_uav_plots.csv`
Plot-level traits aggregated from the sUAS instance segmentation and geodesic centerline pipeline across the 2024 Clinton ($n = 144$) and Kinston/Cunningham ($n = 144$) field trials ($n = 288$ total plots; $n = 286$ paired with optical sorter records).

| Column | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `location_name` | String | — | Field trial station name (`Clinton - Horticultural Crops Research` or `Cunningham Research Station`) |
| `plot_name` | String | — | Anonymized unique plot identifier |
| `clone_name` | String | — | Anonymized genotype identifier (`G1`–`G137`) |
| `plot_number` | Integer | — | Standardized field plot number within location |
| `count_total` | Integer | count | Total number of storage roots segmented by Mask R-CNN in the plot |
| `count_trash` | Integer | count | Number of small unmarketable detections classified as trash ($< 2.54$ cm diameter or $< 5.08$ cm length) |
| `diameter_box_cm_avg` | Float | cm | Plot mean bounding-box width ($W_b$) |
| `diameter_max_cm_avg` | Float | cm | Plot mean centerline maximum orthogonal diameter ($d_{\max}$) |
| `diameter_std_cm_avg` | Float | cm | Plot mean Intra-root Diameter Standard Deviation (IDSD) |
| `length_box_cm_avg` | Float | cm | Plot mean bounding-box length ($L_b$) |
| `length_arc_cm_avg` | Float | cm | Plot mean geodesic centerline arc length ($L_a$) |
| `ld_ratio_avg` | Float | ratio | Plot mean adjusted length-to-diameter ratio ($L/D_{\text{adj}} = L_a / d_{\max}$) |
| `straightness_avg` | Float | index | Plot mean Straightness Index ($\text{SI} = L_s / L_a$, range 0–1) |
| `volume_sph_cm3_total` | Float | $\text{cm}^3$ | Plot total uncalibrated Spheroid volume ($V_{\text{spheroid}}$, Eq. 1) |
| `volume_elp_cm3_total` | Float | $\text{cm}^3$ | Plot total uncalibrated Ellipsoid volume ($V_{\text{ellipsoid}}$, Eq. 2) |
| `volume_asph_cm3_total` | Float | $\text{cm}^3$ | Plot total uncalibrated Adjusted Spheroid volume ($V_{\text{adj. spheroid}}$, Eq. 4) |
| `volume_aelp_cm3_total` | Float | $\text{cm}^3$ | Plot total uncalibrated Adjusted Ellipsoid volume ($V_{\text{adj. ellipsoid}}$, Eq. 5) |
| `volume_seg_cm3_total` | Float | $\text{cm}^3$ | Plot total uncalibrated Segmented volume via Simpson's rule ($V_{\text{segmented}}$, Eq. 6) |
| `weight_total_kg` | Float | kg | Plot total sUAS-estimated storage root weight |

---

### 3. `data/2024_uav_roots.csv`
Individual root-level morphological and volumetric traits extracted from all $17,793$ segmented storage roots across the 2024 Clinton and Kinston/Cunningham field trials.

| Column | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `sp_id` | String | — | Anonymized unique root instance identifier (`<plot_name>_<root_index>`) |
| `location_name` | String | — | Field trial station name |
| `plot_name` | String | — | Anonymized unique plot identifier |
| `clone_name` | String | — | Anonymized genotype identifier (`G1`–`G137`) |
| `plot_number` | Integer | — | Standardized field plot number within location |
| `gsd` | Float | cm/pixel | Orthomosaic Ground Sampling Distance |
| `confidence` | Float | score | Mask R-CNN detection confidence score (0–1) |
| `box_width_cm` | Float | cm | Minimum rotated bounding-box width ($W_b$) |
| `box_length_cm` | Float | cm | Minimum rotated bounding-box length ($L_b$) |
| `arc_length_cm` | Float | cm | Smoothed geodesic centerline arc length ($L_a$) |
| `mean_diameter_cm` | Float | cm | Mean orthogonal ray-cast diameter along the centerline |
| `std_diameter_cm` | Float | cm | Intra-root Diameter Standard Deviation (IDSD) |
| `max_diameter_cm` | Float | cm | Maximum orthogonal ray-cast diameter ($d_{\max}$) |
| `area_cm2` | Float | $\text{cm}^2$ | 2D projected segmentation mask contour area ($A$) |
| `perimeter_cm` | Float | cm | 2D projected segmentation mask perimeter ($P$) |
| `sphere_vol_cm3` | Float | $\text{cm}^3$ | Uncalibrated Spheroid volume ($V_{\text{spheroid}}$, Eq. 1) |
| `ellip_vol_cm3` | Float | $\text{cm}^3$ | Uncalibrated Ellipsoid volume ($V_{\text{ellipsoid}}$, Eq. 2) |
| `adj_sphere_vol_cm3` | Float | $\text{cm}^3$ | Uncalibrated Adjusted Spheroid volume ($V_{\text{adj. spheroid}}$, Eq. 4) |
| `adj_ellip_vol_cm3` | Float | $\text{cm}^3$ | Uncalibrated Adjusted Ellipsoid volume ($V_{\text{adj. ellipsoid}}$, Eq. 5) |
| `segmented_vol_cm3` | Float | $\text{cm}^3$ | Uncalibrated Segmented volume ($V_{\text{segmented}}$, Eq. 6) |
| `straightness` | Float | index | Straightness Index ($\text{SI} = L_s / L_a$, Eq. 3) |
| `ld_ratio` | Float | ratio | Adjusted length-to-diameter ratio ($L_a / d_{\max}$) |
| `weight_g` | Float | g | Individual root estimated weight |
| `main_class` | String | — | Size class assignment (`trash`, `canner`, `no1`, `jumbo`) |

---

### 4. `data/2026_uav_test.csv`
Controlled physical validation dataset comprising 70 storage roots (7 anonymized cultivars `C1`–`C7` $\times$ 10 roots per cultivar) imaged across 3 manual resting orientations (`test` $\in \{1, 2, 3\}$, yielding $n = 210$ root images and $n = 21$ simulated 10-root plot-flights) and evaluated using both automated Mask R-CNN predictions (`dataset == "rcnn"`) and hand-corrected SAM 2 reference masks (`dataset == "sam2"`, $420$ rows total).

| Column | Type | Units | Description |
| :--- | :--- | :--- | :--- |
| `filename` | String | — | Anonymized flight-plot image identifier (`C1_1` through `C7_3`) |
| `clone_name` | String | — | Anonymized cultivar identifier (`C1`–`C7`) |
| `root_id` | Integer | — | Root index within cultivar (`1`–`10`) |
| `root_uid` | String | — | Unique physical root identifier (`C1-01` through `C7-10`) |
| `test` | Integer | — | Resting orientation flight replicate (`1`, `2`, or `3`) |
| `dataset` | String | — | Segmentation mask source (`rcnn` = automated Mask R-CNN; `sam2` = hand-corrected SAM 2 reference) |
| `confidence` | Float | score | Mask R-CNN detection confidence score (`1.0` for `sam2` reference masks) |
| `box_width_cm` | Float | cm | Minimum rotated bounding-box width ($W_b$) |
| `box_length_cm` | Float | cm | Minimum rotated bounding-box length ($L_b$) |
| `arc_length_cm` | Float | cm | Geodesic centerline arc length ($L_a$) |
| `mean_diameter_cm` | Float | cm | Mean orthogonal ray-cast diameter |
| `std_diameter_cm` | Float | cm | Intra-root Diameter Standard Deviation (IDSD) |
| `max_diameter_cm` | Float | cm | Maximum orthogonal ray-cast diameter ($d_{\max}$) |
| `area_cm2` | Float | $\text{cm}^2$ | 2D mask contour area ($A$) |
| `perimeter_cm` | Float | cm | 2D mask contour perimeter ($P$) |
| `sphere_vol_cm3` | Float | $\text{cm}^3$ | Uncalibrated Spheroid volume ($V_{\text{spheroid}}$, Eq. 1) |
| `ellip_vol_cm3` | Float | $\text{cm}^3$ | Uncalibrated Ellipsoid volume ($V_{\text{ellipsoid}}$, Eq. 2) |
| `adj_sphere_vol_cm3` | Float | $\text{cm}^3$ | Uncalibrated Adjusted Spheroid volume ($V_{\text{adj. spheroid}}$, Eq. 4) |
| `adj_ellip_vol_cm3` | Float | $\text{cm}^3$ | Uncalibrated Adjusted Ellipsoid volume ($V_{\text{adj. ellipsoid}}$, Eq. 5) |
| `segmented_vol_cm3` | Float | $\text{cm}^3$ | Uncalibrated Segmented volume ($V_{\text{segmented}}$, Eq. 6) |
| `straightness` | Float | index | Straightness Index ($\text{SI} = L_s / L_a$, Eq. 3) |
| `true_volume_cm3` | Float | $\text{cm}^3$ | Physical ground-truth root volume measured via water displacement (Archimedes' principle) |
| `true_weight_g` | Float | g | Physical ground-truth root weight measured on a precision digital scale |
| `true_density_g_cm3` | Float | $\text{g/cm}^3$ | Physical ground-truth root density ($\rho = \text{true\_weight\_g} / \text{true\_volume\_cm3}$, Eq. 8) |

---

## Reproducing Manuscript Results

### 1. Software Requirements
- **Python $\ge$ 3.10** with packages listed in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```
- **R $\ge$ 4.3** with packages `lme4`, `lmerTest`, `emmeans`, and `multcomp`:
  ```R
  install.packages(c("lme4", "lmerTest", "emmeans", "multcomp"))
  ```

### 2. Reproduce Validation Tables & Summary Statistics
From the root of this repository, run:
```bash
python scripts/reproduce_tables.py
```
This script outputs:
- **Table 3:** Individual root-level volumetric validation ($n = 210$ root images across 3 orientations, LORO-CV, LOCO-CV, Boundary-Scaled, Proportional, and Raw).
- **Table 4:** Aggregated simulated plot-level volumetric validation ($n = 21$ plot-flights and $n = 7$ cultivar means).
- **Table 5:** Cross-validated weight estimation across volumetric models and density parameterizations (Fixed $1.05\text{ g/cm}^3$, Pooled CV $\approx 0.98\text{ g/cm}^3$, Cultivar CV, and Cultivar).
- **Table 6:** Out-of-sample field plot-level weight validation across the combined 2024 Clinton and Kinston trials ($n = 286$ plots).
- **Table 7:** Out-of-sample field plot-level weight validation for the 2024 Clinton trial ($n = 144$ plots).
- **In-text statistical metrics:** Breusch–Pagan heteroskedasticity tests and cross-platform genotype selection concordance ($\rho$, $r$, and top-$10\%$ / top-$20\%$ selection overlap).

### 3. Reproduce Publication Figures
From the root of this repository, run:
```bash
python scripts/reproduce_figures.py
```
This saves the publication-ready charts to `figures/`:
- `figures/res_raw_volume_anova.png` (Figure 7)
- `figures/res_4x4_volume.png` (Figure 8)
- `figures/res_genotype_concordance.png` (Figure 9)
- `figures/res_1x3_metrics.png` (Figure 10)
- `figures/res_top_bottom_si.png` & `figures/res_top_bottom_idsd.png` (Figure 11 bar chart panels)

### 4. Reproduce Linear Mixed-Effects Models (LMMs), ANOVAs, ICCs, and Heritability ($H^2$)
From the root of this repository, run:
```bash
Rscript scripts/statistical_models.R
```
This executes:
- Two-way repeated-measures LMM ANOVA and Tukey HSD compact letter displays on raw volumetric APE across geometric models and segmentation mask sources.
- Three-level density parameterization LMM ANOVA and pairwise contrasts at both individual root ($n = 210$) and simulated plot ($n = 21$) scales.
- Single-view and three-view orientation repeatability (intraclass correlation coefficient, ICC) for shape traits on the 70-root dataset.
- Entry-mean broad-sense heritability ($H^2$, Eqs. 9–10) across the 2024 multi-environment field trials for total plot yield and LD ratio (both sUAS and optical sorter) and root-level morphological traits (`std_diameter_cm` [IDSD], `ld_ratio`, and `straightness` [SI]).

---

## License & Citation
If you use these datasets or scripts in your research, please cite the accompanying manuscript.
