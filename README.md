# Cross-continental flood susceptibility mapping (Tetouan, Morocco and Sunamganj, Bangladesh)

Code accompanying the manuscript *Cross-continental Flood Susceptibility Mapping Using Machine Learning on Google Earth Engine: Case Studies from South Asia and North Africa* (submitted to Computers & Geosciences).

Contact: Demba Coulibaly, demba.coulibaly@usmba.ac.ma

## Purpose

Random Forest (RF) and CART flood susceptibility models for two areas, built with the same workflow: 12 conditioning factors (elevation, slope, aspect, plan curvature, LULC, drainage density, distance to rivers, TWI, TPI, SPI, NDVI, rainfall), 210 flood and 115 non-flood points per area, random and spatial 5-fold cross-validation, and an independent 80/20 held-out test (AUROC, accuracy, precision, sensitivity, F1).

## Repository content

| File | Language | Description |
|---|---|---|
| `Code_GEE.js` | JavaScript (GEE Code Editor) | Computes the conditioning factors for both areas (slope, aspect, TWI, SPI, TPI, NDVI, CHIRPS rainfall, LULC with RF and CART), detects floods from Sentinel-1 (pre/post VH backscatter, Refined Lee filter), draws flood / non-flood points, overlays flood extent on LULC, and exports layers as GEE assets or to Google Drive |
| `CODE_COLAB_Tetouan.py` | Python 3 (Google Colab) | Tetouan: stacks the factor layers, trains RF and CART on the flood / non-flood points, random and spatial 5-fold CV, held-out metrics, ROC curve, factor importance, four-class susceptibility maps and class areas |
| `quick_test.py` | Python 3 | Quick test of the RF / CART validation step (real sample table if provided, synthetic data otherwise; no Earth Engine account needed) |
| `training_data_with_coords.csv` | CSV | Sample table (point labels, 12 factor values, coordinates) used by the quick test. Add it to the repository |

## Requirements

- Any computer with a web browser and an internet connection (computations run on Google Earth Engine and Google Colab).
- A registered, free Google Earth Engine account: https://earthengine.google.com
- Python 3 packages: `earthengine-api`, `geemap`, `geopandas`, `rasterio`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`

## Quick test (about 1 minute)

`quick_test.py` re-runs the modelling and validation step (RF with 200 trees and CART; random 5-fold CV, spatial 5-fold CV on grid blocks of the coordinates, and a random 80/20 held-out test; AUROC, accuracy, precision, sensitivity, F1). No Earth Engine account is needed.

```bash
git clone https://github.com/dembacoulibaly-max/Cross-continental-from-South-Asia-and-North-Africa.git
cd Cross-continental-from-South-Asia-and-North-Africa
pip install numpy pandas scikit-learn
python quick_test.py
```

- If `training_data_with_coords.csv` (the sample table written by `CODE_COLAB_Tetouan.py`: `label`, the 12 factors, point coordinates) is in the same folder, the test uses it. Another file can be passed as an argument: `python quick_test.py my_points.csv`.
- If the file is absent, the test uses synthetic data (210 flood / 115 non-flood points) and only checks that the environment works; the numbers then do not reproduce the paper.

Expected output: the data source, the number of points and factors, a table with AUROC (random 5-fold, spatial 5-fold, held-out) and held-out accuracy, precision, sensitivity and F1 for RF and CART, then `Quick test finished successfully.`

## Running the full workflow

1. **GEE script.** Open `Code_GEE.js` in the Code Editor (https://code.earthengine.google.com). The script expects the study-area geometries and training polygons to be defined in the *Imports* panel (see the table below). Run the tasks in the *Tasks* tab.
2. **Colab script.** Open `CODE_COLAB_Tetouan.py` in Google Colab, run `ee.Authenticate()` and `ee.Initialize(project="<your-project>")` with your own Earth Engine project, and run the cells in order.
3. Replace every `projects/ee-dembacoulibaly/assets/...` path with the location of your own copy of the layers produced by step 1.

### Inputs defined outside the script (Imports panel of the Code Editor)

| Name in the script | Content |
|---|---|
| `maroc`, `bangladesh` | Study-area polygons |
| `fiil_morocco`, `fill_bangladesh` | Depression-filled SRTM DEMs |
| `Flow_accma`, `Flow_accba` | Flow accumulation rasters |
| `plancur_ma`, `plancur_ba` | Plan curvature rasters |
| `dis_rima`, `dist_riba`, `drai_ma`, `drai_ba` | Distance to rivers and drainage density rasters |
| `water_ma`, `culture_ma`, `building_ma`, `vegetation_ma`, `barren_landma` (and `*_ba`) | LULC training polygons |
| `PTinontetouan`, `ptnoniniteto`, `PTINONBANG`, `PTNONIONDBANG` | Flood / non-flood point geometries |

> Flow direction, flow accumulation, plan curvature, drainage density and distance to rivers were derived from the depression-filled DEM with ArcGIS; the resulting rasters are the inputs above.

## License

Optional: add a LICENSE file (e.g. MIT) to define third-party use.
