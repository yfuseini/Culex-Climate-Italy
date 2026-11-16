# Culex pipiens habitat suitability: method and results

This guide describes how the *Culex pipiens* habitat suitability map on the CERRA grid was built, why the main choices were made, and how to read the result. Weekly female trap counts from Italy (Longo et al., 2008-2022) were modelled with XGBoost and Random Forest, using season, temperature history, elevation and surface water as predictors. One script produces the features for both the training table and the map. When a whole region is held out, the final model ranks its trap cells with a Spearman correlation of about 0.5. The map shows relative suitability, not mosquito numbers.

## 1. Task and data

The challenge asks for a habitat suitability map on the CERRA grid, as NetCDF or GeoTIFF. No weather data are supplied, so predictors can come from any source. The training data are the harmonised adult abundance records of Longo et al.: weekly counts per 0.1 degree grid cell in 17 Italian regions. Trap positions are confidential, so all work is done per cell.

Three filters were applied: female mosquitoes only, CO2-baited CDC traps only (`CDC_CO2`, `CDC_LIKE_CO2`), and no rows with imputed coordinates. Counts from different trap types are not comparable, and CO2 traps mostly catch host-seeking females. This leaves 25,204 rows and 532 cells. The target is log(1 + count), which limits the influence of a few very large counts.

The main difficulty is that region, institute and trap protocol overlap almost completely. The mean log count in weeks 21-41 runs from 1.2 in Apulia to 4.5 in Emilia-Romagna, a factor of about 27. Part of that gap is real and part is sampling, and the data cannot separate the two. A model that can identify a region looks accurate without learning anything about habitat. Region, trap type and sampling effort are therefore never used as predictors, and validation concentrates on regions the model has not seen.

## 2. Predictors

Temperature comes from E-OBS, which is daily, observation based and on the same 0.1 degree grid as the trap cells. CERRA fields were not used as predictors. Many are forecast variables that need careful handling of accumulations, and the standard CERRA record may end in mid-2021. CERRA is used only as the output grid. Elevation comes from the Copernicus DEM (GLO-90) and surface water from the JRC Global Surface Water occurrence map, with the sea removed.

All features are computed for the Sunday of the ISO sampling week, in one script (`culex_prep_features_v3.py`) that serves both the training table and the map. The model therefore sees the same definitions in both places.

| Group | Features | Meaning |
| --- | --- | --- |
| season | `week_of_year` | Seasonal cycle |
| climate_core | `gdd_ytd` | Degree-days since 1 Jan |
| climate_core | `tas_0_2w` | Mean temperature, last 2 weeks |
| climate_core | `tas_4_8w` | Mean temperature, weeks 4-8 back |
| climate_extra | `tas_2_4w` | Mean temperature, weeks 2-4 back |
| climate_extra | `tasmin_0_4w`, `tasmax_0_4w` | Daily min and max, 4 weeks |
| climate_extra | `hot_days_0_4w` | Days above 30 C in 4 weeks |
| climate_extra | `winter_tasmin` | Mean minimum, previous Dec-Feb |
| topo | `elev_mean`, `elev_std` | Elevation and roughness in the cell |
| water | `water_frac`, `water_frac_nb` | Water share, cell and 3 x 3 cells |

The windows reflect the life cycle: adults caught in a given week developed from eggs and larvae in the preceding weeks. The 10 C degree-day base and the 30 C hot-day threshold are assumptions and were not tuned.

Several other inputs were tested and dropped. CORINE and LUCAS land cover helped within regions that already had traps but not in new regions (section 4). Precipitation and humidity changed skill by less than the noise. Trap type and sampling effort lowered out-of-region skill. Subtracting each region's mean from the target did not help.

## 3. Models and validation

XGBoost and Random Forest regress log(1 + count) on the features. Trees handle nonlinear effects and interactions without specification, and the two models make different errors, so their average is more stable. Each trap cell has equal total weight, because Emilia-Romagna and Veneto together supply over half of the rows. Optuna tunes the settings within deliberately narrow ranges (shallow trees, large minimum leaf size), since the main risk is memorising cells.

Random cross-validation would put a cell and its neighbours in both training and test sets and inflate the scores. In an early check the model reproduced trap-cell means with a Spearman correlation of 0.97 on cells it was trained on, but 0.51 when the region was held out. Models are therefore judged by leave-region-out validation (train on all other regions, predict the held-out one) and by a 2020-2022 holdout. Training uses 2008-2019. The holdout is split into cells seen in training, new cells in regions with training data, and new regions. The last group (722 rows) is the closest match to mapping places without traps.

The score used for decisions is the Spearman rank correlation between observed and predicted cell means over weeks 21-41, which is what the map shows. Scores on individual weekly rows are not used, because the seasonal cycle alone gives about 0.47 on them. A control model with only season and coordinates shows how much skill comes from spatial neighbourhood. It matches the full models on new cells in seen regions, so that subset cannot validate habitat features. It scores near zero on new regions, so that test is not contaminated by neighbours.

## 4. Ablation and final feature set

Feature groups were added one at a time and removed one at a time. Table 1 shows selected stages for the raw target. "Within" is the cell-mean Spearman correlation averaged over held-out regions. "New" is the same score on the 2020-2022 holdout for regions with no training data. RF is Random Forest and XGB is XGBoost.

Table 1. Cell-mean Spearman correlation by feature set.

| Feature set | Within RF | Within XGB | New RF | New XGB |
| --- | --- | --- | --- | --- |
| Season only | -0.003 | 0.004 | 0.068 | -0.022 |
| + elevation | 0.595 | 0.575 | 0.165 | 0.001 |
| + core temperature | 0.601 | 0.583 | 0.184 | 0.087 |
| All groups with land cover | 0.508 | 0.531 | 0.272 | 0.381 |
| All groups without land cover | 0.563 | 0.559 | 0.460 | 0.492 |
| Season + coordinates | 0.330 | 0.275 | -0.228 | -0.327 |

Elevation is the strongest single signal, and degree-days add a little. Land cover raises the pooled score and the score on new cells in seen regions, but it lowers within-region skill and does not transfer to new regions: models that include it score 0.22 to 0.41 there, against 0.46 to 0.49 without it. The coordinates control matches the land cover models on new cells in seen regions, so the gain there may be spatial interpolation. The effect of land cover also changed sign between regions (urban, industrial, rice), which points to trap placement more than habitat. Only elevation (negative) and degree-days (positive) kept the same sign in all nine regions checked.

Subtracting the region mean from the target lowered within-region skill (0.60 to 0.47 for the core model) and made land cover worse on new regions, so the raw target is used.

The final model uses season, topography, core and extra temperature, and water. It scored 0.46 and 0.49 on new regions (Random Forest, XGBoost) and 0.56 within regions, close to the best within-region value of 0.60. These numbers rest on few regions, and the new-region score on 722 rows. No confidence intervals were computed, and the feature set was chosen using the same results that are reported, so the scores are optimistic. A bootstrap over cells is the next step.

## 5. Final model and map

Each model is tuned once with leave-region-out folds on the chosen features and then fitted on all years with five seeds. The out-of-region cell-mean Spearman correlation of this configuration is 0.51 for Random Forest and 0.53 for XGBoost.

For the map, the models predict log(1 + count) for every E-OBS land cell in the box 6.5 to 18.8 E and 36.0 to 47.2 N (6,345 cells), for each week 21 to 41 and each weather year 2008 to 2022. Predictions are averaged over weeks, years, seeds and models, clipped at the 1st and 99th percentiles, and rescaled to 0-1. The map is therefore climatological and relative. A value of 1 means one of the highest predicted abundances on this map and 0 one of the lowest. It is not a probability.

The 0.1 degree fields are regridded to the CERRA grid (about 5.5 km) by bilinear interpolation after filling the coast by two cells, and masked to land. This adds no detail, since each 0.1 degree value covers about four CERRA cells.

| File | Contents |
| --- | --- |
| `culex_suitability_0.1deg.nc`, `.tif` | 0.1 degree grid |
| `culex_suitability_CERRA.nc` | CERRA grid |
| `final_model_spec.json` | Features, settings, seeds |
| `final_loro_check.csv` | Out-of-region score per region |

| Variable | Meaning |
| --- | --- |
| `suitability` | Relative suitability, 0 to 1 |
| `suitability_masked` | Same, missing where extrapolating |
| `outside_frac` | Share of features outside the training range |
| `interannual_sd` | Year-to-year spread of the prediction |
| `log1p_abundance_mean` | Mean predicted log(1 + count), unscaled |

`suitability_masked` is missing where more than half of the features fall outside the range seen in the trap cells, which removes the Alps, the Balkan mountains and parts of Tunisia.

The highest values lie in the Po plain, which is also where the highest counts were trapped, so part of that signal may reflect trapping methods. Values are low in the mountains and somewhat higher on parts of the Tyrrhenian coast. Traps exist only in northern and central Italy and parts of Apulia and Basilicata. Values in Sicily, Sardinia and the south are extrapolated even where the features look familiar, and values outside Italy have no support.

## 6. Limitations

- The map ranks places. It does not correct for how institutes sample and does not predict mosquito numbers.
- The CO2 CDC subset has no traps in Sicily and Sardinia and almost none in the south.
- Evidence of skill in new regions is weak and uncertain (section 4).
- The model works at 0.1 degree, so the CERRA map has no finer detail.
- The map check has not yet been repeated for the final configuration.

## 7. Running the pipeline

1. `culex_prepare_static_layers.py` mosaics the DEM and water tiles and builds the CORINE fractions the feature script expects. It prints the file paths to paste into the next script.
2. `culex_prep_features_v3.py` writes the training table, the feature group file and the map features. It needs several gigabytes of memory.
3. `culex_ablation.py` (optional) reproduces Table 1. Set `TARGET_MODE` to `raw` and change `RUN_TAG` whenever the prep settings change.
4. `culex_final_map.py` tunes and fits the final models and writes the maps. Set `FINAL_GROUPS` to the five groups in section 4.
5. `culex_map_check.py` and `culex_plot_maps.py` check and plot the result.