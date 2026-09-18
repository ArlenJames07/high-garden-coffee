# Databricks notebooks

Export or sync the completed Databricks notebooks into this directory using the following names:

- `02_bronze_ingestion.py`
- `03_silver_transformation.py`
- `05_gold_business_metrics.py`
- `06_gold_ml_features.py`
- `07_model_experiments.py`
- `08_model_registration.py`
- `09_batch_forecasting.py`
- `10_market_segmentation.py`
- `11_growth_classifier.py`
- `12_anomaly_detection.py`
- `13_market_opportunity.py`

`04_eda_business_analysis` is intentionally excluded from production orchestration because it is exploratory analysis.

For the technical challenge, keep the notebooks as transparent entry points. A mature production refactor would move reusable logic into `src/high_garden/` and keep notebooks thin.
