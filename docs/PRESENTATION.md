# High Garden Coffee — Technical Challenge Presentation Guide

## 1. Business problem

The source contains historical domestic coffee consumption by country and coffee type. The solution forecasts next-period consumption, estimates growth direction, segments markets, detects anomalies, and consolidates model outputs into a decision-support opportunity layer.

Important boundary: the source does not contain prices, so the solution does not claim to forecast prices.

## 2. Architecture

```text
ADLS Gen2
   ↓
Bronze / Delta
   ↓
Silver longitudinal consumption
   ↓
Gold business metrics + ML features
   ↓
Forecast | Growth classifier | K-Means | Isolation Forest
   ↓
MLflow + Unity Catalog
   ↓
Gold model outputs
   ↓
Market opportunity decision layer
   ↓
AI/BI Dashboard + Genie

GitHub → CI → Databricks Bundle → Lakeflow Jobs → Serverless
```

## 3. Data engineering

Show Unity Catalog and the three layers:

- `high_garden.bronze.coffee_raw`
- `high_garden.silver.coffee_consumption`
- Gold business and ML tables

Explain that the wide source is normalized into one row per country/crop-year, zeros are preserved as observed values, and `Total_domestic_consumption` is excluded from predictive features because it leaks future information.

## 4. Forecasting

Temporal rolling-origin validation is used instead of random splitting.

Current model comparison:

- Persistence/naive baseline: WAPE ~2.25%, MASE ~0.76
- CatBoost: WAPE ~8.95%, MASE ~3.00

The simpler baseline remains Champion because it generalized better. This is intentional model governance, not a failure to use a complex model.

## 5. Growth classifier

Report:

- ROC-AUC ~0.920
- F1 ~0.741
- Precision ~0.725
- Recall ~0.769
- Brier ~0.106

Do not claim perfect calibration from Brier alone.

## 6. Unsupervised models

- K-Means: two data-driven market segments.
- Isolation Forest: retrospective unusual market-year behavior. The 5% contamination parameter is an operational assumption, not a known true anomaly rate.

## 7. Opportunity layer

The opportunity score combines forecast demand, growth probability, stability, and anomaly safety. It is a configurable business decision layer, not a supervised estimate of profit or commercial success.

## 8. MLOps

Show:

- GitHub repository
- `databricks.yml`
- `resources/*.yml`
- three successful Lakeflow Jobs
- MLflow experiment
- Unity Catalog registered model `high_garden.ml.coffee_forecaster`
- `Champion` alias

The project uses serverless compute for notebook tasks.

## 9. CI/CD

CI uses GitHub Actions with `uv`, Ruff, and Pytest.

Production deployment is defined with Databricks Declarative Automation Bundles. The production GitHub deployment workflow is manual until the workspace's GitHub OIDC federation policy and GitHub environment variables are configured. This avoids storing long-lived Databricks credentials.

## 10. Dashboard / GenAI

The AI/BI dashboard should show executive KPIs, top opportunities, demand-vs-growth matrix, segments, anomaly table, and model performance.

Genie is used only as a grounded conversational analytics layer over governed Gold tables. It explains and queries model outputs; it does not generate the numerical forecast itself.

## Suggested 8–10 minute demo

1. 45 sec — problem and data limitation.
2. 60 sec — architecture.
3. 60 sec — medallion data pipeline and Unity Catalog.
4. 90 sec — temporal backtest and Champion selection.
5. 60 sec — classifier / segmentation / anomalies.
6. 60 sec — opportunity layer.
7. 90 sec — GitHub, Bundle, Lakeflow Jobs, MLflow, UC model.
8. 60 sec — dashboard / Genie.
9. 30 sec — production hardening and monitoring.
