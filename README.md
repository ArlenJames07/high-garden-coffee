# High Garden Coffee — MLOps Market Intelligence

Technical challenge implementing a governed Azure Databricks lakehouse and multi-model ML workflow for domestic coffee-consumption intelligence.

## Business objective

Use historical domestic coffee-consumption data to support demand planning and market prioritization. The source dataset contains consumption, not prices, so this implementation does **not** fabricate price forecasts.

## Architecture

```text
ADLS Gen2
   ↓
Bronze (Delta)
   ↓
Silver longitudinal consumption
   ↓
Gold business metrics + ML features
   ↓
┌──────────────┬──────────────────┬──────────────┬────────────────┐
│ Forecasting  │ Growth classifier│ Segmentation │ Anomaly detect │
│ Naive/CatBoost│ CatBoost        │ K-Means      │ IsolationForest│
└──────────────┴──────────────────┴──────────────┴────────────────┘
   ↓
MLflow + Unity Catalog
   ↓
Gold market opportunities
   ↓
Dashboard / GenAI consumption layer
```

## Current validated results

- Forecast Champion: persistence/naive baseline
  - mean temporal-backtest WAPE ≈ 2.25%
  - global scaled MASE ≈ 0.76
- CatBoost forecast candidate underperformed the baseline
  - WAPE ≈ 8.95%
  - MASE ≈ 3.00
- Growth classifier
  - ROC-AUC ≈ 0.920
  - F1 ≈ 0.741
  - Precision ≈ 0.725
  - Recall ≈ 0.769
  - Brier ≈ 0.106
- K-Means segmentation: 2 data-driven market segments
- Isolation Forest: market-year anomaly detection

## MLOps design

- Data: ADLS Gen2 + Delta Lake + Unity Catalog
- Experiment tracking: MLflow
- Model lifecycle: Unity Catalog registered models and Champion alias
- Orchestration: Lakeflow Jobs defined as Declarative Automation Bundle resources
- Dependency management: `uv` / `pyproject.toml` (generate and commit `uv.lock` after resolving dependencies)
- CI: GitHub Actions + Ruff + Pytest
- CD: GitHub Actions + Databricks Declarative Automation Bundles
- Inference: batch, appropriate for annual source data

## Repository layout

```text
.
├── databricks.yml
├── notebooks/
├── resources/
│   ├── data_job.yml
│   ├── training_job.yml
│   └── inference_job.yml
├── src/high_garden/
├── tests/
└── .github/workflows/
```

## Local setup

```bash
uv lock
uv sync
uv run ruff check .
uv run pytest -v
```

## Databricks bundle

Set the bundle variables for your workspace and cluster, then:

```bash
export BUNDLE_VAR_workspace_host="https://<workspace>.azuredatabricks.net"
export BUNDLE_VAR_cluster_id="<cluster-id>"

databricks bundle validate -t dev
databricks bundle deploy -t dev
```

For production deployment use the `prod` target. Production CI/CD should authenticate with a service principal through GitHub OIDC rather than a long-lived personal token.

## Important modeling decisions

1. Temporal validation is used instead of random train/test splitting.
2. `Total_domestic_consumption` is excluded as leakage.
3. Zero consumption is preserved as an observed value, not treated as missing.
4. The most complex model is not automatically promoted; the baseline remains Champion when it generalizes better.
5. The opportunity score is a configurable decision-support rule, not a learned prediction of commercial success.
