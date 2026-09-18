# High Garden Coffee — AI/BI Dashboard Build

Create one Databricks AI/BI dashboard named **High Garden Coffee — Market Intelligence**.

## Data source

Use a SQL Warehouse and the queries in `sql/dashboard_queries.sql` as separate dashboard datasets.

## Recommended canvas

### Header KPIs

Dataset: `kpi_market_summary`

Create four KPI widgets:
- Markets analyzed
- High opportunity markets
- Average growth probability
- Average opportunity score

### Forecast model performance

Dataset: `kpi_forecast_performance`

Use a compact table or KPI comparison showing model, WAPE %, MASE, MAE and RMSE. The key business message is that the naive persistence baseline remains Champion because it generalizes better under temporal validation.

### Top opportunities

Dataset: `top_market_opportunities`

Create:
- horizontal bar: `country` vs `opportunity_score`
- executive table with rank, forecast consumption, growth probability, segment, anomaly count and tier

### Demand vs growth matrix

Dataset: `demand_growth_matrix`

Scatter plot:
- X: `forecast_consumption`
- Y: `growth_probability`
- Color: `segment`
- Size: `opportunity_score`
- Tooltip: country, coffee type, tier, anomalies

### Segment distribution

Dataset: `segment_distribution`

Bar chart:
- X: `segment`
- Y: `markets`

### Opportunity tiers

Dataset: `opportunity_tiers`

Bar or donut chart:
- Category: `opportunity_tier`
- Value: `markets`

### Recent anomalies

Dataset: `recent_anomalies`

Table sorted by maximum anomaly score.

### Growth classifier performance

Dataset: `growth_classifier_performance`

Show ROC-AUC, F1, precision, recall and Brier score.

## Filters

Add dashboard filters for:
- country
- coffee_type
- segment
- opportunity_tier

## Publish

Publish the dashboard after validating that all widgets refresh from the Gold tables.
