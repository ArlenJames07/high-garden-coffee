# Executive AI/BI Dashboard

## Goal

Create a concise executive dashboard from governed Gold tables. The dashboard should explain market size, growth likelihood, segmentation, anomalies, and the final decision-support score without implying that the opportunity score is a learned probability of commercial success.

## Build in Databricks

1. Open **Dashboards** and create a new AI/BI dashboard named `High Garden Coffee — Market Intelligence`.
2. Select a Serverless SQL Warehouse.
3. Open the **Data** tab.
4. Create datasets by copying each query from `sql/dashboard_queries.sql` as a separate SQL dataset.
5. Build one page named `Executive Overview`.

## Recommended layout

### Row 1 — KPI cards

Use dataset `kpi_market_summary`:

- Markets analyzed
- High-opportunity markets
- Average growth probability
- Average opportunity score

Use `kpi_forecast_performance` to display the Champion forecast WAPE and MASE. The current validated temporal-backtest result is approximately 2.25% WAPE for the naive/persistence Champion.

### Row 2 — Market prioritization

**Top Market Opportunities**

Dataset: `top_market_opportunities`

Visualization: horizontal bar chart

- Y: `country`
- X: `opportunity_score`
- Group/color: `opportunity_tier`
- Sort: descending `opportunity_score`

**Demand vs Growth Matrix**

Dataset: `demand_growth_matrix`

Visualization: scatter plot

- X: `forecast_consumption`
- Y: `growth_probability`
- Color: `segment`
- Size: `opportunity_score`
- Tooltip: country, coffee_type, tier, anomaly count

### Row 3 — Market structure

**Market Segments**

Dataset: `segment_distribution`

Visualization: bar chart

- X: `segment`
- Y: `markets`

**Opportunity Tiers**

Dataset: `opportunity_tiers`

Visualization: bar or donut chart

- Category: `opportunity_tier`
- Value: `markets`

### Row 4 — Risk and model evidence

**Recent Anomalies**

Dataset: `recent_anomalies`

Visualization: table

Display country, coffee type, anomaly count, maximum anomaly score, opportunity score, and tier.

**Model Performance**

Dataset: `kpi_forecast_performance`

Visualization: table

This is important because the simpler persistence model outperformed CatBoost in temporal backtesting. The dashboard should make model selection evidence visible rather than presenting complexity as an advantage.

## Filters

Add dashboard filters for:

- `coffee_type`
- `segment`
- `opportunity_tier`

Avoid adding too many filters because there are only 55 markets.

## Business interpretation

Use these definitions consistently:

- **Forecast consumption**: one-step forecast of domestic coffee consumption in source units.
- **Growth probability**: probability from the growth-direction classifier that next-period consumption is greater than the previous period.
- **Segment**: unsupervised market profile from K-Means.
- **Anomaly**: unusual historical market-year pattern detected by Isolation Forest; it is not a causal diagnosis.
- **Opportunity score**: configurable decision-support score combining demand, growth, stability, and anomaly evidence. It is not a supervised prediction of market success or profitability.

## Final presentation view

For the technical challenge, keep the dashboard to one page. The reviewer should be able to understand the business result in under two minutes.
