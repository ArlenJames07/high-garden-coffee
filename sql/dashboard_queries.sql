-- High Garden Coffee — AI/BI dashboard datasets
-- Run these as separate datasets in Databricks AI/BI Dashboard.
-- Source tables are read-only Gold outputs.

-- ============================================================
-- DATASET: kpi_market_summary
-- Recommended visualization: 4 KPI counters
-- ============================================================
SELECT
  COUNT(*) AS markets_analyzed,
  SUM(CASE WHEN opportunity_tier = 'High' THEN 1 ELSE 0 END) AS high_opportunity_markets,
  ROUND(AVG(growth_probability), 3) AS avg_growth_probability,
  ROUND(AVG(opportunity_score), 1) AS avg_opportunity_score
FROM high_garden.gold.market_opportunities;

-- ============================================================
-- DATASET: kpi_forecast_performance
-- Recommended visualization: KPI/table
-- ============================================================
SELECT
  model,
  ROUND(wape * 100, 2) AS wape_pct,
  ROUND(mase, 3) AS mase,
  ROUND(mae, 0) AS mae,
  ROUND(rmse, 0) AS rmse,
  negative_raw_predictions
FROM high_garden.gold.model_metrics
ORDER BY wape ASC;

-- ============================================================
-- DATASET: top_market_opportunities
-- Recommended visualization: horizontal bar + executive table
-- ============================================================
SELECT
  opportunity_rank,
  country,
  coffee_type,
  ROUND(opportunity_score, 1) AS opportunity_score,
  opportunity_tier,
  ROUND(growth_probability, 3) AS growth_probability,
  ROUND(forecast_consumption, 0) AS forecast_consumption,
  segment,
  recent_anomaly_count
FROM high_garden.gold.market_opportunities
ORDER BY opportunity_rank
LIMIT 15;

-- ============================================================
-- DATASET: demand_growth_matrix
-- Recommended visualization: scatter plot
-- X = forecast_consumption
-- Y = growth_probability
-- Color = segment
-- Size = opportunity_score
-- ============================================================
SELECT
  country,
  coffee_type,
  forecast_consumption,
  growth_probability,
  opportunity_score,
  opportunity_tier,
  segment,
  recent_anomaly_count
FROM high_garden.gold.market_opportunities;

-- ============================================================
-- DATASET: segment_distribution
-- Recommended visualization: bar chart
-- ============================================================
SELECT
  segment,
  COUNT(*) AS markets,
  ROUND(AVG(forecast_consumption), 0) AS avg_forecast_consumption,
  ROUND(AVG(growth_probability), 3) AS avg_growth_probability,
  ROUND(AVG(opportunity_score), 1) AS avg_opportunity_score
FROM high_garden.gold.market_opportunities
GROUP BY segment
ORDER BY markets DESC;

-- ============================================================
-- DATASET: opportunity_tiers
-- Recommended visualization: donut or bar chart
-- ============================================================
SELECT
  opportunity_tier,
  COUNT(*) AS markets,
  ROUND(AVG(opportunity_score), 1) AS avg_opportunity_score
FROM high_garden.gold.market_opportunities
GROUP BY opportunity_tier
ORDER BY
  CASE opportunity_tier
    WHEN 'High' THEN 1
    WHEN 'Medium' THEN 2
    ELSE 3
  END;

-- ============================================================
-- DATASET: recent_anomalies
-- Recommended visualization: table
-- ============================================================
SELECT
  country,
  coffee_type,
  recent_anomaly_count,
  ROUND(recent_max_anomaly_score, 4) AS recent_max_anomaly_score,
  ROUND(opportunity_score, 1) AS opportunity_score,
  opportunity_tier
FROM high_garden.gold.market_opportunities
WHERE recent_anomaly_count > 0
ORDER BY recent_max_anomaly_score DESC, recent_anomaly_count DESC;

-- ============================================================
-- DATASET: growth_classifier_performance
-- Recommended visualization: KPI/table
-- ============================================================
SELECT *
FROM high_garden.gold.growth_classifier_metrics;
