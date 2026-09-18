-- High Garden Coffee — lightweight production monitoring

-- Data freshness / row counts
SELECT 'bronze.coffee_raw' AS object_name, COUNT(*) AS row_count
FROM high_garden.bronze.coffee_raw
UNION ALL
SELECT 'silver.coffee_consumption', COUNT(*)
FROM high_garden.silver.coffee_consumption
UNION ALL
SELECT 'gold.market_metrics', COUNT(*)
FROM high_garden.gold.market_metrics
UNION ALL
SELECT 'gold.market_opportunities', COUNT(*)
FROM high_garden.gold.market_opportunities;

-- Core data-quality checks
SELECT
  COUNT(*) AS rows,
  COUNT(DISTINCT country) AS countries,
  SUM(CASE WHEN domestic_consumption IS NULL THEN 1 ELSE 0 END) AS null_consumption,
  SUM(CASE WHEN domestic_consumption < 0 THEN 1 ELSE 0 END) AS negative_consumption,
  SUM(CASE WHEN domestic_consumption = 0 THEN 1 ELSE 0 END) AS zero_consumption
FROM high_garden.silver.coffee_consumption;

-- Opportunity output integrity
SELECT
  COUNT(*) AS rows,
  COUNT(DISTINCT concat_ws('||', country, coffee_type)) AS unique_markets,
  SUM(CASE WHEN opportunity_score IS NULL THEN 1 ELSE 0 END) AS null_scores,
  SUM(CASE WHEN opportunity_rank IS NULL THEN 1 ELSE 0 END) AS null_ranks,
  MIN(opportunity_score) AS min_score,
  MAX(opportunity_score) AS max_score
FROM high_garden.gold.market_opportunities;

-- Forecast monitoring from stored backtests
SELECT
  model,
  ROUND(AVG(wape) * 100, 2) AS avg_wape_pct,
  ROUND(AVG(mase), 3) AS avg_mase
FROM high_garden.gold.model_metrics
GROUP BY model
ORDER BY avg_wape_pct;

-- Growth-classifier metrics
SELECT *
FROM high_garden.gold.growth_classifier_metrics;

-- Anomaly-rate monitoring
SELECT
  COUNT(*) AS records,
  SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) AS anomalies,
  ROUND(AVG(CASE WHEN is_anomaly = 1 THEN 1.0 ELSE 0.0 END) * 100, 2) AS anomaly_rate_pct
FROM high_garden.gold.market_anomalies;

-- Segment distribution drift snapshot
SELECT
  segment,
  COUNT(*) AS markets,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS share_pct
FROM high_garden.gold.market_clusters
GROUP BY segment
ORDER BY markets DESC;
