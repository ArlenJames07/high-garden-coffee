-- High Garden Coffee — grounded query examples for Databricks Genie
-- Use these against governed Gold tables. The LLM explains; it does not generate forecasts.

-- Top opportunities
SELECT
  opportunity_rank,
  country,
  coffee_type,
  forecast_consumption,
  growth_probability,
  segment,
  recent_anomaly_count,
  opportunity_score,
  opportunity_tier
FROM high_garden.gold.market_opportunities
ORDER BY opportunity_rank
LIMIT 10;

-- Explain one market
SELECT
  country,
  coffee_type,
  forecast_consumption,
  growth_probability,
  cagr_recent,
  latest_yoy_growth,
  volatility_cv,
  segment,
  recent_anomaly_count,
  recent_max_anomaly_score,
  demand_score,
  growth_score,
  stability_score,
  anomaly_safety_score,
  opportunity_score,
  opportunity_tier
FROM high_garden.gold.market_opportunities
WHERE country = :country
ORDER BY opportunity_rank;

-- High growth probability, ordered by demand
SELECT
  country,
  coffee_type,
  forecast_consumption,
  growth_probability,
  opportunity_score,
  opportunity_tier
FROM high_garden.gold.market_opportunities
WHERE growth_probability >= 0.60
ORDER BY forecast_consumption DESC;

-- Recent anomalies
SELECT
  country,
  coffee_type,
  recent_anomaly_count,
  recent_max_anomaly_score,
  opportunity_score,
  opportunity_tier
FROM high_garden.gold.market_opportunities
WHERE recent_anomaly_count > 0
ORDER BY recent_max_anomaly_score DESC;

-- Forecast model evidence
SELECT
  model,
  mae,
  rmse,
  wape,
  mase,
  negative_raw_predictions
FROM high_garden.gold.model_metrics
ORDER BY wape ASC;

-- Segment profiles
SELECT
  segment,
  COUNT(*) AS markets,
  AVG(forecast_consumption) AS avg_forecast_consumption,
  AVG(growth_probability) AS avg_growth_probability,
  AVG(volatility_cv) AS avg_volatility,
  AVG(opportunity_score) AS avg_opportunity_score
FROM high_garden.gold.market_opportunities
GROUP BY segment
ORDER BY markets DESC;
