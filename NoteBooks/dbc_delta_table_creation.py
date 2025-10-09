# Databricks notebook source
# MAGIC %sql
# MAGIC -- Check current DBR capabilities
# MAGIC SELECT 
# MAGIC   'DBR Version' as info_type,
# MAGIC   current_version().dbr_version as value
# MAGIC UNION ALL
# MAGIC SELECT 
# MAGIC   'Delta Lake Features',
# MAGIC   CASE WHEN current_version().dbr_version LIKE '%delta%' THEN 'Available' ELSE 'Limited' END
# MAGIC UNION ALL
# MAGIC SELECT
# MAGIC   'Photon Status',
# MAGIC   CASE WHEN current_version().dbr_version LIKE '%photon%' THEN 'Enabled' ELSE 'Disabled' END;

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG demo_catalog

# COMMAND ----------

# MAGIC %sql
# MAGIC USE SCHEMA raw

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Test advanced SQL features available in newer DBR (ONLY Run at the start of once need shcema changes)
# MAGIC CREATE OR REPLACE TABLE dbr_feature_test (
# MAGIC   id BIGINT,
# MAGIC   category STRING,
# MAGIC   value DECIMAL(10,2),
# MAGIC   created_date DATE
# MAGIC ) USING DELTA;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Insert test data with DBR optimizations (ONLY Execute if you need new data generation)
# MAGIC INSERT INTO dbr_feature_test
# MAGIC SELECT 
# MAGIC   id,
# MAGIC   CASE WHEN id % 5 = 0 THEN 'Premium'
# MAGIC        WHEN id % 3 = 0 THEN 'Standard' 
# MAGIC        ELSE 'Basic' END as category,
# MAGIC   CAST(RANDOM() * 1000 AS DECIMAL(10,2)) as value,
# MAGIC   DATE_ADD(CURRENT_DATE(), CAST(-(id % 365) AS INT)) as created_date
# MAGIC FROM RANGE(100000);
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC
# MAGIC -- Test SQL features that benefit from newer DBR
# MAGIC -- Window functions with optimization
# MAGIC SELECT 
# MAGIC   category,
# MAGIC   created_date,
# MAGIC   value,
# MAGIC   AVG(value) OVER (
# MAGIC     PARTITION BY category 
# MAGIC     ORDER BY created_date 
# MAGIC     ROWS BETWEEN 7 PRECEDING AND CURRENT ROW
# MAGIC   ) as moving_avg_7day,
# MAGIC   RANK() OVER (
# MAGIC     PARTITION BY category 
# MAGIC     ORDER BY value DESC
# MAGIC   ) as value_rank
# MAGIC FROM dbr_feature_test
# MAGIC WHERE created_date >= CURRENT_DATE() - INTERVAL 30 DAYS
# MAGIC ORDER BY category, created_date;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Test adaptive query execution benefits
# MAGIC SELECT 
# MAGIC   t1.category,
# MAGIC   COUNT(*) as record_count,
# MAGIC   AVG(t1.value) as avg_value,
# MAGIC   SUM(t2.value) as related_sum
# MAGIC FROM dbr_feature_test t1
# MAGIC JOIN (
# MAGIC   SELECT category, value 
# MAGIC   FROM dbr_feature_test 
# MAGIC   WHERE value > 500
# MAGIC ) t2 ON t1.category = t2.category
# MAGIC GROUP BY t1.category
# MAGIC HAVING COUNT(*) > 1000
# MAGIC ORDER BY avg_value DESC;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM dbr_feature_test 
# MAGIC WHERE category = 'Premium' 
# MAGIC AND created_date >= CURRENT_DATE() - INTERVAL 30 DAYS;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Test Delta Lake time travel (DBR enhancement)
# MAGIC -- Update records to create new version
# MAGIC UPDATE dbr_feature_test 
# MAGIC SET value = value * 1.1 
# MAGIC WHERE category = 'Premium' AND created_date >= CURRENT_DATE() - INTERVAL 30 DAYS;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Query historical data
# MAGIC SELECT 
# MAGIC   'Current' as version_type,
# MAGIC   category,
# MAGIC   COUNT(*) as record_count,
# MAGIC   AVG(value) as avg_value
# MAGIC FROM dbr_feature_test
# MAGIC WHERE category = 'Premium'
# MAGIC GROUP BY category
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT 
# MAGIC   'Version 0' as version_type,
# MAGIC   category,
# MAGIC   COUNT(*) as record_count, 
# MAGIC   AVG(value) as avg_value
# MAGIC FROM dbr_feature_test VERSION AS OF 1
# MAGIC WHERE category = 'Premium' 
# MAGIC GROUP BY category;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Performance optimization testing
# MAGIC OPTIMIZE dbr_feature_test ZORDER BY (category, created_date);
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Cleanup
# MAGIC DROP TABLE dbr_feature_test;

# COMMAND ----------

