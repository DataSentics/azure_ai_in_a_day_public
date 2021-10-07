# Databricks notebook source
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read data as a pyspark DataFrame
# MAGIC 
# MAGIC Note: pyspark DataFrame is only a plan to be executed

# COMMAND ----------

raw_data_path = 's3a://ai-in-a-day/raw/immo_data_cl4.tsv'
parsed_data_path = 's3a://ai-in-a-day/parsed/immo_data'

# COMMAND ----------

df = spark.read.csv(raw_data_path, header=True, inferSchema=True, sep='\t', nullValue='NA', multiLine=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview data

# COMMAND ----------

display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# DBTITLE 1,Fix some types..
# Now balcony got recognized correctly
# df = df.withColumn('balcony', F.col('balcony').cast('boolean'))

# COMMAND ----------

display(df)

# COMMAND ----------

display(
  df
  .filter(F.col('totalRent') < 10000)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write as parquet

# COMMAND ----------

(
  df
  .repartition(8)
  .write
  .parquet(parsed_data_path, mode='overwrite')
)