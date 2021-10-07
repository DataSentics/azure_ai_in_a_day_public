# Databricks notebook source
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression, RandomForestRegressor
#from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from datetime import date

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read data as a pyspark DataFrame
# MAGIC 
# MAGIC Note: pyspark DataFrame is only a plan to be executed

# COMMAND ----------

parsed_data_path = 's3a://ai-in-a-day/parsed/immo_data'

# COMMAND ----------

df = spark.read.parquet(parsed_data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview data

# COMMAND ----------

df.count()

# COMMAND ----------

display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df)

# COMMAND ----------

display(df.filter(F.col('totalRent') < 10000))

# COMMAND ----------

display(df)

# COMMAND ----------

display(df)

# COMMAND ----------

display(df.summary())

# COMMAND ----------

# DBTITLE 1,Example of SQL
df_rent_per_region = (
    df
    .filter(F.col('totalRent') < 10000)
    .groupBy('regio1')
    .agg(
        F.count('*').alias('n_rows'),
        F.avg('totalRent').alias('totalRent_avg'),
        F.min('totalRent').alias('totalRent_min'),
        F.max('totalRent').alias('totalRent_max'),
        F.stddev('totalRent').alias('totalRent_sd'),
        F.min('yearConstructed').alias('oldest_building')
    )
    .orderBy(F.desc('totalRent_avg'))
)

# COMMAND ----------

df_rent_per_region.explain()

# COMMAND ----------

display(df_rent_per_region)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature engineering and selection

# COMMAND ----------

# MAGIC %md
# MAGIC We skip some steps that we should normally do
# MAGIC * Filter / repair broken data
# MAGIC * Possibly impute missing data
# MAGIC * Transform variables: e.g. log-transform totalRent
# MAGIC * Exclude outliers: e.g. mean(log_totalRent) +- 3 x SD
# MAGIC * Understand data, engineer more features..

# COMMAND ----------

# DBTITLE 1,New feature: years_old
this_year = date.today().year

df_feat = (
    df
    .withColumn('years_old', F.lit(this_year) - F.col('yearConstructed'))
)

# COMMAND ----------

display(df_feat)

# COMMAND ----------

# DBTITLE 1,Select columns
target = 'totalRent'
numeric_features = ['livingSpace', 'years_old']
boolean_features = ['hasKitchen', 'balcony']
categorical_features = ['regio1', 'date']

# COMMAND ----------

# DBTITLE 1,Remove nulls, strange values..
df_clean = (
    df_feat
    .filter((F.col('totalRent') > 100) & (F.col('totalRent') < 5000))  # Remove "strange", nontypical values of rent
    .filter(F.col('years_old').between(0, 200))
    .filter(F.col('livingSpace').between(0, 300))
    .select(target, *numeric_features, *boolean_features, *categorical_features)
    .na.drop()
)

df_clean.cache()
df_clean.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare model pipeline

# COMMAND ----------

# Convert strings to numbers
string_indexers = [
    StringIndexer(inputCol=col, outputCol=f'{col}_si')
    for col in categorical_features
]
# Dummy-encode categorical columns encoded as numbers
ohe_estimator = OneHotEncoder(
    inputCols=[f'{col}_si' for col in categorical_features],
    outputCols=[f'{col}_ohe' for col in categorical_features],
    dropLast=True
)
# Combine all feature columns into one feature vector
vector_assembler = VectorAssembler(
    inputCols=[f'{col}_ohe' for col in categorical_features] + numeric_features + boolean_features,
    outputCol='features'
)

# COMMAND ----------

# Prepare Linear regression model
model_lr = LinearRegression(featuresCol='features', labelCol=target)

# Prepare lognormal GLM
model_glm = GeneralizedLinearRegression(featuresCol='features', labelCol=target, family='gaussian', link='log')

# Prepare Random forest model
model_rf = RandomForestRegressor(featuresCol='features', labelCol=target, numTrees=100)

# COMMAND ----------

# Put all these steps into one processing pipeline
pipeline_lr = Pipeline(stages=string_indexers + [ohe_estimator, vector_assembler, model_lr])
pipeline_glm = Pipeline(stages=string_indexers + [ohe_estimator, vector_assembler, model_glm])
pipeline_rf = Pipeline(stages=string_indexers + [ohe_estimator, vector_assembler, model_rf])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit models
# MAGIC Here we duplicate some of the preprocessing steps, in real situations we could do it a bit differently..

# COMMAND ----------

# DBTITLE 0,Fit LR
# Linear regression
pipeline_lr_fitted = pipeline_lr.fit(df_clean)

# COMMAND ----------

# GLM
pipeline_glm_fitted = pipeline_glm.fit(df_clean)

# COMMAND ----------

# Random forest
pipeline_rf_fitted = pipeline_rf.fit(df_clean)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inspect results

# COMMAND ----------

model_lr_fitted = pipeline_lr_fitted.stages[-1]
model_lr_fitted.summary.r2

# COMMAND ----------

model_lr_fitted.coefficients

# COMMAND ----------

model_rf_fitted = pipeline_rf_fitted.stages[-1]
model_rf_fitted.featureImportances

# COMMAND ----------

# MAGIC %md
# MAGIC #### Prediction for training data
# MAGIC 
# MAGIC **Note: in practice we would need to use test set**

# COMMAND ----------

df_predictions_lr = pipeline_lr_fitted.transform(df_clean)
df_predictions_glm = pipeline_glm_fitted.transform(df_clean)
df_predictions_rf = pipeline_rf_fitted.transform(df_clean)

# COMMAND ----------

display(df_predictions_lr)

# COMMAND ----------

display(df_predictions_lr)

# COMMAND ----------

display(df_predictions_glm)

# COMMAND ----------

display(df_predictions_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate accuracy
# MAGIC Again, we would need test set

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol=target, predictionCol='prediction', metricName='rmse')

eval_lr = evaluator.evaluate(df_predictions_lr)
print(eval_lr)
eval_glm = evaluator.evaluate(df_predictions_glm)
print(eval_glm)
eval_rf = evaluator.evaluate(df_predictions_rf)
print(eval_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC Random forest is the winner here, but could be just overfitting, because we didnt do proper cross-validation..

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export selected model

# COMMAND ----------

# MAGIC %md
# MAGIC * We can save the model object including preprocesing pipeline and its metadata (accuracy, hyperparameters..) into MLFlow repository
# MAGIC * Not shown here, productionalizatoin

# COMMAND ----------

mlflow.log_metric("accuracy", 0.9)

# COMMAND ----------

mlflow.mleap.log_model(spark_model=pipeline_lr_fitted, sample_input=df_clean, artifact_path="model")

# COMMAND ----------

mlflow.spark.log_model(pipeline_lr_fitted, "pipeline_lr_fitted")