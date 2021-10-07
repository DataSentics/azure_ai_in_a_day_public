# Databricks notebook source
# MAGIC %md
# MAGIC #MLflow
# MAGIC - An open source platform for the machine learning lifecycle
# MAGIC - 3 components
# MAGIC   - MLflow Tracking &ndash; Record and query experiments: code, data, config, and results.
# MAGIC   - MLflow Projects &ndash; Packaging format for reproducible runs on any platform.
# MAGIC   - MLflow Models &ndash; General format for sending models to diverse deployment tools.
# MAGIC - "mlflow" library

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Tracking
# MAGIC - API and UI for logging parameters, code versions, metrics, and output files
# MAGIC - organization: runs, experiments
# MAGIC - log_metric(), log_param(), log_artifact()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example Problem: Iris Classification
# MAGIC - classify Iris plants into 3 classes using the morphometric variables

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# COMMAND ----------

iris = load_iris()
df_iris = pd.DataFrame(
  np.append(iris["target"][:, np.newaxis], iris["data"], axis=1),
  columns=(["target"] + iris["feature_names"])
)

display(df_iris)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Choosing and Tunning a Classifier
# MAGIC - we can log the model parameters and the loss metrics and (using 5-fold cross-validation) find the best performing model specification

# COMMAND ----------

# create the set of classifiers and the tunning parameters
classifiers = []
# KNeighborsClassifier
classifiers += [
  KNeighborsClassifier(3),
  KNeighborsClassifier(5),
  KNeighborsClassifier(10)
]
# SVC
classifiers += [
  SVC(kernel="rbf", C=0.025, probability=True),
  SVC(kernel="rbf", C=0.05, probability=True),
  SVC(kernel="poly", C=0.025, probability=True),
  SVC(kernel="poly", C=0.05, probability=True),
  SVC(kernel="sigmoid", C=0.025, probability=True),
  SVC(kernel="sigmoid", C=0.05, probability=True)
]
# RandomForestClassifier
classifiers += [
  RandomForestClassifier(n_estimators=10, criterion="gini"),
  RandomForestClassifier(n_estimators=20, criterion="gini"),
  RandomForestClassifier(n_estimators=30, criterion="gini"),
  RandomForestClassifier(n_estimators=10, criterion="entropy"),
  RandomForestClassifier(n_estimators=20, criterion="entropy"),
  RandomForestClassifier(n_estimators=30, criterion="entropy")
]

# COMMAND ----------

np.random.seed(123)
mlflow.set_experiment("/Presentations/MLflowShowcase/ModelSpec")
for clf in classifiers:
  with mlflow.start_run():
    mlflow.log_param('model_name', clf.__class__.__name__)
    # log model parameters to MLflow
    clf_param_dict = clf.__dict__
    for param_name in clf_param_dict:
        mlflow.log_param(param_name, clf_param_dict[param_name])
    mlflow.log_metric('log_loss', -cross_val_score(clf, iris.data, iris.target, cv=5, scoring='neg_log_loss').mean())
mlflow.end_run() 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Learning Curves
# MAGIC - the metrics can be updated continuously troughout the run
# MAGIC - we can use mlflow tracking to further diagnose the model via learning curves

# COMMAND ----------

# select the best classifiers
sel_classifiers = [
  RandomForestClassifier(n_estimators=10),
  KNeighborsClassifier(10),
  SVC(kernel="poly", C=0.05, probability=True)
]

# COMMAND ----------

# split on train and test dataset
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# COMMAND ----------

np.random.seed(456)
N = y_train.size
obs_order = np.arange(N)
np.random.shuffle(obs_order)
mlflow.set_experiment("/Presentations/MLflowShowcase/LearningCurves")
for clf in sel_classifiers:
  model_name = clf.__class__.__name__
  with mlflow.start_run(run_name=model_name):
    mlflow.log_param('model_name', model_name)
    # log model parameters to MLflow
    clf_param_dict = clf.__dict__
    for param_name in clf_param_dict:
        mlflow.log_param(param_name, clf_param_dict[param_name])
    for i in range(10, N+1):
      use_obs = obs_order[:i]
      fit = clf.fit(X_train[use_obs], y_train[use_obs])
      # predictions on the train data
      y_pred = fit.predict_proba(X_train[use_obs])
      mlflow.log_metric('log_loss_train', log_loss(y_true=y_train[use_obs], y_pred=y_pred))
      # predictions on the test data
      y_pred = fit.predict_proba(X_test)
      mlflow.log_metric('log_loss_test', log_loss(y_true=y_test, y_pred=y_pred))

# COMMAND ----------

 # Register some model
mlflow.sklearn.log_model(
    sk_model=sel_classifiers[0],
    artifact_path="sklearn-model",
    registered_model_name="sk-learn-random-forest-reg-model"
)