#!/usr/bin/env python3
import os
import csv
import time
import pickle
import openml
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import trange
from sklearn.metrics import roc_auc_score, log_loss, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate

import config

logging.basicConfig(level=logging.INFO)
openml.config.apikey = config.OPENML_APIKEY  # set the OpenML Api Key

def evaluate_pipeline(X, y, cv: list, pipeline_config: dict, dataset_id: int, dataset_name: str) -> None:
  if len(np.unique(y)) == 2:
    scorer_name = "roc_auc"
    score_func = roc_auc_score
  else:
    scorer_name = "neg_log_loss"
    score_func = log_loss

  # Prepare formatting of the results for every cv-fold
  fold_res = config.RESULT_FORMAT.copy()
  fold_res.update({
    "data_id": dataset_id,
    "name": dataset_name,
    "metric": scorer_name
  })
  pipeline_config_names = {setting: str(value) for setting, value in pipeline_config.items()}
  fold_res.update(pipeline_config_names)
  results = [(fold_res.copy() | {"fold_nr": cv_idx}) for cv_idx in range(len(cv))] 

  try:
    y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
    # set string features to "category" to avoid issues in libraries
    str_features = [col for col in X if isinstance(X[col].dtype, pd.core.dtypes.dtypes.CategoricalDtype) or X[col].dtype == "O"]
    X[str_features] = X[str_features].astype("category")
   
    ct = ColumnTransformer(transformers=[("encoder", pipeline_config["string_encoder"], str_features)], remainder="passthrough")
    pipe = make_pipeline(ct, pipeline_config["imputer"], pipeline_config["learning_algorithm"])
    scorer = make_scorer(score_func, greater_is_better=(scorer_name=="roc_auc"), needs_proba=True, labels=y.unique())

    logging.info("Starting cross validation.")
    cv_results = cross_validate(estimator=pipe, X=X, y=y, scoring=scorer, cv=cv, n_jobs=config.CV_N_JOBS)

    for fold_idx in range(len(results)):
      results[fold_idx].update({
        "fit_time": cv_results.get("fit_time")[fold_idx],
        "predict_time": cv_results.get("score_time")[fold_idx],
        "cv_score": cv_results.get("test_score")[fold_idx],
      })
  except Exception as e:
    logging.exception(f"Error with fold: {e}")
  finally:
    results_file_exists = os.path.isfile(config.OUTPUT_FILENAME)
    with open(config.OUTPUT_FILENAME, "a+") as f:
      w = csv.DictWriter(f, results[0].keys())
      if not results_file_exists:
        w.writeheader()
      w.writerows(results)

if __name__ == "__main__":
  if not os.path.isdir(config.CV_DIR):
    logging.error(f"No folder for CV folds found: {config.CV_DIR}")
    exit(1)

  for _, d_id in zip(trange(len(config.INPUT_DATASET_IDS)), config.INPUT_DATASET_IDS):
    try:
      logging.info("Loading dataset from OpenML.")
      dataset = openml.datasets.get_dataset(d_id)
      X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    except Exception as e:
      logging.exception("OpenML error", e)

    logging.info("Loading CV folds.")
    with open(os.path.join(config.CV_DIR, f"{dataset.id}.pkl"), "rb") as f:
      cv = pickle.load(f) 

    if isinstance(config.PIPELINE_CONFIGS, dict):
      evaluate_pipeline(X, y, cv, config.PIPELINE_CONFIGS, dataset.id, dataset.name)
    elif isinstance(config.PIPELINE_CONFIGS, list):
      if not all(config.PIPELINE_CONFIGS[0].keys() == pipeline_config.keys() for pipeline_config in config.PIPELINE_CONFIGS):
        raise ValueError("The pipeline configurations do not match up. Please check your configurations.")
      for pipeline_config in config.PIPELINE_CONFIGS:
        evaluate_pipeline(X, y, cv, pipeline_config, dataset.id, dataset.name)
    else:
      raise ValueError("PIPELINE_CONFIG is not of type dict or a list.")
 
