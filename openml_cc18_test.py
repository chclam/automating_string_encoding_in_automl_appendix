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
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

import config

logging.basicConfig(level=logging.INFO)
openml.config.apikey = config.OPENML_APIKEY  # set the OpenML Api Key

def get_openml_cc18_benchmark():
  '''
  The OpenML servers are slow.
  Retrieve the benchmark from disk or retrieve from OpenML itself and save to disk otherwise.
  '''
  cache_openml_cc18_filename = "cache_openml_cc18.pkl"
  cache_path = os.path.join(config.CACHE_DIR, cache_openml_cc18_filename)

  if os.path.isfile(cache_path):
    with open(cache_path, "rb") as f:
      benchmark_suite = pickle.load(f) 
  else:
    benchmark_suite = openml.study.get_suite('OpenML-CC18')
    if not os.path.isdir(config.CACHE_DIR):
      os.mkdir(config.CACHE_DIR)
    with open(cache_path, "wb") as f:
      pickle.dump(benchmark_suite, f)
  return benchmark_suite

if __name__ == "__main__":
  benchmark_suite = get_openml_cc18_benchmark()

  if not os.path.isdir(config.CV_DIR):
    logging.error("No folder for CV folds found: {}".format(config.CV_DIR))
    exit(1)

  for i, t_id in zip(trange(len(benchmark_suite.tasks)), benchmark_suite.tasks):
    try:
      logging.info("Loading dataset from OpenML.")
      task = openml.tasks.get_task(t_id)
      dataset = task.get_dataset()
      X, y, _, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
      )

    except Exception as e:
      logging.error("OpenML error", e)

    if len(np.unique(y)) == 2:
      scorer_name = "roc_auc"
      scorer = roc_auc_score
    else:
      scorer_name = "neg_log_loss"
      scorer = log_loss

    logging.info("Loading CV folds.")
    with open(f"{config.CV_DIR}/{dataset.id}.pkl", "rb") as f:
      cv = pickle.load(f) 
 
    # TODO: use cross validation
    for fold_nr, (train_idx, test_idx) in enumerate(cv):
      # Set result data
      fold_res            = config.RESULT_FORMAT
      fold_res["data_id"] = str(dataset.id)
      fold_res["name"]    = dataset.name
      fold_res["metric"]  = scorer_name
      # Write experimental setup to result data
      pipeline_config_names = {setting: value.__name__ for setting, value in config.PIPELINE_CONFIG.items()}
      fold_res.update(pipeline_config_names)

      try:
        y       = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_test  = X.iloc[test_idx].reset_index(drop=True)
        y_test  = y.iloc[test_idx].reset_index(drop=True)
        
        # prepare pipeline
        pipe = make_pipeline(
          config.PIPELINE_CONFIG["string_encoder"](),
          config.PIPELINE_CONFIG["imputer"](),
          config.PIPELINE_CONFIG["learning_algorithm"](),
        )

        logging.info(f"Start training fold number: {fold_nr}")
        train_start_time = time.time()
        pipe.fit(X_train, y_train)
        fold_res["train_time"] = time.time() - train_start_time

        logging.info(f"Start prediction fold number: {fold_nr}")
        predict_start_time = time.time()
        y_proba = pipe.predict_proba(X_test)
        fold_res["predict_time"] = time.time() - predict_start_time

        if scorer_name == "roc_auc":
          fold_res["score"] = scorer(y_test, y_proba[:,1], labels=np.unique(y))
        else:
          fold_res["score"] = scorer(y_test, y_proba, labels=np.unique(y))

      except Exception as e:
        logging.error(f"Error with fold: {e}")

      finally:
        results_file_exists = os.path.isfile(config.OUTPUT_FILENAME)
        with open(config.OUTPUT_FILENAME, "a+") as f:
          w = csv.DictWriter(f, fold_res.keys())
          if not results_file_exists:
            w.writeheader()
          w.writerow(fold_res)

    exit(0)
        
