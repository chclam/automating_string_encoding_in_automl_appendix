import os
import numpy as np
import fasttext
import pandas as pd
from sklearn.metrics import accuracy_score 
from sklearn.base import ClassifierMixin, BaseEstimator
from time import time
from sklearn.exceptions import NotFittedError
from random import random
from sklearn.model_selection import train_test_split 
import multiprocessing
import uuid

fasttext.FastText.eprint = lambda x: None # surpress the load_model warning

class TabularFastText(BaseEstimator, ClassifierMixin):
  def __init__(self, lr=0.1, epoch=5, wordNgrams=1, minn=0, maxn=0, pretrainedVectors="", dim=100, autotune=False, thread=None, autotuneDuration=60*5, char_level=False):
    self._estimator_type = "classifier"
    self.classes_ = None
    self.model_filename = None
    self.lr = lr
    self.epoch = epoch
    self.wordNgrams = wordNgrams
    self.minn = minn
    self.maxn = maxn
    self.pretrainedVectors=pretrainedVectors # by default dim=100 if pretrainedVectors="" 
    self.dim = dim
    self.autotune = autotune
    self.autotuneDuration = autotuneDuration
    self.thread = multiprocessing.cpu_count() - 1 if thread is None else thread 
    self.char_level = char_level
    
  def fit(self, X, y):
    '''
    TODO: Label encode y
    '''
    if not os.path.isdir("cache"):
      try:
        os.mkdir("cache")
      except Exception as e:
        print(e)
    data_fn = f"cache/test_data{str(uuid.uuid1())}.txt"
    val_data_fn = data_fn.replace(".txt", "_val.txt")

    if self.classes_ is None:
      self.classes_ = np.unique(y)
    pd.set_option('display.max_colwidth', None) # do this so that .to_string() actually converts all data to string

    if self.autotune:
      X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y)
      self.preprocess(X_train, y_train, data_fn=data_fn)
      self.preprocess(X_val, y_val, data_fn=val_data_fn)
      model = fasttext.train_supervised(data_fn, pretrainedVectors=self.pretrainedVectors, autotuneValidationFile=val_data_fn if self.autotune else "", thread=self.thread, autotuneDuration=self.autotuneDuration)
    else:
      self.preprocess(X, y, data_fn=data_fn)
      model = fasttext.train_supervised(data_fn, lr=self.lr, epoch=self.epoch, wordNgrams=self.wordNgrams, minn=self.minn, maxn=self.maxn, pretrainedVectors=self.pretrainedVectors, dim=self.dim)

    self.model_filename = f"cache/ft_model_{str(uuid.uuid1())}.bin"
    # save and load the model due to issues with multiprocessing when passing on the fit model.
    model.save_model(self.model_filename)
    return self

  def predict(self, X, ret_proba=False):
    if self.model_filename is None:
      raise NotFittedError("Model is not fitted yet. Please fit the model first.")
    pd.set_option('display.max_colwidth', None)
    data = self.preprocess(X)
    data = data.split("\n") # split the data into an array of rows
    model = fasttext.load_model(self.model_filename)
    classes_pred, probs_pred = model.predict(data, k=len(self.classes_) if ret_proba else 1)
    if not ret_proba:
      return np.array([int(x[0].replace("__label__", "")) for x in classes_pred]) # flatten list and get rid of "__label__" prefix
    else:
      ret = []
      for row_classes, row_probs in zip(classes_pred, probs_pred):
        sorted_by_classes = sorted(list(zip(row_classes, row_probs)), key=lambda x: x[0])
        probas = [lbl[1] for lbl in sorted_by_classes]
        # make sure probabilities nicely sum up to 1 by subtracting the excess from the highest class probability.
        # highest because subtracting from lowest can result in negative numbers.
        probas[np.argmax(probas)] -= sum(probas) - 1.
        ret.append(probas)
      return np.array(ret)

  def predict_proba(self, X):
    return self.predict(X, ret_proba=True)
  
  def score(self, X, y_true):
    y_pred = self.predict(X)
    return accuracy_score(y_true, y_pred)

  def _get_n_grams(self, x, char_level=True, sep=" "):
    sliced_x = x.split(sep)
    ret = []
    for word in sliced_x:
      if "__label__" in word or any(x.isdigit() for x in word):
        ret.append(word)
      else:
        for i in range(0, len(word)-2, 1):
          ret.append(word[i:i+3])
    ret = " ".join(ret)
    return ret

  def preprocess(self, X, y=None, del_spec_chars=True, data_fn=f"cache/test_data{str(uuid.uuid1())}.txt") -> str:
    X = pd.DataFrame(X, columns=X.columns if isinstance(X, pd.DataFrame) else None).reset_index(drop=True)
    data = X.copy()
    obj_cols = [col for col in X.columns if X[col].dtype == "O" or isinstance(X[col].dtype, pd.core.dtypes.dtypes.CategoricalDtype)]
    for obj_col in obj_cols:
      data[obj_col] = data[obj_col].str.replace("\n", "")
    data = data.astype(str)
    data = data.fillna(" ")
    if y is not None:
      # formatting y to fit fasttext expected format
      y = pd.DataFrame(y, columns=[0]).reset_index(drop=True)
      y.name = None
      y[1] = y[0]
      y[0] = "__label__"
      y = y.astype(str).apply(lambda r: "".join(v for v in r.values), axis=1)
      data = pd.concat([data, y], axis=1)
      # Sloppy hack for appending "__label__" faster to all values  
    data.to_csv(data_fn, header=None, index=None, sep=' ', mode='w')
    # call bash script for preprocessing
    sed_cmd = "%s %s" % (os.path.join('tabularfasttext','preprocess_X.sh'), data_fn)
    os.system(sed_cmd)
    with open(data_fn, "r") as f:
      ret = f.read()
    if ret[-1] == "\n" and len(ret) > 0:
      ret = ret[:(len(ret) - 1)] # remove trailing newline
    if self.char_level:
      ret = ret.split("\n")
      ret = [self._get_n_grams(x) for x in ret]
      ret = "\n".join(ret)
      with open(data_fn, "w") as f:
        f.write(ret)
    return ret 

