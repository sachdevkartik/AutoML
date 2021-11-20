import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline# Import estimator and preprocessors you need
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


df_train = pd.read_csv("data_train.csv")

df_test = pd.read_csv("data_test.csv")

sub = pd.read_csv("sample_submission.csv",  index_col=0)

df_train.dropna(inplace=True)
categ = df_train.select_dtypes(include = "object").columns

for feat in categ:
    df_train[feat] = df_train[feat].astype("category") #le.fit_transform(df_train[feat].astype(str))

X = df_train[df_train.columns[df_train.columns!="class"]]
y = df_train[df_train.columns[df_train.columns=="class"]]

train_data = X.copy()
train_data["class"] = y

from autogluon.tabular import TabularPredictor

label = 'class'  # name of target variable to predict in this competition
eval_metric = 'accuracy'  # Optional: specify that competition evaluation metric is AUC

predictor = TabularPredictor(label=label, eval_metric=eval_metric, verbosity=3).fit(
    train_data, presets='best_quality', time_limit=7200
)

results = predictor.fit_summary()

for feat in categ[1:]:
    df_test[feat] = df_test[feat].astype("category") #le.fit_transform(df_train[feat].astype(str))

preds = predictor.predict(df_test.iloc[:,:-1])

preds.index += 1

sub["Predicted"] = preds
sub.to_csv("autogluon.csv")
