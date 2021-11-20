import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from autoPyTorch import AutoNetClassification
import json

from utils import *

df_train = pd.read_csv("data_train.csv")

df_test = pd.read_csv("data_test.csv")

sub = pd.read_csv("sample_submission.csv",  index_col=0)

df_train.dropna(inplace=True)
categ = df_train.select_dtypes(include = "object").columns

le = LabelEncoder()
for feat in categ:
    if feat == "class":
        pass
    else:
        df_train[feat] = le.fit_transform(df_train[feat].astype(str))

le_class = LabelEncoder()
df_train["class"] = le_class.fit_transform(df_train["class"].astype(str))

X = df_train[df_train.columns[df_train.columns!="class"]]
y = df_train[df_train.columns[df_train.columns=="class"]]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

train_data = X.copy()
train_data["class"] = y

label = 'class'  
eval_metric = 'accuracy' 

# Seed
seed = 42
seed_everything(seed)

# Get autonet config
min_budget=30
max_budget=90
max_runtime = 500 # 2*60*60 # 2*60*60 if args.test=="false" else 5*60
autonet_config = get_autonet_config_lcbench(min_budget=min_budget,
                                            max_budget=max_budget, 
                                            max_runtime=max_runtime,
                                            num_workers=12, 
                                            logdir="logs", 
                                            seed=seed)


autonet_config["algorithm"] = "bohb"
# Categoricals

# autonet_config["categorical_features"] = ["V25","V83",  ]
autonet_config["embeddings"] = ['none', 'learned']

# running Auto-PyTorch
autonet = AutoNetClassification(config_preset="medium_cs", **autonet_config)

# running Auto-PyTorch using default config
# autonet = AutoNetClassification("medium_cs",  # config preset
#                                     log_level='debug',
#                                     max_runtime=500,
#                                     min_budget=30,
#                                     max_budget=90,
#                                     cuda=True,
#                                     use_pynisher=False)

current_configuration = autonet.get_current_autonet_config()
hyperparameter_search_space = autonet.get_hyperparameter_search_space()

results_fit = autonet.fit(X_train, y_train, **autonet.get_current_autonet_config())

# Save fit results as json
with open("logs/results_config.json", "w") as file:
    json.dump(results_fit, file)

for feat in categ[1:]:
    df_test[feat] = le.fit_transform(df_test[feat].astype(str))

score = autonet.score(X_test=X_test, Y_test=y_test)
pred = autonet.predict(X=df_test.iloc[:,:-1])

pred_val = autonet.predict(X=X_test)
acc_val = sklearn.metrics.accuracy_score(y_test, pred_val)

# print(pred)
pred_np_ravel = np.ravel(pred) 
print(pred_np_ravel)

pred = le_class.inverse_transform(pred.astype(int))

sub["Predicted"] = pred
sub.to_csv("autopytorch_mod2.csv")

print("Model prediction:", pred[0:100])
print("Accuracy score", score)
print("Accuracy validation score", acc_val)