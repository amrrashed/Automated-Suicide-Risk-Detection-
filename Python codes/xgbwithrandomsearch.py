# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 23:04:32 2022

@author: amr_r
"""
import numpy as np
import pandas as pd  # To read data
#import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import LabelEncoder
import time
# Hide warnings
import warnings
warnings.filterwarnings("ignore")
# Setting up max columns displayed to 100
pd.options.display.max_columns = 100
# Prepare the data.
df = pd.read_csv('D:/new researches/SUICIDE PAPER/Dataset/our DB1 features/features/resnet18featuresoriginal.csv')
df.dropna(inplace=True)
a=df.describe()
X = df.iloc[:,0:512] #512 for resnet18 and 2048 for resnet50
# integer encode
y = df.iloc[:,512] #512 for resnet18 and 2048 for resnet50
y = LabelEncoder().fit_transform(y)

# A parameter grid for XGBoost
params = {
 'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
 'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
 'min_child_weight' : [ 1, 3, 5, 7 ],
 'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ],
 'booster': ['gbtree', 'gblinear'],
 'objective': ['reg:squarederror', 'reg:tweedie'],
}

#reg=GradientBoostingRegressor(random_state=0)
reg = XGBClassifier(nthread=-1)

# run randomized search
n_iter_search = 500
#https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#neg_mean_squared_error
#r2
#neg_mean_absolute_error
#neg_root_mean_squared_error
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=1)
random_search = RandomizedSearchCV(reg, param_distributions=params,
                                   n_iter=n_iter_search, cv=cv,n_jobs=-1,verbose=3, scoring='accuracy')

start = time.time()
result= random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates" " parameter settings." % ((time.time() - start), n_iter_search))

best_regressor = random_search.best_estimator_

print('Best Hyperparameters: %s' % result.best_params_)
print('Best Score: %s' % result.best_score_)




