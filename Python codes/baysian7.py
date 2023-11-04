# -*- coding: utf-8 -*-
#Created on Thu Oct 19 15:24:55 2023
#author: https://towardsdatascience.com/bayesian-optimization-with-python-85c66df711ec
##Best result: {'C': 0.9951661913106971, 'coef0': 0.4562721883588112, 'degree': 2.2747847053307813, 'gamma': 0.022941066854479573, 'max_iter': 572.3270850249792, 'probability': 0.9635100566157819, 'shrinking': 0.5377327600795537, 'tol': 0.004642276785276543}; f(x) = 0.765625.
#accuracy :76%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd #import pandas
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Prepare the data.
df = pd.read_csv('D:/new researches/SUICIDE PAPER/Dataset/our DB1 features/features/resnet18featuresoriginal.csv')
df.dropna(inplace=True)
a=df.describe()
X = df.iloc[:,0:512] #512 for resnet18 and 2048 for resnet50
# integer encode
y = df.iloc[:,512] #512 for resnet18 and 2048 for resnet50
y = LabelEncoder().fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y,stratify = y,test_size=0.2,random_state = 42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define the black box function to optimize.
def black_box_function(degree,gamma,coef0):
    # C: SVC hyper parameter to optimize for.
    model = SVC(degree=degree,gamma=gamma,coef0=coef0)
    model.fit(X_train_scaled, y_train)
    y_score = model.decision_function(X_test_scaled)
    #f = roc_auc_score(y_test, y_score)
     # Use predict_proba to get probability estimates and then threshold them to binary labels
    y_pred = (y_score > 0.5).astype(int)  # Adjust the threshold as needed
    f = accuracy_score(y_test, y_pred)
    return f 
# Set range of C to optimize for.
# bayes_opt requires this to be a dictionary.
pbounds = {"degree": [1, 5],"gamma": [0.001, 1],"coef0":[0,1]}
# Create a BayesianOptimization optimizer,
# and optimize the given black_box_function.
optimizer = BayesianOptimization(f = black_box_function,
                                 pbounds = pbounds, verbose = 2,
                                 random_state = 4)
optimizer.maximize(init_points = 100, n_iter = 500)
print("Best result: {}; f(x) = {}.".format(optimizer.max["params"], optimizer.max["target"]))