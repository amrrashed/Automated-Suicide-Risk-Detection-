# -*- coding: utf-8 -*-
#Created on Thu Oct 19 15:24:55 2023
#author: https://towardsdatascience.com/bayesian-optimization-with-python-85c66df711ec
#########
#Best result: Kernel=rbf, Parameters={'C': 0.9951661913106971, 'coef0': 0.4562721883588112, 'degree': 2.2747847053307813, 'gamma': 0.022941066854479573, 'max_iter': 572.3270850249792, 'probability': 0.9635100566157819, 'shrinking': 0.5377327600795537, 'tol': 0.004642276785276543}, Accuracy=0.77
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

# Define separate optimization tasks for each kernel
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

results = []

for kernel in kernels:
    # Define a black_box_function specific to the kernel
    def black_box_function(C, degree, gamma, coef0, shrinking, probability, tol, max_iter):
        model = SVC(C=C, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, probability=probability, tol=tol, max_iter=max_iter, kernel=kernel)
        model.fit(X_train_scaled, y_train)
        y_score = model.decision_function(X_test_scaled)
        y_pred = (y_score > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    
    # Set the parameter bounds for this specific kernel
    pbounds = {"C": (0.1, 10), "degree": [1, 5], "gamma": (0.001, 1), "coef0": (0, 1), "shrinking": (0, 1), "probability": (0, 1), "tol": (1e-6, 1e-2), "max_iter": (1, 1000)}
    
    # Create a BayesianOptimization optimizer for this kernel
    optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, verbose=2, random_state=4)
    
    # Perform the optimization
    optimizer.maximize(init_points=100, n_iter=500)
    # Append the results for this kernel
    for result in optimizer.res:
     params = result['params']
     target = result['target']
     results.append({'Kernel': kernel, 'C': params['C'], 'Degree': params['degree'], 'Gamma': params['gamma'], 'Coef0': params['coef0'],
                     'Shrinking': params['shrinking'], 'Probability': params['probability'], 'Tol': params['tol'], 'MaxIter': params['max_iter'], 'Accuracy': target})

     # Create a DataFrame from the results
     df = pd.DataFrame(results)

     # Save the results to a CSV file
     df.to_csv('svm_optimizer_results_resnet18_original.csv', index=False)
    
   