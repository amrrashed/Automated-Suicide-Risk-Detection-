# -*- coding: utf-8 -*-
#Created on Thu Oct 19 15:24:55 2023
#author: https://towardsdatascience.com/bayesian-optimization-with-python-85c66df711ec

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
# Create the optimizer. The black box function to optimize is not
# specified here, as we will call that function directly later on.
optimizer = BayesianOptimization(f = None, 
                                 pbounds = {"C": [0.01, 10], 
                                            "degree": [1, 5]}, 
                                 verbose = 2, random_state = 1234)
# Specify the acquisition function (bayes_opt uses the term
# utility function) to be the upper confidence bounds "ucb".
# We set kappa = 1.96 to balance exploration vs exploitation.
# xi = 0.01 is another hyper parameter which is required in the
# arguments, but is not used by "ucb". Other acquisition functions
# such as the expected improvement "ei" will be affected by xi.
utility = UtilityFunction(kind = "ucb", kappa = 1.96, xi = 0.01)
# We want to optimize both C and degree simultaneously.
def black_box_function(C, degree):
    model = SVC(C = C, degree = degree)
    model.fit(X_train_scaled, y_train)
    y_score = model.decision_function(X_test_scaled)
    # f = roc_auc_score(y_test, y_score)
    y_pred = (y_score > 0.5).astype(int)  # Adjust the threshold as needed
    f = accuracy_score(y_test, y_pred)
    return f
# Optimization for loop.
for i in range(25):
    # Get optimizer to suggest new parameter values to try using the
    # specified acquisition function.
    next_point = optimizer.suggest(utility)
    # Force degree from float to int.
    next_point["degree"] = int(next_point["degree"])
    # Evaluate the output of the black_box_function using 
    # the new parameter values.
    target = black_box_function(**next_point)
    try:
        # Update the optimizer with the evaluation results. 
        # This should be in try-except to catch any errors!
        optimizer.register(params = next_point, target = target)
    except:
        pass
print("Best result: {}; f(x) = {:.3f}.".format(optimizer.max["params"], optimizer.max["target"]))
plt.figure(figsize = (15, 5))
plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "-o")
plt.grid(True)
plt.xlabel("Iteration", fontsize = 14)
plt.ylabel("Black box function f(x)", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()