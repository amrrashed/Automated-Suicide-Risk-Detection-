# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 14:29:24 2023
@author: amr_r
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Prepare the data.
df = pd.read_csv('D:/new researches/SUICIDE PAPER/Dataset/our DB1 features/features/resnet50featuresoriginal.csv')
df.dropna(inplace=True)
a = df.describe()
X = df.iloc[:, 0:2048]  # 2048 for resnet50
# integer encode
y = df.iloc[:, 2048]
y = LabelEncoder().fit_transform(y)

# Define separate optimization tasks for each kernel
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

results = []

for kernel in kernels:
    # Define a black_box_function specific to the kernel
    def black_box_function(C, degree, gamma, coef0, shrinking, probability, tol, max_iter):
        model = SVC(C=C, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, probability=probability, tol=tol, max_iter=max_iter, kernel=kernel)
        accuracy = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
        return accuracy
    
    # Set the parameter bounds for this specific kernel
    pbounds = {"C": (0.1, 10), "degree": (1, 5), "gamma": (0.001, 1), "coef0": (0, 1), "shrinking": (0, 1), "probability": (0, 1), "tol": (1e-6, 1e-2), "max_iter": (1, 1000)}
    
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
df.to_csv('svm_optimizer_results_resnet50_original.csv', index=False)
