# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:57:50 2020

@author: amr_r
"""
#import tpot
from tpot import TPOTClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
#from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
#from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
#database
#

 
df = pd.read_csv('G:/new researches/breast cancer text paper/databases/DB1/DB1.csv')



cols = ['age', 'menopause', 'tumor-size', 'inv-nodes','node-caps','breast', 'breast-quad','irradiat', 'class']

df[cols] = df[cols].apply(LabelEncoder().fit_transform)

X = np.array(df.drop(['class'], 1)) #input
y = np.array(df['class'])   #output
# integer encode
#label_encoder = LabelEncoder()
#y = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.80, test_size=0.20 ,random_state=123)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
tpot = TPOTClassifier(generations=10,population_size=50,scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)
tpot.fit(X_train, y_train)
preds = tpot.predict(X_test)
print(accuracy_score(y_test, preds))