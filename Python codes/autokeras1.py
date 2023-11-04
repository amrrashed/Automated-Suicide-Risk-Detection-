# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 16:10:24 2020
https://machinelearningmastery.com/autokeras-for-classification-and-regression/
@author: amr_r
"""
import time
import numpy as np
import pydot
from numpy import asarray
import pandas as pd #import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from autokeras import StructuredDataClassifier
#from keras.utils.vis_utils import plot_model
#from keras.models import model_from_json
import tensorflow as tf


#suppress all FutureWarnings
# import warnings filter
from warnings import simplefilter
from keras.utils.vis_utils import plot_model
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=Warning)
# check GPU
#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) 

df = pd.read_csv('G:/new researches/breast cancer text paper/databases/DB1/DB1.csv') 

# basic data preparation
X = np.array(df.drop(['class'], 1)) #input
X = X.astype('float32')
y = np.array(df['class'])   #output
# integer encode
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=123)
search = StructuredDataClassifier(max_trials=50)
search.fit(x=X_train, y=y_train, verbose=1)
loss, acc = search.evaluate(X_test, y_test, verbose=0) #classification
print('Accuracy: %.3f' % acc)
print('loss: %.3f' % loss)
y_predictions = search.predict(X_test)
model = search.export_model()
model.summary()
print("\n%s: %.2f%%" % (model.metrics_names[1], acc*100))
model.save('breast_cancer_model.tf')

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)