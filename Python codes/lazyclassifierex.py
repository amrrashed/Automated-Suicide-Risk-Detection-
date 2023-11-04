import numpy as np
import pydot
from numpy import asarray
import pandas as pd #import pandas
from lazypredict.Supervised import LazyClassifier
#from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


#df = pd.read_csv('D:/new researches/SUICIDE PAPER/Dataset/our DB1 features/features/resnet18featurescrop.csv') 
##acc 66% BaggingClassifier  balanced 63%
#df = pd.read_csv('D:/new researches/SUICIDE PAPER/Dataset/our DB1 features/features/resnet18featuresoriginal.csv')
##acc 70% BernoulliNB balanced 70%
#df = pd.read_csv('D:/new researches/SUICIDE PAPER/Dataset/our DB1 features/features/resnet50featurescrop.csv')
##acc 66% BernoulliNB balanced 65%
df = pd.read_csv('D:/new researches/SUICIDE PAPER/Dataset/our DB1 features/features/efficientnetb0_features_crop.csv')
##acc 69% gaussianNB balanced 68%
df.dropna(inplace=True)
a=df.describe()
X = df.iloc[:,0:1280] #512 for resnet18 and 2048 for resnet50
# basic data preparation
#X = np.array(df.drop(['class'], 1)) #input
#X = X.astype('float32')
#y = np.array(df['class'])   #output
# integer encode
y = df.iloc[:,1280] #512 for resnet18 and 2048 for resnet50
y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state =123)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)