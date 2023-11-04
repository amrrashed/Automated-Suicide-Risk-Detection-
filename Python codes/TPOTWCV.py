import numpy as np
import pandas as pd #import pandas
from tpot import TPOTClassifier
from sklearn import preprocessing
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
  ########not working ###
del()  
#database
# Prepare the data.
#resnet18featuresoriginal
#resnet18featurescrop
#resnet50featuresoriginal
#resnet50featurescrop
df = pd.read_csv('D:/new researches/SUICIDE PAPER/Dataset/our DB1 features/features/nasnetmobile_features_original.csv')
df.dropna(inplace=True)
a=df.describe()
X = df.iloc[:,0:1056] #512 for resnet18 and 2048 for resnet50
# integer encode
y = df.iloc[:,1056] #512 for resnet18 and 2048 for resnet50
y = LabelEncoder().fit_transform(y)


# Look at the dataset again
print(f'Number of Rows: {df.shape[0]}')
print(f'Number of Columns: {df.shape[1]}')
print(df.head())
# define model evaluation
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=123)
tpot = TPOTClassifier(generations=5, population_size=50, cv=cv, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)
tpot.fit(X, y)
tpot.export('bestmodel2.py')
# clf = TPOTClassifier(config_dict='TPOT NN', template='Selector-Transformer-PytorchLRClassifier',
#                      verbosity=2, population_size=10, generations=5)
# clf.fit(X_train, y_train)
# print(clf.score(X_test, y_test))
# clf.export('tpot_nn_demo_pipeline.py')