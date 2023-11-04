# -*- coding: utf-8 -*-
"""h2otest.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1eFl7aRBbcWi5df5Qimq1Nx3fV3-ciBvB

[h2o website](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)

Performance :https://docs.h2o.ai/h2o/latest-stable/h2o-docs/performance-and-prediction.html

tutorial :https://www.youtube.com/watch?v=91QljBnvM7s&ab_channel=AIEngineering

another_code:https://colab.research.google.com/github/srivatsan88/YouTubeLI/blob/master/H2O_AutoML.ipynb#scrollTo=BiTtGL6IcA8J

https://towardsdatascience.com/automated-machine-learning-with-h2o-258a2f3a203f

https://h2oai.atlassian.net/browse/PUBDEV-7936?filter=21603
"""

pip install requests

pip install tabulate

pip install "colorama>=0.3.8"

pip install future

pip install h2o

"""# our code"""

import h2o
from h2o.automl import H2OAutoML
h2o.init()

from google.colab import drive
drive.mount('/content/drive')

# Import a sample binary outcome train/test set into H2O
prostate = h2o.import_file("/content/drive/MyDrive/breast cancer datasets/DB1.csv")
prostate.head()

# split into train and testing sets
train, test = prostate.split_frame(ratios = [0.8], seed = 123)
x = train.columns
y = "class"
x.remove(y)
# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Run AutoML for 10 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)

preds = aml.leader.predict(test)
print(preds)

# Get the best model using the metric
m = aml.leader
# this is equivalent to
m = aml.get_best_model()
print(m)

# Get the best model using a non-default metric
#m = aml.get_best_model(criterion="logloss")
#print(m)
best_model = aml.get_best_model()
print(best_model)

best_model.model_performance(test)

best_model.accuracy()