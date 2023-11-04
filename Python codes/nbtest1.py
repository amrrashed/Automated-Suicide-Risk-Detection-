import pandas as pd #import pandas
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Prepare the data.
#nasnetmobile_features_cropped
#nasnetmobile_features_original
#efficientnetb0_features_crop
#efficientnetb0_features_original
df = pd.read_csv('D:/new researches/SUICIDE PAPER/Dataset/our DB1 features/features/efficientnetb0_features_original.csv')
df.dropna(inplace=True)
a=df.describe()
X = df.iloc[:,0:1280] #512 for resnet18 and 2048 for resnet50
# integer encode
y = df.iloc[:,1280] #512 for resnet18 and 2048 for resnet50
y = LabelEncoder().fit_transform(y)

# Define a function to optimize with Bayesian Optimization
def optimize_alpha(alpha):
    # Create a BernoulliNB model with the specified alpha
    bnb = BernoulliNB(alpha=alpha)

    # Perform 5-fold cross-validation and calculate the mean accuracy
    accuracy = cross_val_score(bnb, X, y, cv=5, scoring='accuracy').mean()

    return accuracy

# Define the parameter bounds for alpha
pbounds = {'alpha': (1e-3, 10.0)}

# Create a BayesianOptimization optimizer
optimizer = BayesianOptimization(
    f=optimize_alpha,
    pbounds=pbounds,
    random_state=42,
    allow_duplicate_points=True
)

# Perform the optimization
optimizer.maximize(init_points=5, n_iter=50)

# Get the best alpha value
best_alpha = optimizer.max['params']['alpha']
print("Best alpha:", best_alpha)

# Train a BernoulliNB model with the best alpha
bnb_best = BernoulliNB(alpha=best_alpha)
accuracy = cross_val_score(bnb_best, X, y, cv=5, scoring='accuracy').mean()
print("accuracy:",accuracy)
