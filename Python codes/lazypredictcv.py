from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix

# Load a sample dataset (e.g., Iris dataset)
data = load_iris()
X = data.data
y = data.target

# Initialize LazyClassifier
clf = LazyClassifier(predictions=True)

# Define the number of splits (e.g., 5-fold cross-validation)
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits)

# Initialize lists to store metrics
cv_auc_scores = []
specificity_scores = []
sensitivity_scores = []
logloss_scores = []
balanced_accuracy_scores = []

# Perform cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calculate AUC
    cv_auc = roc_auc_score(y_test, y_pred)
    cv_auc_scores.append(cv_auc)

    # Calculate additional metrics using confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    logloss = log_loss(y_test, y_pred)
    balanced_accuracy = (specificity + sensitivity) / 2

    specificity_scores.append(specificity)
    sensitivity_scores.append(sensitivity)
    logloss_scores.append(logloss)
    balanced_accuracy_scores.append(balanced_accuracy)

# Print the metrics
print("Cross-validation AUC scores:", cv_auc_scores)
print("Specificity scores:", specificity_scores)
print("Sensitivity scores:", sensitivity_scores)
print("Log Loss scores:", logloss_scores)
print("Balanced Accuracy scores:", balanced_accuracy_scores)
