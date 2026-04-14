import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

data = load_breast_cancer()
X = data.data
y = data.target

X = pd.DataFrame(x, columns=data.feature_names)
y = pd.DataFrame(y, columns=['target'])

data = pd.concat([X, y], axis = 1)

for feature in X.columns:
  for targ in y['target'].unique():
    df = data[(data['target']) == targ]
    plt.hist(df[feature], bins = 20, alpha = 0.3, label = targ)
    plt.title(feature)
    plt.legend()
  plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter = 5000))
    ])

param_grid = [
    {
        'model__penalty': ['l1'],
        'model__C': [0.01, 0.1, 1., 10.],
        'model__solver': ['liblinear'],
        'model__class_weight': ['balanced']
    },
    {
        'model__penalty': ['l2'],
        'model__C': [0.01, 0.1, 1., 10.],
        'model__solver': ['lbfgs'],
        'model__class_weight': ['balanced']
    }
]

grid = GridSearchCV(pipe, param_grid, cv = StratifiedKFold(n_splits = 5, shuffle=True), scoring = 'accuracy')
grid.fit(X_train, np.array(y_train).ravel())
print(grid.best_params_)
model = grid.best_estimator_
coef = model.named_steps['model'].coef_
coef = pd.Series(coef.ravel(), index= X.columns)
coef = coef.sort_values(ascending = False)
print(coef)
