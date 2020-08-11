import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv(".\src\Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_x1 = LabelEncoder()
label_encoder_x2 = LabelEncoder()
X[:, 1] = label_encoder_x1.fit_transform(X[:, 1])
X[:, 2] = label_encoder_x2.fit_transform(X[:, 2])

ct = ColumnTransformer([("Geography", OneHotEncoder(drop="first", dtype=np.int), [1])], remainder='passthrough')
X = ct.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


def build_nn(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, input_dim=11, activation="relu", kernel_initializer="uniform"))
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
    classifier.add(Dense(units=1, activation="relu", kernel_initializer="uniform"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return classifier


classifier = KerasClassifier(build_fn=build_nn)
params = {
    'batch_size': [10, 15, 19, 25, 27, 8000],
    'nb_epoch': [100, 200, 300, 400],
    'optimizer': ['adam', 'rmsprop']
}
grid_search = GridSearchCV(estimator=classifier, param_grid=params, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(best_params)
print(best_score)
