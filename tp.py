import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow
# import dataset
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv(".\src\Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_X_1 = LabelEncoder()
label_encoder_X_2 = LabelEncoder()
X[:, 2] = label_encoder_X_2.fit_transform(X[:, 2])
X[:, 1] = label_encoder_X_1.fit_transform(X[:, 1])

ct = ColumnTransformer([("Geography", OneHotEncoder(drop="first", dtype=np.int), [1])], remainder='passthrough')
X = ct.fit_transform(X)

# split  the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Build ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

# classifier = Sequential()
#
# #   First hidden layer
# # classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11))
#
# #   Second hidden layer
# classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
#
# #   output layer
# classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
#
# #   Compile Ann
# classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#
# classifier.fit(X_train, Y_train, batch_size=512, epochs=100)
#
# y_pred = classifier.predict(X_test)
# y_pred = (y_pred > 0.5)
# print(y_pred)
#
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(Y_test, y_pred)
# print(cm)

# k Fold CV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform", input_dim=11))
    classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
    classifier.add(Dense(units=1, activation="relu", kernel_initializer="uniform"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=100)
accuracies = cross_val_score(estimator=classifier, X= X_train, y=Y_train, cv=10)
mean = accuracies.mean()
variance = accuracies.std()

# Dropout regularization

