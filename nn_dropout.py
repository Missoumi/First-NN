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

from keras.models import Sequential
from keras.layers import Dense, Dropout

# dropout
classifier = Sequential()
classifier.add(Dense(units=6, input_dim=11, activation="relu", kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(units=6, activation="relu", kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(units=1, activation="sigmoid", kernel_initializer="uniform"))
classifier.compile(optimizer="adam", metrics=["accuracy"], loss="binary_crossentropy")

classifier.fit(X_train, Y_train, batch_size=512, epochs=100)


