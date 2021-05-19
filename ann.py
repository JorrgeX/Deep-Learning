# Artificial Neural Network

# import the libraries
import numpy as np
import pandas as pd 
import tensorflow as tf 
# print(tf.__version__)


# Part 1 --- Data Processing
# import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
# print(x)
# print(y)

# encoding categorical data
# label encoding the "gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])
# print(x)

# one hot encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))
# print(x)

# split the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# future scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# Part 2 --- ANN
# initializing ANN
ann = tf.keras.models.Sequential()

# adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Part 3 --- Train the ANN
# compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the ANN on the training set
ann.fit(x_train, y_train, batch_size=32, epochs=100)


# Part 4 --- Making the prediction and evaluating the model
# predict the result from a single observation
# France
# 600
# Male
# 40 years old
# 3 years
# $60000
# 2 products
# yes 
# yes 
# $50000
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

# predicting the test set results
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_pred, y_test)