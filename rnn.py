# Recurrent Neural Network



# Part 1 - Data Preprocessing

# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the training sets
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled  = sc.fit_transform(training_set)

# creating a data structure with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range(60, 1258):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# reshaping
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Part 2 - Building the RNN

# importing the keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# initialize the RNN
regressor = Sequential()

# adding the first LSTM layer and some Dropout regularisations
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2))

# adding the second LSTM layer and some Dropout regularisations
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# adding the third LSTM layer and some Dropout regularisations
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# adding the fourth LSTM layer and some Dropout regularisations
regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))

# adding the output layer
regressor.add(Dense(units=1))

# compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# fitting the RNN to the training set
regressor.fit(x_train, y_train, epochs=100, batch_size=32)


# Part 3 - Making the predictions and visualising the results

# getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# getting the predicted stock of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualizing the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
