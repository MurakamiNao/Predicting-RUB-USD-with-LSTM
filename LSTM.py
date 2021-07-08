from pandas import DataFrame
from pandas import read_csv
from pandas import Series
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
from matplotlib import pyplot as plt
import numpy as np
from keras import optimizers

# fix random seed for reproducibility
np.random.seed(7)

# Set hyperparameters
n_batch=1
n_epochs=10
neurons=5

# read data from csv file
close = read_csv('Data\RUB_close.csv',index_col=0,header=0)
# print(close.head())
raw_data=close.values
# print(raw_data)
# print(raw_data.shape)

# plot historical data
# close.plot()
# plt.show()

# remove a trend by differencing the data
diff = close.diff().dropna()
# print(diff.head())

# convert data to supervised learning
df=diff.assign(x=diff.shift(1))
df.columns=['y','x']
df=df.reindex(columns=['x','y'])
df.fillna(0,inplace=True)
#print(df.head())


# split data into  train and test subsets
train_size = int(len(df) * 0.7)
print(train_size)
test_size = len(df) - train_size
print(test_size)
# print(train_size,test_size)
train, test = df[0:-test_size], df[-test_size:]
# print(train.shape)
# print(test.shape)


# scale train and test data to the range of activation function
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)
# print(train_scaled.shape)
# print(test_scaled.shape)
#
# split  data into input and output
train_x, train_y = train_scaled[:, 0], train_scaled[:, 1]
test_x, test_y = test_scaled[:, 0], test_scaled[:, 1]
# print(train_x.shape)
# print(test_x.shape)
# reshape data into LSTM input format: [samples, time steps, features]
train_x = train_x.reshape(train_x.shape[0], 1, 1)
# print(train_x.shape)
test_x= test_x.reshape(test_x.shape[0], 1, 1)
# print(test_x.shape)
# print(raw_data[1:train_size+1])
# print(raw_data[1:train_size+1].shape)
# print(raw_data[train_size+1:])
# print(raw_data[train_size+1:].shape)

# build the network
model = Sequential()
# LSTM hidden layer
model.add(LSTM(neurons, batch_input_shape=(n_batch, train_x.shape[1], train_x.shape[2]),stateful=True))
# densely-connected layer - output layer
model.add(Dense(1))

# compile the network
adam = optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=adam)

# manually fit the network to the training data: reset the internal state at the end of the training epoch
for i in range(n_epochs):
    model.fit(train_x, train_y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
    model.reset_states()

# forecast the entire training dataset to build up state for forecasting
outputTrain=model.predict(train_x, batch_size=n_batch)
# print(outputTrain)
print(len(outputTrain))

# forecast test dataset
outputTest = model.predict(test_x, batch_size=n_batch)
# print(outputTest)
print(len(outputTest))

#walk-forward validation on the test data
def walk_forward_validation(output,scaled,raw_data,scaler):
    predictions = list()
    for i in range(len(output)):
        yhat = output[i, :]
        X=scaled[i, 0]
        new_row = [X] + [yhat]
        array = np.array(new_row)
        array = array.reshape(1, len(array))
        inverted = scaler.inverse_transform(array)
        yhat=inverted[0, -1]
        yhat=yhat + raw_data[-(len(scaled)+1)+i]
        # print(raw_data[-len(scaled)+i])
        # print(yhat)
        predictions.append(yhat)
    return predictions

Train_Predicted=walk_forward_validation(outputTrain,train_scaled,raw_data[:train_size+1],scaler)
Test_Predicted=walk_forward_validation(outputTest,test_scaled,raw_data,scaler)

# report performance
rmseTrain = sqrt(mean_squared_error(raw_data[1:train_size+1], Train_Predicted))
print('Train RMSE: %.3f' % rmseTrain)
rmseTest = sqrt(mean_squared_error(raw_data[train_size+1:], Test_Predicted))
print('Test RMSE: %.3f' % rmseTest)

# line plot of observed vs predicted
plt.plot(raw_data[train_size+1:])
plt.plot(Test_Predicted)
plt.legend(['actual','predicted'],loc='upper left')
plt.show()

