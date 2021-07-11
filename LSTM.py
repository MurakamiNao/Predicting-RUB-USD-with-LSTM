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
# np.random.seed(7)

# set hyperparameters
n_batch=1
n_epochs=1000
neurons=5

# read data from csv file
close = read_csv('Data\RUB_close.csv',index_col=0,header=0)
# print(close.head())
raw_data=close.values

# remove trend by differencing data
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
test_size = len(df) - train_size
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

# split  data into input and output
train_x, train_y = train_scaled[:, 0], train_scaled[:, 1]
test_x, test_y = test_scaled[:, 0], test_scaled[:, 1]

# reshape data into LSTM input format: [samples, time steps, features]
train_x = train_x.reshape(train_x.shape[0], 1, 1)
# print(train_x.shape)
test_x= test_x.reshape(test_x.shape[0], 1, 1)
# print(test_x.shape)

# build the network
model = Sequential()
# LSTM hidden layer
model.add(LSTM(neurons, batch_input_shape=(n_batch, train_x.shape[1], train_x.shape[2]),stateful=True))
# densely-connected layer - output layer
model.add(Dense(1))

# compile the network
adam = optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=adam)

# manually fit the network to the training data
for i in range(n_epochs):
    model.fit(train_x, train_y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
    model.reset_states()

# forecast training dataset 
predictedTrain=model.predict(train_x, batch_size=n_batch)
# forecast test dataset
predictedTest = model.predict(test_x, batch_size=n_batch)

# invert predicted value to the original scale
def invert_scale(predicted,scaled):
    x=scaled[:,0]
    agg=concat([DataFrame(x),DataFrame(predicted)],axis=1)
    inverted=scaler.inverse_transform(agg)
    return inverted[:, -1]

invertedTrain=invert_scale(predictedTrain,train_scaled)
invertedTest=invert_scale(predictedTest,test_scaled)

# return differenced values back to a form of prices
def invert_diff(inverted,raw_data):
    predictions = list()
    for i in range(len(inverted)):
        yhat=inverted[i]+raw_data[i]
        yhat=yhat[0]
        predictions.append(yhat)
    return predictions

invertedTrain=invert_diff(invertedTrain,raw_data[:train_size])
invertedTest=invert_diff(invertedTest,raw_data[train_size:])

# evaluate the skill of the model
rmseTrain = sqrt(mean_squared_error(raw_data[1:train_size+1], invertedTrain))
print('Train RMSE: %.3f' % rmseTrain)
rmseTest = sqrt(mean_squared_error(raw_data[train_size+1:], invertedTest))
print('Test RMSE: %.3f' % rmseTest)

# plot actual vs predicted test values
plt.plot(raw_data[train_size+1:])
plt.plot(invertedTest)
plt.legend(['actual','predicted'],loc='upper left')
plt.show()

