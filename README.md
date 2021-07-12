# Univariate LSTM model for one-step USD/RUB forecast
This is a univariate LSTM model for one-step USD/RUB price forecast, i.e. observation from the prior time step (t-1) is used to predict the observation at the current time step (t). I used python 3.7 and keras 2.3.1 with TensorFlow backend.

## 1st step: Import data
File [import_data.py](https://github.com/MurakamiNao/Predicting-RUB-USD-with-LSTM/blob/main/import_data.py). Import a 10-year daily prices of USD/RUB from Moscow Exchange (MOEX), using DataReader. This returns DataFrame with multiple columns: open, close, low, high price etc. Select close price. Also MOEX has several trade modes (BOARDID field).  Select  systemic trade mode (CETS). 
```
# import data from MOEX
eq=pandas_datareader.DataReader(ticker,'moex',start,end)

# systemic trade mode
eq=eq[eq['BOARDID']=='CETS']

# close prices
close=eq['CLOSE']
```
## 2nd step: Prepare Data 
File [LSTM.py](https://github.com/MurakamiNao/Predicting-RUB-USD-with-LSTM/blob/main/LSTM.py). 
### 2.1 Remove trend
Data set is non stationary, in particular it has an increasing trend.
![Alt-текст](https://github.com/MurakamiNao/Predicting-RUB-USD-with-LSTM/blob/main/Figure_1.png)
Remove a trend by differencing the data, i.e. subtract  previous day price from current price.
```
# remove trend by differencing data
diff = close.diff().dropna()
```
|TRADEDATE  |CLOSE   |
|-----------|--------|
|2011-07-12 |0.0450  |
|2011-07-13 |-0.2750 |
|2011-07-14 |0.0650  |
|2011-07-15 |-0.0025 |
|2011-07-18 |0.1350  |

### 2.2 Supervised learning
Divide data into input x and output y. Use price from the previous day as the input and current price as the output.
```
# convert data to supervised learning
df=diff.assign(x=diff.shift(1))
df.columns=['y','x']
df=df.reindex(columns=['x','y'])
df.fillna(0,inplace=True)
```

|TRADEDATE  |x       |y       | 
|-----------|--------|--------|
|2011-07-12 |0.0000  |0.0450  |
|2011-07-13 |0.0450  |-0.2750 |
|2011-07-14 |-0.2750 |0.0650  |
|2011-07-15 |0.0650  |-0.0025 |
|2011-07-18 |-0.0025 |0.1350  |

### 2.3 Train and test subsets
Split data into 70% train and 30% test subsets.
```
# split data into  train and test subsets
train_size = int(len(df) * 0.7)
test_size = len(df) - train_size
train, test = df[0:-test_size], df[-test_size:]
```
### 2.4 Scaling to the range of activation function
Transform data to the range of LSTM activation function (by default hyperbolic tangent), using the MinMaxScaler class.
Calculate Min and max values on the train dataset to avoid test data influencing model.
```
# scale train and test data to the range of activation function
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)
```
### 2.5 Reshape data into LSTM input format
Input in LTSM is a 3D tensor with shape [batch, timesteps, feature]. Reshape input data into 3D format. Input is the price of a previous day.
Therefore, there is only one time step (t-1) and one feature (price itself).
```
# split  data into input and output
train_x, train_y = train_scaled[:, 0], train_scaled[:, 1]
test_x, test_y = test_scaled[:, 0], test_scaled[:, 1]

# reshape data into LSTM input format: [samples, time steps, features]
train_x = train_x.reshape(train_x.shape[0], 1, 1)
test_x= test_x.reshape(test_x.shape[0], 1, 1)
```
## 3rd step: Build and compile network
Build  Sequential model, which is a linear stack of layers: LSTM hidden layer with 5 memory blocks or neurons and densely-connected layer with 1 neuron to output forecast.

Input data specified in LSTM layer must have  3D dimention:  [batch_size, time steps, features], where batch_size is the number of samples shown to Neural Network before updating the weights. The size of  training and test datasets must be divisible by batch_size without a remainder. Set n_batch=1, i.e. update weights every timestep.

Make LSTM stateful by setting stateful=True to avoid clearing the state between batches. 

Leave default hyperbolic tangent as an activation function for LSTM layer. For output layer in the case of  regression problem use default linear activation function.

I didn't tune the network parameters and used 5 neurons in LSTM layer and 1000 Epochs, because dataset is large and training takes a long time.
```
# set hyperparameters
n_batch=1
n_epochs=1000
neurons=5

# build the network
model = Sequential()
# LSTM hidden layer
model.add(LSTM(neurons, batch_input_shape=(n_batch, train_x.shape[1], train_x.shape[2]),stateful=True))
# densely-connected layer - output layer
model.add(Dense(1))
```
After building the network compile it. Specify popular Adam optimization algorithm. Use Mean Squared Error loss function for the regression problem.
```
# compile the network
adam = optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=adam)
```
## 4th step: Train model
Train the model for 1000 epochs. Reset the internal state at the end of the training epoch.
Set shuffle=False to disable shuffling  samples prior to being exposed to the network.
```
# manually fit the network to the training data
for i in range(n_epochs):
    model.fit(train_x, train_y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
    model.reset_states()
```
## 5th step: Make predictions
Use model to predict train and test datasets. Set batch size to 1 to  make one-step forecasts.
```
# forecast train dataset 
predictedTrain=model.predict(train_x, batch_size=n_batch)
# forecast test dataset
predictedTest = model.predict(test_x, batch_size=n_batch)
```
## 6th step: Transform predictions to the original format
Predicted values are in a scale of LSTM activation function, i.e. tanh. Function invert_scale returns forecasts to the original scale, using inverse_transform method. 
Because data were scaled before spliting into input (x) and output(y), invert_scale requires input shape to be [n_samples,2]. Concatenate input values with values predicted by a model.
```
# invert predicted value to the original scale
def invert_scale(predicted,scaled):
    x=scaled[:,0]
    agg=concat([DataFrame(x),DataFrame(predicted)],axis=1)
    inverted=scaler.invert_scale(agg)
    return inverted[:, -1]

invertedTrain=invert_scale(predictedTrain,train_scaled)
invertedTest=invert_scale(predictedTest,test_scaled)
```
After inverting to the original scale, predicted values are still need to be transformed, because they're differenced values. Function  invert_diff  invert differenced values back to a form of prices.
```
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
```
## 7th step: Evaluate performance
Finally, calculate an error score to evaluate the skill of the model. Choose root mean squared error (RMSE) because it punishes large errors. 
I recieved 0.595 RUB train RMSE and 0.770 RUB test RMSE. The results are pretty good: test error isn't much bigger than train error.
![Alt-текст](https://github.com/MurakamiNao/Predicting-RUB-USD-with-LSTM/blob/main/Figure_1.png)
```
# evaluate the skill of the model
rmseTrain = sqrt(mean_squared_error(raw_data[1:train_size+1], invertedTrain))
rmseTest = sqrt(mean_squared_error(raw_data[train_size+1:], invertedTest))

# plot actual vs predicted test values
plt.plot(raw_data[train_size+1:])
plt.plot(invertedTest)
plt.legend(['actual','predicted'],loc='upper left')
plt.show()
```
