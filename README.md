# Univariate LSTM model for one-step USD/RUB forecast
In this code I built univariate LSTM model for one-step USD/RUB price forecast, i.e. observation from the prior time step (t-1) is used to predict the observation at the current time step (t).
## 1st step: Import data
File [import_data.py](https://github.com/MurakamiNao/Predicting-RUB-USD-with-LSTM/blob/main/import_data.py). Import 10-year daily prices of USD/RUB from Moscow Exchange (MOEX), using DataReader. This returns DataFrame with multiple columns: open, close, low, high price etc. Select close price. Also MOEX has several trade modes (BOARDID field).  Select  systemic trade mode (CETS). 
```
import pandas_datareader as web

ticker='USD000UTSTOM'
end=dt.datetime.today()
start=dt.date(end.year-10,end.month,end.day)

eq=web.DataReader(ticker,'moex',start,end)
# CETS trade mode
eq=eq[eq['BOARDID']=='CETS']
# close rices
close=eq['CLOSE']
```
## 2nd step: Prepare Data 
File [LSTM.py](https://github.com/MurakamiNao/Predicting-RUB-USD-with-LSTM/blob/main/LSTM.py). 
### Remove trend
Data set is non stationary, in particular it has an increasing trend.
![Alt-текст](https://github.com/MurakamiNao/Predicting-RUB-USD-with-LSTM/blob/main/Figure_1.png)
Remove a trend by differencing the data.
```
diff = close.diff().dropna()
```
|TRADEDATE  |CLOSE   |
|-----------|--------|
|2011-07-12 |0.0450  |
|2011-07-13 |-0.2750 |
|2011-07-14 |0.0650  |
|2011-07-15 |-0.0025 |
|2011-07-18 |0.1350  |

### Supervised learning
Divide data into input x and output y. Use price from the previous day as the input and current price as the output.
```
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

### Train and test subsets
Split data into 70% train and 30% test subsets
```
train_size = int(len(df) * 0.7)
test_size = len(df) - train_size
train, test = df[0:-test_size], df[-test_size:]
```
### Scaling to the range of activation function
Transform data to the range of LSTM activation function ( by default hyperbolic tangent), using the MinMaxScaler class.
To avoid contaminating the experiment with knowledge from the test dataset, calculate Min and max values on the train dataset.

```
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)
```
### Reshape data into LSTM input format
Input in LTSM is a 3D tensor with shape [batch, timesteps, feature]. Reshape input data into 3D format. Input is the price of a previous day.
Therefore, there is only one time step (t-1) and one feature (price itself)

```
# split  data into input and output
train_x, train_y = train_scaled[:, 0], train_scaled[:, 1]
test_x, test_y = test_scaled[:, 0], test_scaled[:, 1]

# reshape data into LSTM input format: [samples, time steps, features]
train_x = train_x.reshape(train_x.shape[0], 1, 1)
test_x= test_x.reshape(test_x.shape[0], 1, 1)
```
## 3rd step: Build Network
Build  Sequential model, which is a linear stack of layers:  LSTM hidden layer with 5 memory blocks or neurons and densely-connected layer with 1 neuron to output forecast.
Input data specified in LSTM layer must have  3D dimention:  [batch_size, time steps, features], where batch_size is the number of samples shown to Neural Network before updating the weights. Number of training samples must be divisible without reminder on batch_size. Set n_batch=1, i.e. update weights every timestep.
Make LSTM stateful by setting stateful=True to avoid clearing the state between batches.
Leave  default hyperbolic tangent for LSTM layer. For output layer in the case of  regression problem use default linear activation function.
After building the network compile it. Specify popular Adam optimization algorithm. Use Mean Squared Error loss function for the regression problem.

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

# compile the network
adam = optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=adam)
```
## 4th step: Train model
Train the model for 1000 epochs or iterations. Reset the internal state at the end of the training epoch.
Set shuffle=False to disable shuffling  samples prior to being exposed to the network.

```
# manually fit the network to the training data: reset the internal state at the end of the training epoch
for i in range(n_epochs):
    model.fit(train_x, train_y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
    model.reset_states()
```
