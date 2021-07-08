# Univariate LSTM model for predicting USD/RUB
This is univariate LSTM model to predict USD/RUB exchange rate, using (Long Short-Term Memory) LSTM recurrent neural network. Previous day prices are used to predict Observation from the prior time step (t-1) is used to predict the observation at the current time step (t).
## 1st step: Importing data
File [import_data.py](https://github.com/MurakamiNao/Predicting-RUB-USD-with-LSTM/blob/main/import_data.py). Extract a 10-year daily prices of USD/RUB from Moscow Exchange, using pandas_datareader.DataReader. Select  systemic trade mode (CETS) and only close prices. Write data to csv.
```
import pandas_datareader as web

eq=web.DataReader(ticker,'moex',start,end)
# CETS trade mode
eq=eq[eq['BOARDID']=='CETS']
# close prices
close=eq['CLOSE']
close.to_csv('Data\RUB_close.csv')
```
## 2nd step: data preparation
File LSTM.py. Read data from csv.
```
close = read_csv('Data\RUB_close.csv',index_col=0,header=0)
```
### Remove trend
Data set is non stationary, in particular it has an increasing trend.
![Alt-текст](https://github.com/MurakamiNao/Predicting-RUB-USD-with-LSTM/blob/main/Historical_prices.png)
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

### split data into  train and test subsets
```
train_size = int(len(df) * 0.7)
test_size = len(df) - train_size
train, test = df[0:-test_size], df[-test_size:]
```


# scale train and test data to the range of activation function
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train)
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)
# print(train_scaled.shape)
# print(test_scaled.shape)
