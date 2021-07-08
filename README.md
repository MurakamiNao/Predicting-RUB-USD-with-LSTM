# Univariate LSTM model for one-step USD/RUB forecast
In this code I built univariate LSTM model for one-step USD/RUB price forecast, i.e. observation from the prior time step (t-1) is used to predict the observation at the current time step (t).
## 1st step: Importing data
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
## 2nd step: data preparation
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
