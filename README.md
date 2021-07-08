# Univariate LSTM model for predicting USD/RUB
In this project i built simple univariate model to predict USD/RUB exchange rate, using (Long Short-Term Memory) LSTM recurrent neural network.
## 1st step: Importing data
In import_data.py I extract a 10-year daily prices of USD/RUB from Moscow Exchange, using pandas_datareader.DataReader.
DataReader returns DataFrame with multiple columns. i select only close price. Also MOEX has several trade modes (BOARDID field). We only needed systemic mode (CETS). 
Finally, Write data to csv.
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
Data set is non stationary, in particular it has increasing trend.
![Alt-текст]()
