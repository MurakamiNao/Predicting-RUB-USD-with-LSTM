# Extracting 10-year historical close prices of USD/RUB from Moscow Exchange (MOEX), using DataReader
import pandas_datareader as web
import datetime as dt

ticker='USD000UTSTOM'
end=dt.datetime.today()
start=dt.date(end.year-10,end.month,end.day)
#print(start)

eq=web.DataReader(ticker,'moex',start,end)
# Choose systemic trade mode.
eq=eq[eq['BOARDID']=='CETS']
print(eq.columns)
print(eq.head())

# Choose close prices
close=eq['CLOSE']
print(close.head())

close.to_csv('RUB_close.csv')
