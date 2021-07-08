"""
Importing 10-year historical close prices of USD/RUB from Moscow Exchange (MOEX), using DataReader
"""

import pandas_datareader as web
import datetime as dt
from pandas import DataFrame

ticker='USD000UTSTOM'
end=dt.datetime.today()
start=dt.date(end.year-10,end.month,end.day)

# import data from MOEX
eq=web.DataReader(ticker,'moex',start,end)

# systemic trade mode
eq=eq[eq['BOARDID']=='CETS']
# print(eq.columns)

# close prices
close=eq['CLOSE']

# write data to csv
close=DataFrame(close)
close.to_csv('Data\RUB_close.csv')
