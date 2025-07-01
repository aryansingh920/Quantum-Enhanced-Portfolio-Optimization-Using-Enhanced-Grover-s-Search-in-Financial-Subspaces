from alpha_vantage.timeseries import TimeSeries

ts = TimeSeries(key='NSMUG8PQ0AAF77JV', output_format='pandas')
data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')
print(data.head())
