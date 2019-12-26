import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import time

# api_key = 'D3YIZ9Q9T9COACJ4'
# ts = TimeSeries(key=api_key, output_format='pandas')
# while(1):
#     data, meta_data = ts.get_intraday(
#         symbol='MSFT', interval='1min', outputsize='full')
#     data = data[['4. close']]
#     print(data)
#     data.to_csv(('per minute data.csv'))
#     time.sleep(60)
df = pd.DataFrame(columns=['Grapes'])
df.loc['Day1', 'Grapes'] = 40
df.loc['Day2', 'Grapes'] = 40
print(df)
