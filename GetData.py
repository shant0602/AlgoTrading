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
df = pd.DataFrame(columns=['Grapes','apples'])
df.loc['Day1', :] = [40,49]

df.loc['Day2', :] = [41,51]
df.loc['Day3', :] = [42,52]
df.loc['Day4', :] = [43,53]
df.loc['Day5', :] = [44,54]
df.loc['Day6', :] = [45,55]
df.loc['Day7', :] = [46,56]
df.columns = ['G','A']
df2 = pd.DataFrame(index=df.index,columns=df.columns)
# df2[(df < 55) & (df > 51)]=-1
df2[(df == 43) | (df == 55)] = 0
# print(
#     df2 )
# print(df2.loc['Day4'])
df3 = df.copy()
print(df3)
# df3.loc[:,:]=0
# print(df3)
df3 = df2.copy()

print(df3)


