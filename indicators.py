import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from util import get_
# data, plot_data

# author: Shantanu Singh


def author():
    return 'ssingh341'


def technicalIndicators(df_prices):
    df_price = df_prices.copy()
    df_price = df_price/df_price.iloc[0]
    # print(df_price)
    window_ = 20
    # sma_plot = plt.figure(figsize=(12, 8))
    # plt.title('Rolling mean')
    # print(df_price[symbol])
    rm_symbol = df_price.rolling(window_).mean()
    # print(rm_symbol)
    # print(rm_symbol)
    # print(df_price.loc[])
    sma_ratio = df_price/rm_symbol - 1
    # print("SMA_ratio", sma_ratio)
    # print("SMA_ratio mean", sma_ratio.mean())

    # print("SMA_ratio std", sma_ratio.std())

    sma_ratio_norm = -1 + 2*(sma_ratio - sma_ratio.min()) / \
        (sma_ratio.max()-sma_ratio.min())
    # plt.plot(df_price, 'b', label='Price')
    # plt.plot(rm_symbol, 'green', label='SMA')
    # plt.plot(sma_ratio_norm, 'red', label='Price/SMA')
    # plt.xlabel('Date')
    # plt.ylabel('Normalized Value')
    # plt.legend()
    # plt.ylim(-1, 2)

    # sma_plot.savefig("sma")
    # plt.show()
    # plt.close()
    rstd_symbol = df_price.rolling(window_).std()
    upperBand = rm_symbol + rstd_symbol*2
    lowerBand = rm_symbol - rstd_symbol*2
    bb_value = (df_price - rm_symbol)/(2*rstd_symbol)
    # bb_plot = plt.figure(figsize=(12, 8))
    # plt.title('Bollinger Bands')
    # plt.plot(df_price, 'b', label='Price')
    # plt.plot(upperBand, 'r', label='bollinger bands')
    # plt.plot(lowerBand, 'r')
    # plt.plot(bb_value, 'green')
    # plt.plot(rm_symbol, 'green', label='SMA')
    # plt.xlabel('Date')
    # plt.ylabel('Normalized Value')
    # plt.legend()
    # bb_plot.savefig('Bollinger_Bands.png')
    # plt.axhline(y=-1, linestyle='--')
    # plt.axhline(y=1, linestyle='--')
    # plt.show()
    # plt.close()
    # print(df_price.shift(window_))
    momentum = df_price/df_price.shift(window_) - 1
    # momentum.columns = ['momentum']
    # ax = momentum.plot(grid=True, title='Momentum',
    #                    use_index=True)
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Momentum')
    # fig = ax.get_figure()
    # fig.savefig('Momentum.png')

    exp12 = df_price.ewm(span=12, adjust=False).mean()
    exp26 = df_price.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    # macd_plot = plt.figure(figsize=(12, 8))
    # plt.title('MACD')
    # # plt.plot(df_price, 'b', label='Price')
    # plt.plot(macd, 'g', label='MACD')
    # plt.plot(macd_signal, 'r', label='MACD signal line')
    # plt.xlabel('Date')
    # plt.ylabel('Normalized Value')
    # plt.legend()
    # # plt.show()
    # macd_plot.savefig('MACD.png')
    # plt.close()
    # RSI
    n = 14
    rsi = df_price.copy()
    rsi.ix[:,:] = 0
    daily_rets = df_price - df_price.shift(1)
    up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()

    up_gain = df_price.copy()
    up_gain.iloc[:, :] = 0
    up_gain.values[n:, :] = up_rets.values[n:, :] - up_rets.values[:-n, :]

    down_loss = df_price.copy()
    down_loss.iloc[:, :] = 0
    down_loss.values[n:, :] = down_rets.values[n:, :] - \
        down_rets.values[:-n, :]

    for day in range(df_price.shape[0]):
        up = up_gain.ix[day, :]
        down = down_loss.ix[day, :]
        rs = up/down
        rsi.ix[day, :]=100 - (100/(1 + rs))
    rsi[rsi == np.inf]=100

    # deltas = np.diff(df_price)
    # seed = deltas[:n+1]
    # up = seed[seed >= 0].sum()/n
    # down = -seed[seed < 0].sum()/n
    # rs = up/down
    # rsi = np.zeros_like(df_price)
    # rsi[:n] = 100. - 100./(1.+rs)

    # for i in range(n, len(df_price)):
    #     delta = deltas[i-1]  # cause the diff is 1 shorter
    #     for j in range(delta.size):

    #         if delta[j] > 0:
    #             upval = delta[j]
    #             downval = 0.
    #         else:
    #             upval = 0.
    #             downval = -delta[j]

    #         up = (up*(n-1) + upval)/n
    #         down = (down*(n-1) + downval)/n

    #         rs = up/down
    #         rsi[i, j] = 100. - 100./(1.+rs)
    df_rsi=pd.DataFrame(rsi, index = df_price.index)
    # rsi_plot = plt.figure(figsize=(12, 8))
    # plt.title('Relative Stregth Index')
    # # plt.plot(df_price, 'b', label='Price')
    # plt.plot(df_rsi, 'g', label='RSI')
    # plt.axhline(y=70, linestyle='--')
    # plt.axhline(y=30, linestyle='--')
    # plt.xlabel('Date')
    # plt.ylabel('RSI Value')
    # plt.legend()
    # # plt.show()
    # rsi_plot.savefig('RSI.png')
    # plt.close()
    # df_indicators = pd.DataFrame(
    #     index=df_price.index, columns=['MACD', 'MACD signalLine', 'BB', 'RSI'])
    # df_indicators['MACD'] = macd
    # df_indicators['MACD signalLine'] = macd_signal
    # df_indicators['BB'] = bb_value
    # df_indicators['RSI'] = df_rsi
    # print(bb_value)
    return macd, macd_signal, bb_value, df_rsi, momentum, rm_symbol


def testCode():
    start_date='2008-01-01'
    end_date='2009-12-31'
    symbol='JPM'
    # df_price = get_data([symbol], pd.date_range(start_date, end_date))
    df_price.fillna('ffill')
    df_price.fillna('bfill')
    df_price.drop(['SPY'], axis = 1, inplace = True)
    technicalIndicators(df_price, symbol)


if __name__ == "__main__":
    testCode()
