import math
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
# from util import get_data, plot_data
import indicators as ind
import marketsim as msim
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import time


class ManualStrategy:
    def __init__(self):
        self.order_type_buy = []
        self.order_type_sell = []

    def testPolicy(self, symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        window_ = 20

        lookback_window = 200
        # extra_days = math.ceil(lookback_window/7)*2
        # lookback_window += extra_days
        # adjustedStartDay = sd - dt.timedelta(days=lookback_window)
        # # example usage of the old backward compatible util function
        # syms = [symbol]
        # dates = pd.date_range(sd, ed)
        # dates_adj = pd.date_range(adjustedStartDay, ed)
        # prices_all = get_data(syms, dates)  # automatically adds SPY
        # prices_all_adj = get_data(syms, dates_adj)  # automatically adds SPY
        # prices_all.fillna('ffill', inplace=True)
        # prices_all.fillna('bfill', inplace=True)
        # prices_all_adj.fillna('ffill', inplace=True)
        # prices_all_adj.fillna('bfill', inplace=True)
        # df_prices = pd.DataFrame(prices_all[syms])  # only portfolio symbols
        # # only SPY, for comparison later
        # prices_SPY = pd.DataFrame(prices_all['SPY'])
        # # only portfolio symbols

        # df_prices_adj = pd.DataFrame(prices_all_adj[syms])
        # # only SPY, for comparison later
        # prices_SPY_adj = pd.DataFrame(prices_all_adj['SPY'])
    # Getting live stock data
        symbols = ["IBM", "MSFT", "AAPL"]
        isSymbolFirst = True
        api_key = 'D3YIZ9Q9T9COACJ4'
        ts = TimeSeries(key=api_key, output_format='pandas')
        count = 1
        while(count > 0):
            t0 = time.time()
            for sym in symbols:
                data, meta_data = ts.get_intraday(
                    symbol=sym, interval='1min', outputsize='full')
                data = data[['4. close']]
                if isSymbolFirst:
                    df_prices_adj = data
                    isSymbolFirst = False
                else:
                    df_prices_adj = pd.concat(
                        [df_prices_adj, data], axis=1)
            df_prices_adj.columns = symbols
            df_prices_adj = df_prices_adj[-lookback_window:-1]
            print(df_prices_adj)
            # data.to_csv(('per minute data.csv'))
            # time.sleep(60)

            macd, macd_signal, bb_value, rsi, momentum, sma = ind.technicalIndicators(
                df_prices_adj)
            t1 = time.time()
            elapsed_time = t1-t0
            print("Elapsed Time = ", elapsed_time)
        # df_prices_adj.index = df_prices.index[0]
        # s_date_minus1 = df_prices_adj.index[df_prices_adj.index.index(
        #     s_date) - 1]
        # s_date = df_prices.index[0]

        # # for i in df_prices_adj.index:
        # #     if i==
        # temp = np.where(df_prices_adj.index ==
        #                 np.datetime64(s_date))[0][0] - 1
        # s_date_minus1 = df_prices_adj.index[temp]
        # macd_signal = macd_signal.loc[s_date:]
        # macd = macd.loc[s_date:]
        # bb_value = bb_value.loc[s_date:]
        # rsi = rsi.loc[s_date_minus1:]
        # momentum = momentum.loc[s_date:]
        # sma = sma.loc[s_date:]

        # print(sma)
        # print(momentum)
        # print(rsi)
        # print(macd)
        # print(df_p)
            current_time = df_prices_adj.index[-1]
            prev_time = df_prices_adj.index[-2]
            holding = 1000
            order = [0 for i in range(df_prices.shape[0])]
            order[0] = 1000
            if count == 1:
                trade_orders = pd.DataFrame(columns=symbols)
            # trade_orders.fillna(0, inplace=True)
            for j in range(symbols.size()):
                if bb_value.iloc[i-1, j] <= -1 and bb_value.iloc[i, j] > -1:
                    if rsi.iloc[i-1, j] <= 30 and rsi.iloc[i, j] > 30:
                        if holding == 0 or holding == -1000:
                            trade_orders.loc[current_time, symbols[j]]
                            order[i] = 1000
                            holding += 1000
                # elif rsi.iloc[i-1] <= 30 and rsi.iloc[i] > 30:
                #     if holding == 0 or holding == -1000:
                #         order.append(1000)
                #         holding += 1000
                #     else:
                #         order.append(0)
                elif momentum.iloc[i-1] <= 0 and momentum.iloc[i] > 0:
                    if holding == 0 or holding == -1000:
                        order[i] = 1000
                        holding += 1000
                elif bb_value.iloc[i-1] >= 1 and bb_value.iloc[i] < 1:
                    if rsi.iloc[i-1] >= 70 and rsi.iloc[i] < 70:
                        if holding == 1000 or holding == 0:
                            order[i] = -1000
                            holding -= 1000
                # elif rsi.iloc[i-1] >= 70 and rsi.iloc[i] < 70:
                #     if holding == 1000 or holding == 0:
                #         order.append(-1000)
                #         holding -= 1000
                #     else:
                #         order.append(0)
                elif momentum.iloc[i-1] >= 0 and momentum.iloc[i] < 0:
                    if holding == 1000 or holding == 0:
                        order[i] = -1000
                        holding -= 1000
            trade_orders[symbol] = order
            count += 1

        # print(trade_orders)
        return trade_orders

# author: Shantanu Singh


def author():
    return 'ssingh341'


def calcPortfolioParameters(ms, port_val, lines=True, sample_type='insample'):
    if isinstance(port_val, pd.DataFrame):
        port_val = port_val[port_val.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    daily_rf = 0.0
    k = 252
    daily_return = (port_val/port_val.shift(1))-1
    daily_return.iloc[0] = 0
    daily_return = daily_return.iloc[1:]
    cum_ret = (port_val.iloc[-1]/port_val.iloc[0]) - 1
    avg_daily_ret = daily_return.mean()
    std_daily_ret = daily_return.std()
    sharpe_ratio = math.sqrt(k) * (daily_return -
                                   daily_rf).mean()/std_daily_ret
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = port_val.index[0]
    end_date = port_val.index[-1]

    # BenchMark
    lis = [1000]
    bm = pd.DataFrame(index=port_val.index, columns=['JPM'])
    for i in range(1, bm.shape[0]):
        lis.append(0)
    bm['JPM'] = lis
    port_val_bm = msim.compute_portvals(
        bm, start_val=100000, commission=9.95, impact=0.005)
    if isinstance(port_val_bm, pd.DataFrame):
        # just get the first column
        port_val_bm = port_val_bm[port_val_bm.columns[0]]
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    daily_rf = 0.0
    k = 252
    daily_return_bm = (port_val_bm/port_val_bm.shift(1))-1
    daily_return_bm.iloc[0] = 0
    daily_return_bm = daily_return_bm.iloc[1:]
    cum_ret_bm = (port_val_bm.iloc[-1]/port_val_bm.iloc[0]) - 1
    avg_daily_ret_bm = daily_return_bm.mean()
    std_daily_ret_bm = daily_return_bm.std()
    sharpe_ratio_bm = math.sqrt(k) * (daily_return_bm -
                                      daily_rf).mean()/std_daily_ret_bm

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of benchMark : {sharpe_ratio_bm}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of benchMark : {cum_ret_bm}")
    print()
    print(f"Standard Deviation of daily return  of Fund: {std_daily_ret}")
    print(
        f"Standard Deviation of daily return  of benchMark : {std_daily_ret_bm}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of benchMark : {avg_daily_ret_bm}")
    print()
    print(f"Final Portfolio Value: {port_val[-1]}")
    print(f"Final Portfolio Value benchMark: {port_val_bm[-1]}")

    if lines and sample_type == 'insample':
        port_graph1 = plt.figure(figsize=(12, 8))
        title_ = 'Portfolio comparison Manual Strategy'
    elif not lines and sample_type == 'insample':
        port_graph2 = plt.figure(figsize=(12, 8))
        title_ = 'Portfolio comparison Manual Strategy (insample)'
    elif not lines and sample_type == 'outsample':
        port_graph3 = plt.figure(figsize=(12, 8))
        title_ = 'Portfolio comparison Manual Strategy (outsample)'

    plt.title(title_)
    plt.plot(port_val/port_val[0], 'r', label='Optimal portfolio')
    plt.plot(port_val_bm/port_val_bm[0], 'g',
             label='Benchmark portfolio')
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.legend()
    if lines:
        ymin, ymax = plt.gca().get_ylim()
        plt.vlines(ms.order_type_buy, ymin, ymax, color='blue')
        plt.vlines(ms.order_type_sell, ymin, ymax, color='black')
    # plt.show()
    if lines and sample_type == 'insample':
        port_graph1.savefig('Manual Strategy')
    elif not lines and sample_type == 'insample':
        port_graph2.savefig('Manual Strategy (insample data)')
    elif not lines and sample_type == 'outsample':
        port_graph3.savefig('Manual Strategy (outsample data)')
    plt.close()


def testCode():
    ms = ManualStrategy()
    df_trades = ms.testPolicy('JPM', dt.datetime(2008, 1, 1),
                              dt.datetime(2009, 12, 31), 100000)
    # print(df_trades)
    port_val = msim.compute_portvals(
        df_trades, start_val=100000, commission=9.95, impact=0.005)
    calcPortfolioParameters(ms, port_val, lines=True, sample_type='insample')
    calcPortfolioParameters(ms, port_val, lines=False, sample_type='insample')
    df_trades = ms.testPolicy('JPM', dt.datetime(2010, 1, 1),
                              dt.datetime(2011, 12, 31), 100000)
    # print(df_trades)
    port_val = msim.compute_portvals(
        df_trades, start_val=100000, commission=9.95, impact=0.005)
    calcPortfolioParameters(ms, port_val, lines=False, sample_type='outsample')


if __name__ == "__main__":
    testCode()
