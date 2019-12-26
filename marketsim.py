"""MC2-P1: Market simulator.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Shantanu Singh
GT User ID: ssingh341@gatech.edu
GT ID:
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
# from util import get_data, plot_data
import warnings
import math

# author: Shantanu Singh


def author():
    return 'ssingh341'  # replace tb34 with your Georgia Tech username.


def compute_portvals(df_trade, start_val=1000000, commission=9.95, impact=0.005):
    start_date = df_trade.index[0]
    end_date = df_trade.index[-1]
    symbol_ = df_trade.columns[0]
    # df_prices = get_data(
    #     [symbol_], pd.date_range(start_date, end_date))
    df_prices.fillna('ffill')
    df_prices.fillna('bfill')
    df_prices.drop(['SPY'], axis=1, inplace=True)
    cash = [1.0 for i in range(df_prices.shape[0])]
    # ***Creating df_prices****
    df_prices['cash'] = cash
    df_prices[['cash']] = 1
    df_trade['cash'] = 0.0
    for ind in df_trade.index:
        price_i = df_prices.loc[ind, symbol_]
        holding = df_trade.loc[ind, symbol_]
        if holding > 0:
            df_trade.loc[ind, 'cash'] = -holding * \
                price_i*(1.0+impact) - commission
        elif holding < 0:
            df_trade.loc[ind, 'cash'] = -holding * \
                price_i*(1.0-impact) - commission
    # ***Creating df_holding***
    df_holding = df_trade.copy()
    df_holding.iloc[0, -1] = start_val + df_trade.iloc[0, -1]
    for i in range(1, df_holding.shape[0]):
        df_holding.iloc[i] = df_holding.iloc[i-1] + df_holding.iloc[i]
    # ***Creating  df_value***
    df_value = df_prices*df_holding
    df_portVal = df_value.sum(axis=1)

    return df_portVal


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    port_val = compute_portvals(orders_file=of, start_val=sv)
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
    end_date = port_val.index[0]
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [
    #     0.2, 0.01, 0.02, 1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [
        0.2, 0.01, 0.02, 1.5]

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {port_val[-1]}")


if __name__ == "__main__":
    test_code()
