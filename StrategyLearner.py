"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

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

Student Name: Tucker Balch (replace with your name)
GT User ID: tb34 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import util as ut
import random
import indicators as ind
import numpy as np
import math
import QLearner as ql
# import marketsim as ms
from ManualStrategy import ManualStrategy
import marketsim as msim
import time
import random as rand


class StrategyLearner(object):

    # author: Shantanu Singh
    def author():
        return 'ssingh341'  # replace tb34 with your Georgia Tech username.
    # constructor

    def __init__(self, verbose=False, commission=0.0,impact=0.0):
        rand.seed(903037575)
        self.verbose = verbose
        self.impact = impact
        self.discretization_matrix = []
        self.learner = ql.QLearner(num_states=1000, num_actions=3, alpha=0.2, gamma=0.9,
                                   rar=0.5, radr=0.99, dyna=0, verbose=False)
        self.holding = 0
        self.impact = impact
        self.commission = commission
    # Check convergence of port_val

    def check_convergence(self, port_val, port_val_copy):
        num = 0.0
        den = 0.0
        residual = 0.0
        # print(port_val)
        for i in range(port_val.shape[0]):
            v_Tplus1 = port_val.iloc[i, 0]
            v_T = port_val_copy.iloc[i, 0]
            num += (v_Tplus1 - v_T)**2
            den += v_T**2
        residual = math.sqrt(num/den)
        if residual < 1e-10:
            return True
        else:
            return False

    # Discretizer
    def discretize(self, tech_indicators):
        steps = 10
        # series len(data)/steps also correct
        stepsize = math.ceil(tech_indicators.shape[0]/steps)
        for col in range(tech_indicators.shape[1]):
            data = tech_indicators.iloc[:, col]
            data = data.sort_values()
            threshold = np.zeros(steps - 1)
            for i in range(0, steps - 1):
                threshold[i] = data[(i+1)*stepsize-1]
            self.discretization_matrix.append(threshold)

    def discretize_value(self, X):
        X = np.array(X)  # X should be array
        lenX = len(X)
        value = 0
        for i in range(lenX):
            x = X[i]
            bins = self.discretization_matrix[i]
            value += np.digitize(x, bins) * 10**(lenX - i-1)
        return value
    # move the trader to next state and report reward

    # def movetrader(self, df_trade, a, holding, cash_value, index_, index_i, symbol, df_price,
    #                df_holding, port_val):
    def movetrader(self, a, holding, cash_value, index_, index_i, symbol, df_price,
                   port_val):
                   # to reduce TIME we can completely remove the frames
        price_i = df_price.iloc[index_i, 0]
        h = 0
        c_value = 0
        if a == 1:
            if holding == 0 or holding == -1000:
                h = 1000
                c_value = -1000.0 * price_i * \
                    (1.0 + self.impact) - self.commission
                # df_trade.loc[index_] = [h, c_value]
                holding += 1000
        elif a == 2:
            if holding == 1000 or holding == 0:
                h = -1000
                c_value = 1000.0 * price_i\
                    * (1.0-self.impact) - self.commission
                # df_trade.loc[index_] = [h, c_value]
                holding -= 1000
        # cash_value += df_trade.iloc[index_i, 1]
        cash_value += c_value

        # df_holding.iloc[index_i] = [holding, cash_value]
        p = holding * price_i + cash_value
        if index_i != 0:
            reward = p/port_val - 1.0
        else:
            reward = 0
        port_val = p
        return holding, cash_value, reward,port_val

    # move the trader to next state for testPolicy

    def movetrader_testPolicy(self, df_trade, a, holding, cash_value, index_, index_i, df_price, port_val):
                   # to reduce TIME we can completely remove the frames
        price_i = df_price.iloc[index_i, 0]
        h = 0
        c_value = 0
        if a == 1:
            if holding == 0 or holding == -1000:
                h = 1000
                c_value = -1000.0 * price_i * \
                    (1.0 + self.impact) - self.commission
                df_trade.loc[index_] = [h, c_value]
                holding += 1000
        elif a == 2:
            if holding == 1000 or holding == 0:
                h = -1000
                c_value = 1000.0 * price_i\
                    * (1.0-self.impact) - self.commission
                df_trade.loc[index_] = [h, c_value]
                holding -= 1000
        cash_value += c_value

        p = holding * price_i + cash_value
        port_val.loc[index_] = [p]
        return holding, cash_value

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol="IBM",
                    sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 1, 1),
                    sv=10000):

        # add your code to do learning here
        # Adjusting lookback period
        lookback_window = 35
        extra_days = math.ceil(lookback_window/7)*2
        lookback_window += extra_days
        adjustedStartDay = sd - dt.timedelta(days=lookback_window)
        # example usage of the old backward compatible util function
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        dates_adj = pd.date_range(adjustedStartDay, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices_all_adj = ut.get_data(syms, dates_adj)  # automatically adds SPY
        prices_all.fillna('ffill', inplace=True)
        prices_all.fillna('bfill', inplace=True)
        prices_all_adj.fillna('ffill', inplace=True)
        prices_all_adj.fillna('bfill', inplace=True)
        prices = pd.DataFrame(prices_all[syms])  # only portfolio symbols
        # only SPY, for comparison later
        prices_SPY = pd.DataFrame(prices_all['SPY'])
        # only portfolio symbols
        prices_adj = pd.DataFrame(prices_all_adj[syms])
        # only SPY, for comparison later
        prices_SPY_adj = pd.DataFrame(prices_all_adj['SPY'])
        # if self.verbose:
        #     print(prices)

        macd, macd_signal, bb_value, rsi, momentum, sma = ind.technicalIndicators(
            prices_adj, symbol)
        tech_indicators_all = [macd, macd_signal, bb_value, rsi, momentum, sma]
        tech_indicators = [bb_value, rsi, momentum]
        # filling NAN values in indicators
        tech_indicators = pd.concat(tech_indicators, axis=1)
        tech_indicators.columns = ['bb_value', 'rsi', 'momentum']
        # extracting the relevant time frame00
        tech_indicators = tech_indicators.loc[prices.index[0]:]
        # Initialized df frame
        port_val1 = pd.DataFrame(index=prices.index, columns=['port_val'])
        port_val1.fillna(0, inplace=True)
        # Discretizing bins of indicators
        self.discretize(tech_indicators)
        X_start = self.discretize_value(
            tech_indicators.loc[tech_indicators.index[0]])
        isConverged = False
        count = 0
        scores = []
        # Trade will happen on first day also
        while not isConverged:
            # Clearing the dataframes
            for col in port_val1.columns:
                port_val1[col].values[:] = 0
            # for col in df_trade.columns:
            #     df_trade[col].values[:] = 0
            # for col in df_holding.columns:
            #     df_holding[col].values[:] = 0
            total_reward = 0
            X = X_start
            # set the state and get first action
            action = self.learner.querysetstate(X)
            holding = 0
            cash_val = sv
            position = 0
            port_val = 0
            p_val = np.zeros(prices.shape[0])
            for i in prices.index:
                # move to new location according to action and then get a new action
                # holding, cash_val, stepreward = self.movetrader(df_trade,
                #                                                 action, holding, cash_val, i, position, symbol, prices,
                #                                                 df_holding, port_val)
                holding, cash_val, stepreward,port_val = self.movetrader(action, holding, cash_val, i, position, symbol, prices,
                                                                port_val)
                p_val[position] = port_val
                if i == prices.index[-1]:
                    continue
                state = self.discretize_value(
                    tech_indicators.iloc[position + 1])
                action = self.learner.query(state, stepreward)
                # if verbose: time.sleep(1)
                total_reward += stepreward
                position += 1
            port_val1['port_val']=p_val
            count = count + 1
            scores.append(total_reward)
            # Covergence criterion
            if count > 20 and count < 1000:
                isConverged = self.check_convergence(port_val1, port_val_copy)
                if isConverged:
                    if self.verbose:
                        print("Result Converged!!!")
                        print("No of iterations = ", count)
            elif count == 1000:
                if self.verbose:
                    print("timeout")
                break
            port_val_copy = port_val1.copy()
        # port_val.to_csv('without Frames42.csv')
        # Check if the port_val are almost constant
        cum_ret = 100*(port_val1.iloc[-1, 0]/port_val1.iloc[0, 0] - 1.0)
        # print('Cumulative Return for training data of ',
        # symbol, ' = ', cum_ret)
        if self.verbose:
            cum_ret = 100*(port_val1.iloc[-1, 0]/port_val1.iloc[0, 0] - 1.0)
            print('Cumulative Return for training data of ',
                  symbol, ' = ', cum_ret)
            s=np.array(scores)
            print('Median of scores = ', np.median(s))
        # plt.plot(port_val1/port_val1.iloc[0, 0],
        #          'r', label = 'Training portfolio')
        # example use with new colname
        # automatically adds SPY
        # volume_all = ut.get_data(syms, dates, colname="Volume")
        # volume = volume_all[syms]  # only portfolio symbols
        # volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        # if self.verbose:
        #     print(volume)

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM",
                   sd = dt.datetime(2009, 1, 1),
                   ed = dt.datetime(2010, 1, 1),
                   sv = 10000):
        # add your code to do learning here
        # Adjusting lookback period
        lookback_window=35
        extra_days=math.ceil(lookback_window/7)*2
        lookback_window += extra_days
        adjustedStartDay=sd - dt.timedelta(days = lookback_window)
        # example usage of the old backward compatible util function
        syms=[symbol]
        dates=pd.date_range(sd, ed)
        dates_adj=pd.date_range(adjustedStartDay, ed)
        prices_all=ut.get_data(syms, dates)  # automatically adds SPY
        prices_all_adj=ut.get_data(syms, dates_adj)  # automatically adds SPY
        prices_all.fillna('ffill', inplace = True)
        prices_all.fillna('bfill', inplace = True)
        prices_all_adj.fillna('ffill', inplace = True)
        prices_all_adj.fillna('bfill', inplace = True)
        prices=pd.DataFrame(prices_all[syms])  # only portfolio symbols
        # only SPY, for comparison later
        prices_SPY=pd.DataFrame(prices_all['SPY'])
        # only portfolio symbols
        prices_adj=pd.DataFrame(prices_all_adj[syms])
        # only SPY, for comparison later
        prices_SPY_adj=pd.DataFrame(prices_all_adj['SPY'])
        # if self.verbose:
        #     print(prices)

        macd, macd_signal, bb_value, rsi, momentum, sma=ind.technicalIndicators(
            prices_adj, symbol)
        tech_indicators_all=[macd, macd_signal, bb_value, rsi, momentum, sma]
        tech_indicators=[bb_value, rsi, momentum]
        # filling NAN values in indicators
        tech_indicators=pd.concat(tech_indicators, axis = 1)
        tech_indicators.columns=['bb_value', 'rsi', 'momentum']
        # extracting the relevant time frame00
        tech_indicators=tech_indicators.loc[prices.index[0]:]
        # Initialized df frame
        df_trade = pd.DataFrame(index = prices.index, columns = [symbol, 'cash'])
        df_trade.fillna(0, inplace = True)
        df_holding = pd.DataFrame(index = prices.index, columns = [symbol, 'cash'])
        df_holding.fillna(0, inplace = True)
        df_holding.iloc[0, -1]=sv
        df_value = pd.DataFrame(index = prices.index, columns = [symbol, 'cash'])
        df_value.fillna(0, inplace = True)
        port_val = pd.DataFrame(index = prices.index, columns = ['port_val'])
        port_val.fillna(0, inplace = True)
        holding=0
        cash_val=sv
        numeric_index=0
        # states_df = pd.DataFrame(index = prices.index, columns = ['states'])
        for i in prices.index:
            state=self.discretize_value(
                tech_indicators.loc[i])
            # states_df.loc[i] =[state]
            action=self.learner.querysetstate(state)
            holding, cash_val=self.movetrader_testPolicy(df_trade,
                                                           action, holding, cash_val, i, numeric_index, prices, port_val)
            numeric_index += 1
        df_trade.drop('cash', axis = 1, inplace = True)
        cum_ret=100*(port_val.iloc[-1, 0]/port_val.iloc[0, 0] - 1.0)
        # print('Cumulative Return for Testing data of ', symbol, ' = ', cum_ret," with impact = ",self.impact)
        if self.verbose:
            print(type(df_trade))  # it better be a DataFrame!
        if self.verbose:
            print(df_trade)
        if self.verbose:
            print(prices_all)
        # plt.plot(port_val/port_val.iloc[0, 0],
        #          'g', label = 'Testing portfolio')
        # plt.plot(states_df,
        #     'b', label = 'Testing portfolio')           
        return df_trade


if __name__ == "__main__":
    test_symbol = 'JPM'
    start=time.time()
    learner=StrategyLearner(verbose = False, commission=9.95,impact = 0.005)  # constructor
    learner.addEvidence(symbol = test_symbol, sd = dt.datetime(2008,1,1),\
         ed = dt.datetime(2009,12,31), sv = 100000) # training phase
    df_trades = learner.testPolicy(symbol = test_symbol, sd = dt.datetime(2008,1,1),\
         ed = dt.datetime(2009,12,31), sv = 100000) # testing phase
    port_val_ML = msim.compute_portvals(
        df_trades, start_val=100000, commission=9.95, impact=0.005)
    port_val_ML =  port_val_ML/port_val_ML.iloc[0]
    # PortVal for manual strategy
    mst = ManualStrategy
    df_trades = mst.testPolicy(test_symbol, sd=dt.datetime(2008,1,1),\
        ed=dt.datetime(2009,12,31), sv=100000)
    port_val_MS = msim.compute_portvals(
    df_trades, start_val=100000, commission=9.95, impact=0.005)  
    port_val_MS =  port_val_MS/port_val_MS.iloc[0]
    #PortVal for benchmark
    df_trades = pd.DataFrame(index=port_val_ML.index,columns=[test_symbol])
    df_trades.fillna(0, inplace = True)
    df_trades.iloc[0,0]=1000
    port_val_benchmark = msim.compute_portvals(
    df_trades, start_val=100000, commission=9.95, impact=0.005) 
    port_val_benchmark =  port_val_benchmark/port_val_benchmark.iloc[0]
    comparison_chart = pd.concat([port_val_ML, port_val_MS,port_val_benchmark], axis=1)
    comparison_chart.columns = ['ML Strategy', 'Manual Strategy','Benchmark']
    comparison_chart.plot(grid=True, title='Comparison of Strategy Learner with Manual strategy', use_index=True, color=['Black', 'Red', 'Green'])
    end=time.time()
    # plt.show()
    print('Execution time = ', (end-start))
    print("One does not simply think up a strategy")
