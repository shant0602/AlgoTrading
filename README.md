# AlgoTrading
The objective of the current algo trader is to devise an optimal trading strategy for a set of given stocks. To
devise this trading strategy, I have implemented the technique of model free reinforcement learning using Q
learner. The learner is trained by fetching it financial data of stocks, from which it learns to trade or not
to trade stock each minute. After this, the learner trades on out of sample data to predict
a portfolio which is then evaluated for performance.

# Future Work
Once the learner consistently performs well, then portfolio optimization can be done amongst a set of given stocks using optimization algorithms.
