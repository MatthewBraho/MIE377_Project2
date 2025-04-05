import numpy as np
from services.estimators import *
from services.optimization import *


# this file will produce portfolios as outputs from data - the strategies can be implemented as classes or functions
# if the strategies have parameters then it probably makes sense to define them as a class


def equal_weight(periodReturns):
    """
    computes the equal weight vector as the portfolio
    :param periodReturns:
    :return:x
    """
    T, n = periodReturns.shape
    x = (1 / n) * np.ones([n])
    return x


class HistoricalMeanVarianceOptimization:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns=None):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param periodReturns:
        :param factorReturns:
        :return: x
        """
        factorReturns = None  # we are not using the factor returns
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        print(len(returns))
        mu = np.expand_dims(returns.mean(axis=0).values, axis=1)
        Q = returns.cov().values
        x = MVO(mu, Q)

        return x


class OLS_MVO:
    """
    uses historical returns to estimate the covariance matrix and expected return
    """

    def __init__(self, NumObs=36):
        self.NumObs = NumObs  # number of observations to use

    def execute_strategy(self, periodReturns, factorReturns):
        """
        executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns:
        :param periodReturns:
        :return:x
        """
        T, n = periodReturns.shape
        # get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]
        mu, Q = OLS(returns, factRet)
        x = MVO(mu, Q)
        return x


class BestOptimization:
    """
    Uses Ridge Cross Validation to get a factor model to predict asset returns and cov matrix
    Then Robust MVO with a risk adjusted return objective function to determine portfolio weigting
    """

    def __init__(self, NumObs=30, rf=0):
        # number of months of data to use for calibration
        self.NumObs = NumObs  


        # Intialized to 0 but could be useful idea
        self.rf = rf

    def execute_strategy(self, periodReturns, factorReturns):
        """
        Executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns: DataFrame with factor returns for each period
        :param periodReturns: DataFrame with period returns for each asset
        :return: x (optimized portfolio allocation)
        """
        T, n = periodReturns.shape

        # Get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]

        # Get factor-based return estimates and covariance
        mu, Q= LASSO_CV(returns, factRet)

        # Solve Robust MVO, takes in alpha, lambda and gamma paramters (explained optimization.py)
        x = RiskParityOptimization(mu, Q, T, 0.85, 2)
        return x
    

class grid_search:
    """
    Executes the portfolio allocation strategy based on the parameters in the __init__
    Used to pass in different parameters which is helpful in grid search algorithm
    """

    def __init__(self, NumObs=36, w_prev=None, rf=0):
        # number of months of data to use for calibration
        self.NumObs = NumObs  
        
        # Previous weights
        self.w_prev = w_prev

        # Intialized to 0 but could be useful idea
        self.rf = rf

    def execute_grid(self, periodReturns, factorReturns, alpha, lamda):
        """
        Executes the portfolio allocation strategy based on the parameters in the __init__

        :param factorReturns: DataFrame with factor returns for each period
        :param periodReturns: DataFrame with period returns for each asset
        :alpha: Adjustable signifcance level alpha used in Robust MVO
        :lambda: Adjustable risk averse parameter lambda in Robust MVO (risk adjusted varaince objective)
        :return: x (optimized portfolio allocation)
        """
        T, n = periodReturns.shape

        # Get the last T observations
        returns = periodReturns.iloc[(-1) * self.NumObs:, :]
        factRet = factorReturns.iloc[(-1) * self.NumObs:, :]

        # Get factor-based return estimates and covariance
        mu, Q= LASSO_CV(returns, factRet)

        # Solve Robust MVO (Keep gamma at 0.1 found to be optimal)
        x = Robust_MVO(mu, Q, T, alpha, lamda, 0.1, self.w_prev)

        return x