from services.strategies import *


def project_function(periodReturns, periodFactRet):
    """
    Please feel free to modify this function as desired
    :param periodReturns:
    :param periodFactRet:
    :return: the allocation as a vector
    """

    # Initialize BestOptimization Class called stratagy
    Strategy = MixedStrategy(NumObs=36, alpha=0.95, k=0, gamma=0.1)

    # Executre stratagy and obtain optimal portfolio weights
    x = Strategy.execute_strategy(periodReturns, periodFactRet)
    return x

# Called model for Grid Search Process
def grid_search_function(periodReturns, periodFactRet, x0, num_periods, alpha, lamda):
    """
    num_periods: Number of months to calibrate on
    alpha: Significance level
    lambda: Risk Averse Parameter
    """

    # Initialize Grid search class
    Strategy = grid_search(num_periods, x0)

    # Executre Grid Search Stratagy that returns porfolio weights
    x = Strategy.execute_grid(periodReturns, periodFactRet, alpha, lamda)
    return x