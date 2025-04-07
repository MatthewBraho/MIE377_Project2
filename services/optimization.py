import cvxpy as cp
import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize
import pandas as pd



def MVO(mu, Q):
    """
    #---------------------------------------------------------------------- Use this function to construct an example of a MVO portfolio.
    #
    # An example of an MVO implementation is given below. You can use this
    # version of MVO if you like, but feel free to modify this code as much
    # as you need to. You can also change the inputs and outputs to suit
    # your needs.

    # You may use quadprog, Gurobi, or any other optimizer you are familiar
    # with. Just be sure to include comments in your code.

    # *************** WRITE YOUR CODE HERE ***************
    #----------------------------------------------------------------------
    """

    # Find the total number of assets
    n = len(mu)

    # Set the target as the average expected return of all assets
    targetRet = np.mean(mu)

    # Disallow short sales
    lb = np.zeros(n)

    # Add the expected return constraint
    A = -1 * mu.T
    b = -1 * targetRet

    # constrain weights to sum to 1
    Aeq = np.ones([1, n])
    beq = 1

    # Define and solve using CVXPY
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q)),
                      [A @ x <= b,
                       Aeq @ x == beq,
                       x >= lb])
    prob.solve(verbose=False)
    return x.value

def Robust_MVO(mu, Q, N=250, alpha=0.95, lambda_reg=10, rf=0.00):
    '''
    Robust MVO Optimization:
    Objective: Maximize Sharpe Ratio - regularization - robustness penalty
    Subject to: weights sum to 1, non-negative, at most 10 assets

    :param mu: Expected returns vector
    :param Q: Covariance matrix
    :param N: Number of observations (e.g., months)
    :param alpha: Confidence level for robustness
    :param lambda_reg: Regularization weight
    :param rf: Risk-free rate
    :return: Optimal portfolio weights or None
    '''

    n = len(mu)
    
    w = cp.Variable(n)  # continuous weights
    y = cp.Variable(n, boolean=True)  # binary selection

    N = Q.shape[0]
    theta = np.sqrt((1 / N) * np.diag(Q))
    epsilon = np.sqrt(chi2.ppf(alpha, n))

    objective = cp.Maximize(
        (mu - rf).T @ w
        - epsilon * cp.norm(cp.multiply(theta, w), 2)
        - lambda_reg * cp.quad_form(w, Q)
    )
    # Constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= y,              # enforce y[i] = 1 if w[i] > 0
        cp.sum(y) <= 2     # select at most 10 assets
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    if prob.status != cp.OPTIMAL:
        return None

    return w.value


def RiskParity(Q, n):

    y = cp.Variable(n)

    objective = cp.Minimize(
        0.5 * cp.quad_form(y, Q) - cp.sum(cp.log(y))
    )

    constraints = [
        y >= 0
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    if prob.status != cp.OPTIMAL:
        return None

    return y.value / sum(y.value)


def RiskParity_turnover(Q, n, lamda):
    y = cp.Variable(n)

    objective = cp.Minimize(
        0.5 * cp.quad_form(y, Q)
        - cp.sum(cp.log(y))
        + lamda * cp.norm2(y - cp.mean(y) * np.ones(n))
    )

    constraints = [
        y >= 0
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS_BB, verbose=False)

    if prob.status != cp.OPTIMAL:
        return None
    
    return y.value / sum(y.value)



def Sharpe(mu, Q,n):
    y = cp.Variable(n)
    k = cp.Variable(1)


    objective = cp.Minimize(cp.quad_form(y,Q)  )
    
    constraints = [
        (mu).T@y  == 1, 
        y>= 0,
        k>=0,
        cp.sum(y) == k
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return None

    return y.value/sum(y.value)


def Sharpe_turnover(mu, Q,n, lamda, alpha = 0.9):

    y = cp.Variable(n)
    
    objective = cp.Minimize(cp.quad_form(y,Q) 
                            + lamda * cp.norm2(y - cp.mean(y) * np.ones(n))
                            )
    
    N = Q.shape[0]
    theta = np.sqrt((1 / N) * np.diag(Q))
    epsilon = np.sqrt(chi2.ppf(alpha, n))

    constraints = [
        (mu).T@y - epsilon * cp.norm(cp.multiply(theta, y), 2) >= 1, 
        y>= 0    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return Sharpe_turnover(mu, Q,n, lamda, alpha*0.9)

    return y.value/sum(y.value)


def mix_method(z, mu, Q, n):

    x_1 = Sharpe_turnover(mu, Q, n, 15)
    x_2 = RiskParity_turnover(Q, n, 0.5)

    x_avg = (1-z)*x_1 + z*x_2
    return x_avg


def grid_optimization(mu, Q, n, lamda_RP, lamda_S, z, model):
    
    # Iterate over time available to train 30, 36, 40
    # for all

    if model == 0:
        # Loop through lambda 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2
        return RiskParity_turnover(Q, n, lamda_RP)
    elif model == 1:
        # Loop through lambda 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2
        return Sharpe_turnover(mu, Q, n, lamda_S)
    elif model == 2:
        # Loop through lamda_RP 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2
        # Loop through lamda_S 10,12.5, 15, 17.5, 20
        # z = 0.25, 0.5, 0.75
        return mix_method(z, mu, Q, n)