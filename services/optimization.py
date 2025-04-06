import cvxpy as cp
import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize



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

def Robust_MVO(mu, Q, N=250, alpha=0.9, lambda_reg=20, rf=0.00):
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

    objective = cp.Minimize(
        0.5 * cp.quad_form(y, Q)
        + epsilon * cp.norm(cp.multiply(theta, y), 2)
        - cp.sum(cp.log(y))
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



def RiskParityRobust(Q, n):
    """
    Robust Risk Parity Optimization with Ellipsoidal Uncertainty.
    
    Parameters:
    - Q: Covariance matrix (numpy.ndarray)
    - n: Number of assets
    - rho: Size of ellipsoidal uncertainty set (default = 0.05)
    """
    y = cp.Variable(n)
    
    # Objective: nominal term + penalty for worst-case (robust) uncertainty
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