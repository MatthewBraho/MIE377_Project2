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



def RiskParityOptimization(mu, Q, N=250, alpha=0.9, lambda_reg=20, rf=0.00):
    '''
    Risk Parity Optimization:
    Objective: Equalize risk contributions from all assets.

    :param mu: Expected returns vector (not used)
    :param Q: Covariance matrix (n x n)
    :param N: Number of periods (unused)
    :param alpha: Confidence level (unused)
    :param lambda_reg: Regularization (unused)
    :param rf: Risk-free rate (unused)
    :return: Optimal weights w (numpy array)
    '''
    
    n = len(Q)
    
    # Initial guess: equal weight
    w0 = np.ones(n) / n
    
    # Portfolio risk
    def portfolio_risk(w):
        return np.sqrt(w.T @ Q @ w)
    
    # Risk contribution of each asset
    def risk_contribution(w):
        sigma = portfolio_risk(w)
        marginal_contrib = Q @ w
        return w * marginal_contrib / sigma
    
    # Objective: sum of squared differences from average risk contribution
    def risk_parity_objective(w):
        rc = risk_contribution(w)
        avg_rc = np.mean(rc)
        return np.sum((rc - avg_rc)**2)
    
    # Constraints: weights sum to 1, non-negative (long-only)
    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    })
    
    bounds = [(0, 1) for _ in range(n)]
    
    result = minimize(risk_parity_objective, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    
    if not result.success:
        return None
    
    return result.x

import cvxpy as cp
import numpy as np

def SharpeRiskParityOptimization(mu, Q, rf=0.0, c=0.1):
    """
    Sharpe-Risk Parity Optimization:

    Implements a convex program that combines:
      1. A fixed excess return constraint: (mu - rf)^T y = 1,
      2. A variance minimization term: 0.5 * y^T Q y,
      3. A log-barrier risk parity penalty: - c * sum(log(y)).

    The optimization problem is:

        minimize   0.5 * y^T Q y - c * sum(log(y))
        subject to (mu - rf)^T y = 1,
                   y > 0.

    The solution y is then normalized so that the final portfolio weights sum to 1.

    Parameters:
      mu : numpy array (1D)
           Vector of expected returns.
      Q : numpy array (n x n)
          Covariance matrix.
      rf : float, optional
          Risk-free rate (default 0.0).
      c : float, optional
          Log-barrier penalty weight (default 0.1).

    Returns:
      weights : numpy array
          Optimal portfolio weights that sum to 1, or None if the problem is infeasible.
    """
    # Ensure mu is a 1D vector.
    mu = np.ravel(mu)
    n = len(mu)
    
    # Define the optimization variable y with positivity constraint.
    y = cp.Variable(n, pos=True)
    
    # Define the convex objective.
    objective = cp.Minimize(0.5 * cp.quad_form(y, Q) - c * cp.sum(cp.log(y)))
    
    # Constraint: fix portfolio excess return to 1.
    constraints = [(mu - rf).T @ y == 1]
    
    # Solve the convex problem.
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return None
    
    # Normalize the solution to produce weights summing to 1.
    y_opt = y.value
    weights = y_opt / np.sum(y_opt)
    return weights
