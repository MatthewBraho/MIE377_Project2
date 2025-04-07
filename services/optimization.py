import cvxpy as cp
import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize

installed = cp.installed_solvers()

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

import cvxpy as cp
import numpy as np
from scipy.stats import chi2

def Robust_MVO(mu, Q, N=250, alpha=0.95, lambda_reg=17, rf=0.00):
    '''
    Robust MVO Optimization:
    Objective: Maximize excess return minus robustness penalty and regularization term.
    Subject to: weights sum to 1, non-negative, and select at most 10 assets.

    :param mu: Expected returns vector (numpy array)
    :param Q: Covariance matrix (numpy array)
    :param N: Number of observations (e.g., months)
    :param alpha: Confidence level for robustness
    :param lambda_reg: Regularization weight
    :param rf: Risk-free rate
    :return: Optimal portfolio weights or None if no optimal solution is found.
    '''

    n = len(mu)
    
    # Define decision variables:
    # w: continuous portfolio weights.
    # y: binary asset selection variable.
    w = cp.Variable(n)
    y = cp.Variable(n)

    # Robustness parameters:
    theta = np.sqrt((1 / N) * np.diag(Q))
    epsilon = np.sqrt(chi2.ppf(alpha, n))

    # Objective: maximize excess return minus a robustness penalty and regularization term.
    objective = cp.Maximize(
        (mu - rf).T @ w
        - epsilon * cp.norm(cp.multiply(theta, w), 2)
        - lambda_reg * cp.quad_form(w, Q)
    )

    # Constraints:
    # 1. Weights must sum to 1.
    # 2. Weights must be non-negative.
    # 3. Linking constraint: if an asset is not selected (y[i] = 0), then w[i] must be 0.
    # 4. At most 10 assets may be selected.
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
    ]

    # Define and solve the problem using a mixed-integer solver (e.g., GUROBI).
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI, verbose=False)

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

from scipy import optimize 
from scipy.optimize import minimize

def risk_parity_optimization(mu, Q, c=1):
    """
    Solve the convex reformulation of the risk parity problem:
    
        minimize   0.5 * y^T * Q * y  -  c * sum(log(y))
        subject to y >= 0

    and recover x* = y* / sum(y*).

    Parameters
    ----------
    mu : array-like of shape (n,)
        Not directly used in this formulation, but included to match the signature
        that may be used in broader portfolio optimization routines.
    Q : 2D array-like of shape (n, n)
        Covariance matrix (must be positive semidefinite).
    c : float, optional (default=1.0)
        Positive scalar controlling the strength of the log-barrier term.

    Returns
    -------
    x_star : np.ndarray of shape (n,)
        The normalized portfolio weights that solve the risk parity problem.
    """

    # Ensure inputs are NumPy arrays
    mu = np.asarray(mu)
    Q = np.asarray(Q)

    n = len(mu)
    
    # Define the optimization variable
    y = cp.Variable(n, nonneg=True)
    
    # Define the objective: 0.5 * y^T Q y - c * sum(log(y))
    # Note: Q should be PSD to ensure convexity
    objective = 0.5 * cp.quad_form(y, Q) - c * cp.sum(cp.log(y))
    
    # Form and solve the problem
    prob = cp.Problem(cp.Minimize(objective))
    prob.solve(solver=cp.SCS)  # or another solver that supports log() (e.g., ECOS, MOSEK)
    
    # Retrieve the solution for y
    y_opt = y.value
    
    # Convert y_opt to weights x* by normalizing
    x_star = y_opt / np.sum(y_opt)
    
    return x_star

def RiskParity1(Q, n):

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

def RiskParity_turnover2(Q, n, lamda = 0.5):
    y = cp.Variable(n)

    objective = cp.Minimize(
        0.5 * cp.quad_form(y, Q) - cp.sum(cp.log(y)) + lamda * cp.norm2(y - cp.mean(y) * np.ones(n)))

    constraints = [
        y >= 0
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS_BB, verbose=False)

    if prob.status != cp.OPTIMAL:
        return None

    return y.value / sum(y.value)

def Sharpe_turnover(mu, Q, n, lamda):
    y = cp.Variable(n)
    k = cp.Variable(1)

    objective = cp.Minimize(cp.quad_form(y,Q) + lamda*cp.norm2(y-cp.mean(y)-np.ones(n)))

    constraints = [
        (mu).T@y == 1,
        y >= 0,
        k >= 0,
        cp.sum(y) == k
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return None
    
    return y.value/sum(y.value)

def Sharpe(mu, Q, n):
    y = cp.Variable(n)
    k = cp.Variable(1)

    objective = cp.Minimize(cp.quad_form(y,Q))

    constraints = [
        (mu).T@y == 1,
        y >= 0,
        k >= 0,
        cp.sum(y) == k
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return None
    
    return y.value/sum(y.value)