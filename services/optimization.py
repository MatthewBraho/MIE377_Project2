import cvxpy as cp
import numpy as np
from scipy.stats import chi2


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

def Robust_MVO(mu, Q, N=250, alpha=0.9, lambda_reg=20, gamma=0.1, w_prev=None, rf=0.00):
    '''
    This the Robust MVO optimization function. It has the following objective fucntion and constriants:

    objective: Maximize Sharpe Ratio - lambda * l2 norm - gamma * turnover
                        subject to: sum of weights = 1
    Turnover = Average turnover rate


    :param mu: Calculated Expected Return Vector of each asset
    :param Q: Caluclated covariance matrix between assets
    :param N: Number of Months in Data
    :return w.value: Optimal Portfolio Weights
    '''


    # Find the total number of assets
    n = len(mu)
    


    # If no previous portfolio weights are provided, initialize them as zeros

    # Check if previous portfolio weights already exist
    if w_prev is None:
        # If they do not exist then initialize them to 0's
        w_prev = np.zeros(n)  

    # Define optimization variables:

    #Portfolio Asset weights (decision variables), w
    w = cp.Variable(n)  

    # Auxiliary variable for turnover calculation, x
    x = cp.Variable(n)  

    # Compute the robustness adjustment coefficient theta (using diag of cov Matrix)
    theta = np.sqrt((1 / N) * np.diag(Q))

    # Calculate uncertainty adjustment (epsilon) using the Chi-square inverse distribution 
    # and the alpha provided as an input
    epsilon = np.sqrt(chi2.ppf(alpha, n))

    # Define the objective function (Maximize risk-adjusted return)
    objective = cp.Maximize(

        # Expected excess return (mu - risk-free rate)
        # - Robustness adjustment (norm-based uncertainty)
        # - Regularization penalty on risk or variance
        # - High Turnover penalty

        (mu - rf).T @ w  
        - epsilon * cp.norm(cp.multiply(theta, w), 2)  # 
        - lambda_reg * cp.quad_form(w, Q)  # 
        - gamma * cp.sum(x)  # 
    )

    # Define Problem Constraints
    constraints = [
        # Portfolio weghts add up to 1
        cp.sum(w) == 1,  

        #Turnover constraints to track changes in weightings
        x >= w - w_prev,
        x >= -(w - w_prev) 
    ]

    # Initialize optimization problem
    prob = cp.Problem(objective, constraints) 
    
    # Solve Problem
    prob.solve(verbose=False)
    
    # Check if the problem converged
    if prob.status != cp.OPTIMAL:
         # Return None if the problem did not converge and no solution was found
        return None 
    
    return w.value
