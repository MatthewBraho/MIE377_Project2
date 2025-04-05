import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge

def OLS(returns, factRet):
    # Use this function to perform a basic OLS regression with all factors.
    # You can modify this function (inputs, outputs and code) as much as
    # you need to.

    # *************** WRITE YOUR CODE HERE ***************
    # ----------------------------------------------------------------------

    # Number of observations and factors
    [T, p] = factRet.shape

    # Data matrix
    X = np.concatenate([np.ones([T, 1]), factRet.values], axis=1)

    # Regression coefficients
    B = np.linalg.solve(X.T @ X, X.T @ returns)

    # Separate B into alpha and betas
    a = B[0, :]
    V = B[1:, :]

    # Residual variance
    ep = returns - X @ B
    sigma_ep = 1 / (T - p - 1) * np.sum(ep.pow(2), axis=0)
    D = np.diag(sigma_ep)

    # Factor expected returns and covariance matrix
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)
    F = factRet.cov().values

    # Calculate the asset expected returns and covariance matrix
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar
    Q = V.T @ F @ V + D

    # Sometimes quadprog shows a warning if the covariance matrix is not
    # perfectly symmetric.
    Q = (Q + Q.T) / 2

    return mu, Q


def LASSO_CV(returns, factRet):
    '''
    Estimation function that estimates expected returns (mu) and the covariance matrix (Q) 
    using LASSO regression with cross validation.

    LASSO regression is similair to OLS but uses a L1 regularization penalty to select the most 
    relevant factors and set strongly correlated factor loadings to 0.

    :param returns: Asset return matrix (T x N) where T is the number of time periods 
                    and N is the number of assets.
    :param factRet: Factor return matrix (T x p) where p is the number of factors.
    :return mu: Estimated expected returns vecotr for each asset.
    :return Q: Estimated covariance matrix of all assets.
    '''

    # Number of observations (T) and number of assets (N)
    T, N = returns.shape  

    # Number of factors (p)
    _, p = factRet.shape  

    # Initializing outputs for factor model / loadings:
    # a: Intercept terms for each asset
    # V: Factor loadings (coefficients) for each asset
    # sigma_ep: Residual variance for each asset
    a = np.zeros(N)  
    V = np.zeros((p, N))  
    sigma_ep = np.zeros(N)  

    # Step 1: Use LASSO regression to fit returns using factor returns as predictors for each asset

    for i in range(N):
        # Get returns for asset i
        y = returns.iloc[:, i].values 

        # Set the factor return matrix to X
        X = factRet.values  

        # Use Lasso Regression with Cros Validation (5 folds or batches)
        lasso_cv = LassoCV(cv=5, random_state=0)  
        lasso_cv.fit(X, y)  

        # Obtain and save the interecept and factor loadings for asset i
        a[i] = lasso_cv.intercept_  
        V[:, i] = lasso_cv.coef_  

        # Compute residual errors
        ep = y - lasso_cv.predict(X)  

        # Estimate residual variance
        sigma_ep[i] = np.mean(ep**2)  

    # Step 2: Compute covariance and expected return using factor loadings

    # Create a diagonal matrix using sigma_ep
    D = np.diag(sigma_ep)  

    # Calculate factor mean returns
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)

    #Calculate Factor Covariance matrix
    F = factRet.cov().values 

    # Using the trained factor models for each asset, compute the expected return vector
    # contianing expected returns of all assets
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar  

    # Caluclate asset return covariance matrix using factor model
    Q = V.T @ F @ V + D  

    # Ensure covariance matrix is symmetric
    Q = (Q + Q.T) / 2  

    return mu, Q


def RIDGE_CV(returns, factRet, alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10]):
    '''
    Estimation function that estimates expected returns (mu) and the covariance matrix (Q) 
    using Ridge regression with cross-validation.

    Ridge regression is similar to OLS but includes an L2 regularization penalty, which helps 
    prevent overfitting and shrink coefficients toward zero while preserving all predictors.

    :param returns: Asset return matrix (T x N) where T is the number of time periods 
                    and N is the number of assets.
    :param factRet: Factor return matrix (T x p) where p is the number of factors.
    :param alphas: List of alpha values for RidgeCV to determine the best regularization strength.
    :return mu: Estimated expected returns vector for each asset.
    :return Q: Estimated covariance matrix of all assets.
    '''

    # Number of observations (T) and number of assets (N)
    T, N = returns.shape  

    # Number of factors (p)
    _, p = factRet.shape  

    # Initializing outputs for factor model / loadings:
    # a: Intercept terms for each asset
    # V: Factor loadings (coefficients) for each asset
    # sigma_ep: Residual variance for each asset
    a = np.zeros(N)  
    V = np.zeros((p, N))  
    sigma_ep = np.zeros(N)  

    # Step 1: Use Ridge regression to fit returns using factor returns as predictors for each asset
    
    for i in range(N):
        # Get returns for asset i
        y = returns.iloc[:, i].values 

        # Set the factor return matrix to X
        X = factRet.values  

        # Use Ridge Regression with Cross Validation over given alpha values
        ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)  
        ridge_cv.fit(X, y)  

        # Obtain and save the intercept and factor loadings for asset i
        a[i] = ridge_cv.intercept_  
        V[:, i] = ridge_cv.coef_  

        # Compute residual errors
        ep = y - ridge_cv.predict(X)  

        # Estimate residual variance
        sigma_ep[i] = np.mean(ep**2)  

    # Step 2: Compute covariance and expected return using factor loadings

    # Create a diagonal matrix using sigma_ep
    D = np.diag(sigma_ep)  

    # Calculate factor mean returns
    f_bar = np.expand_dims(factRet.mean(axis=0).values, 1)

    # Calculate Factor Covariance matrix
    F = factRet.cov().values 

    # Using the trained factor models for each asset, compute the expected return vector
    # containing expected returns of all assets
    mu = np.expand_dims(a, axis=1) + V.T @ f_bar  

    # Calculate asset return covariance matrix using factor model
    Q = V.T @ F @ V + D  

    # Ensure covariance matrix is symmetric
    Q = (Q + Q.T) / 2  

    return mu, Q


def BAYESIAN_RIDGE(returns, factRet):
    '''
    Estimation function that estimates expected returns (mu) and the covariance matrix (Q) 
    using Bayesian Ridge regression.

    Bayesian Ridge regression is similar to Ridge regression but treats alpha as a random variable 
    with a normal distributionon and the best valaue is based probablistically, leading to automatic 
    regularization based on the data.

    :param returns: Asset return matrix (T x N) where T is the number of time periods 
                    and N is the number of assets.
    :param factRet: Factor return matrix (T x p) where p is the number of factors.
    :return mu: Estimated expected returns vector for each asset.
    :return Q: Estimated covariance matrix of all assets.
    '''

    # Number of observations (T) and number of assets (N)
    T, N = returns.shape  

    # Number of factors (p)
    _, p = factRet.shape  

    # Initializing outputs for factor model / loadings:
    # a: Intercept terms for each asset
    # V: Factor loadings (coefficients) for each asset
    # sigma_ep: Residual variance for each asset
    a = np.zeros(N)  
    V = np.zeros((p, N))  
    sigma_ep = np.zeros(N)  

    # Step 1: Use Bayesian Ridge regression to fit returns using factor returns as predictors for each asset
    
    for i in range(N):
        # Get returns for asset i
        y = returns.iloc[:, i].to_numpy()  

        # Set the factor return matrix to X
        X = factRet.to_numpy()  

        # Use Bayesian Ridge Regression
        bayesian_ridge = BayesianRidge(fit_intercept=True)  
        bayesian_ridge.fit(X, y)  

        # Obtain and save the intercept and factor loadings for asset i
        a[i] = bayesian_ridge.intercept_  
        V[:, i] = bayesian_ridge.coef_  

        # Compute residual errors
        ep = y - bayesian_ridge.predict(X)  

        # Estimate residual variance
        sigma_ep[i] = np.mean(ep**2)  

    # Step 2: Compute covariance and expected return using factor loadings

    # Create a diagonal matrix using sigma_ep
    D = np.diag(sigma_ep)  

    # Calculate factor mean returns
    f_bar = factRet.mean(axis=0).to_numpy().reshape(-1, 1)  

    # Calculate Factor Covariance matrix
    F = factRet.cov().to_numpy()  

    # Using the trained factor models for each asset, compute the expected return vector
    # containing expected returns of all assets
    mu = a.reshape(-1, 1) + V.T @ f_bar  

    # Calculate asset return covariance matrix using factor model
    Q = V.T @ F @ V + D  

    # Ensure covariance matrix is symmetric
    Q = (Q + Q.T) / 2  

    return mu, Q
