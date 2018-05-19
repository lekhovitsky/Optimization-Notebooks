# -*- coding: utf-8 -*-
import numpy as np
from numpy import array
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import OptimizeResult


def is_optimal_point(x, r, jacf, left, right, tol):
    if np.allclose(x[r], left[r]):
        return jacf(x)[r] > 0
    elif np.allclose(x[r], right[r]):
        return jacf(x)[r] < 0
    else:
        return np.allclose(jacf(x)[r], 0)
    

def is_optimal_point_all(x, jacf, left, right, tol):
    
    j = jacf(x)
    
    left_bound = np.isclose(x, left)
    right_bound = np.isclose(x, right)
    inner = (x > left) & (x < right)
    
    return np.alltrue(np.where(
        left_bound, j > 0, np.where(
            right_bound, j < 0, np.where(
                inner, np.isclose(j, 0), False
            )
        )))


def make_step(x, r, jacf, left, right, tol):
    if is_optimal_point(x, r, jacf, left, right, tol):
        return 0
    else:       
        def dfdxr(xr):
            y = x.copy()
            y[r] = xr
            return jacf(y)[r]
        
        xr_new = fsolve(dfdxr, x[r], xtol=tol)[0]
        if xr_new > right[r]:
            return right[r] - x[r]
        elif xr_new < left[r]:
            return left[r] - x[r]
        else: return xr_new - x[r]
        return 0
    

def pprint(value, *values, verbose=False, sep=' ', end='\n'):
    if verbose:
        print(value, *values, sep, end)
    
    
def minimize_diffalg(f, x0, jacf, left, right, tol=1e-4, max_iter=1000, verbose=False):
    
    X = [x0.astype(np.float64)]
    n = len(x0)
    
    success = False
    message = 'Maximum number of iterations has been exceeded.'
    
    for k in range(max_iter):
        
        x = X[k].copy()
        
        for r in range(n):
            s = make_step(x, r, jacf, left, right, tol)
            x[r] += s
        
        X.append(x)
        
        if is_optimal_point_all(x, jacf, left, right, tol):
            success = True
            message = 'Optimization terminated successfully.'
            break
    
    return OptimizeResult(fun = f(X[-1]), jac = jacf(X[-1]), x = X[-1],
                          nit = k, message = message, success = success)
