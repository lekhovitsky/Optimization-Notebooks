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
    
    left_bound = np.isclose(x, left)
    right_bound = np.isclose(x, right)
    inner = (x > left) & (x < right)
    
    j = jacf(x)
    return np.alltrue(np.where(
        left_bound, j > 0, np.where(
            right_bound, j < 0, np.where(
                inner, np.isclose(j, 0), False
            )
        )))


def make_step(x, r, jacf, left, right, tol):
    
    if is_optimal_point(x, r, jacf, left, right, tol):
        return 0
          
    def dfdxr(xr):
        y = x.copy()
        y[r] = xr
        return jacf(y)[r]
    
    xr_new = fsolve(dfdxr, x[r], xtol=tol)[0]
    if xr_new > right[r]:
        return right[r] - x[r]
    elif xr_new < left[r]:
        return left[r] - x[r]
    return xr_new - x[r]
    

def pprint(value, *values, verbose=False):
    if verbose:
        print(value, *values)
    
    
def minimize_diffalg(f, x0, jacf, left, right, tol=1e-4, max_iter=200, verbose=False):
    
    X = [x0.astype(np.float64)]    
    success = False
    message = 'Maximum number of iterations has been exceeded.'
    
    pprint("Starting optimization.\nx0 =", x0, "\n", verbose=verbose)
    
    for k in range(max_iter):
        
        pprint("Iteration %d" % (k+1), verbose=verbose)
        x = X[-1].copy()
        
        for r in range(len(x0)):
            s = make_step(x, r, jacf, left, right, tol)
            pprint("r = %d, step = %.4f" % (r+1, s), verbose=verbose)
            x[r] += s
            X.append(x.copy())
            
        pprint("x%d =" % (k+1), x, "\nf = %.3f\n" % f(x), verbose=verbose)
        
        if is_optimal_point_all(x, jacf, left, right, tol):
            success = True
            message = 'Optimization terminated successfully.'
            break
    
    pprint(message, verbose=verbose)
    
    return OptimizeResult(fun = f(X[-1]), jac = jacf(X[-1]), x = X[-1], hist = X,
                          nit = k, message = message, success = success)
