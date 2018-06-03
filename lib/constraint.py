import numpy as np
from scipy.optimize import OptimizeResult, minimize, linprog


def lagrange(f, g, h, lbda, mu, c):
    """Lagrange function for Augmented Lagrangian Method"""
    return lambda x: (f(x) 
                      + dot(lbda, h(x)) + c/2*dot(h(x), h(x)) 
                      + 0.5/c*np.sum( np.maximum(0, mu+c*g(x))**2 - mu**2 ))


def lagrange_jac(f, g, h, jacf, jacg, jach, lbda, mu, c):
    """ Derivative of Lagrange function for Augmented Lagrangian Method"""
    return lambda x: (jacf(x) 
                      + dot(jach(x).T, lbda+c*h(x))
                      + dot(jacg(x).T, np.maximum(mu+c*g(x), 0)))


def fmin_augmented_lagrangian(f, g, h, x0, jacf, jacg, jach, eps=0.001):
    """"""
    # problem dimensionality
    n, m, k = len(x0), len(g(x0)), len(h(x0))
    X = [x0]
    
    # initial values of Lagrange multipliers
    lbda, mu = np.ones(k), np.ones(m)
    # and hyperparameters
    c = 0.1
    
    while True:
        # create Lagrangian function and its derivative
        # given current values of Lagrangian multipliers
        L = lagrange(f, g, h, lbda, mu, c)
        jacL = lagrange_jac(f, g, h, jacf, jacg, jach, lbda, mu, c)
        
        # perform unconstrained optimization of Lagrangian
        res = minimize(L, X[-1], method='BFGS', jac=jacL, tol=0.1*eps)
        X.append(res.x)
        
        # update Lagrangian multipliers
        lbda += c*h(X[-1])
        mu = np.maximum(0, mu + c*g(X[-1]))
        c *= 8
        
        # check for termination conditions
        if np.linalg.norm(X[-1] - X[-2]) < eps:
            break
        
    return OptimizeResult(x=X[-1], fun=f(X[-1]),
                          x_hist=np.c_[X].T, 
                          f_hist=f(np.c_[X].T), 
                          nit=len(X)-1)


def fmin_cutting_plane(c, g, jacg, bounds, eps=1e-3, max_iter=100, callback=None):
    """"""
    # change the design to 'constraints' interface?
    
    success = False
    message = 'Maximum number of iterations exceeded'
    
    res = linprog(c, bounds=bounds)
    if not res.success:
        pass # error
    
    X = [res.x]
    A = []
    b = []
    
    for k in range(max_iter):
        
        gk = g(X[k])
        # find the inequality that doesn't hold the most
        idx = np.argmax(gk)
        
        # check termination condition
        if gk[idx] < eps:
            success = True
            message = 'Optimization terminated successfylly'
            break
            
        # add the cutting plane gk + jacgk*(x-xk) <= 0
        j = jacg(X[k])
        A.append(j[idx])
        b.append(np.dot(j[idx], X[k]) - gk[idx])
        
        # solve the linear programming problem with updated domain
        res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds)
        if not res.success:
            pass # error termination
        
        X.append(res.x)
        
        if callback is not None:
            callback(X[k+1])
        
        k += 1
            
    return OptimizeResult(x=X[-1], fun=np.dot(c, X[-1]), nit=k, 
                          success=success, message=message)
        

def fmin_gradient_projection(f, x0, jacf, constraints, eps=1e-3, max_iter=100, callback=None):
    """"""
    X = [x0]
    success = False
    message = ""
    
    if callback is not None:
        callback(x0)
    
    # scipy.optimize.minimize requires inequality constraints to be in a form of 
    # g_i(x) >= 0
    sp_constraints = []
    for c in constraints:
        sp_c = {}
        sp_c['type'] = c['type']
        sp_c['fun'] = lambda x, c=c: -c['fun'](x)
        sp_c['jac'] = lambda x, c=c: -c['jac'](x)
        sp_constraints.append(sp_c)
    
    for k in range(max_iter):
        
        s = -jacf(X[k])
        alpha = 1/(k+1)
                
        d = lambda x: np.linalg.norm(x - X[k] - alpha*s)**2
        res = minimize(d, X[k], constraints=sp_constraints, tol=eps)
        
        X.append(res.x)
        
        if callback is not None:
            callback(X[k+1])
        
        if np.linalg.norm(X[k+1] - X[k]) < eps:
            success = True
            message = ""
            break
            
        k += 1
    
    return OptimizeResult(x=X[-1], fun=f(X[-1]), nit=k+1, 
                          success=success, message=message)

