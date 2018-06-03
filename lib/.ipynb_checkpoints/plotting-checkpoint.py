import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("seaborn")
plt.rc('text', usetex=True)
plt.rc('font', serif='serif')


def plot_surface(f, xx, yy, ax=None):
    """3-dimensional surface plot of a function of two variables."""
    if ax is None:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, f([xx, yy]), alpha=0.6)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$f(x, y)$')
    return fig, ax


def plot_contours(f, xx, yy, levels, constraints=None, ax=None, keep_scale=True):
    """Contour lines of a function of two variables."""
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.set_title("Contour plots")
    
    fcs = ax.contour(xx, yy, f([xx, yy]), levels, alpha=0.6,
                    colors=sns.color_palette("Blues"))
    ax.clabel(fcs, levels, fmt="$f = %.2f$")
    
    if constraints is not None:
        feasible = np.ones_like(xx, dtype=bool)
        for i, c in enumerate(constraints):
            cc = c['fun']([xx, yy])
            if c['type'] == 'ineq':
                feasible &= (cc <= 0)
            ccs = ax.contour(xx, yy, cc, levels=[0], 
                             colors=sns.color_palette("gnuplot"), alpha=0.6)
            ax.clabel(ccs, [0], fmt=c['type']+"$_{} = %.2f$".format(i+1))
        
        # fill the feasible region 
        y_max = np.max(np.where(feasible, yy, -np.inf), axis=0)
        y_min = np.min(np.where(feasible, yy, np.inf), axis=0)
        idx = y_max != -np.inf 
        ax.fill_between(xx[0, idx], y_min[idx], y_max[idx], 
                        alpha=0.33, interpolate=True)
    
    if keep_scale:
        ax.set_aspect(1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return fig, ax


def plot_univariate():
    pass
