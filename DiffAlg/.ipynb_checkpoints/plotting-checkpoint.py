# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("seaborn")
plt.rc("text", usetex=True)
plt.rc("font", serif="serif")


def plot_surface_and_contours(ff, xx, yy, levels):
    fig = plt.figure(figsize=(9, 4.5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(xx, yy, ff, alpha=0.6)
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_zlabel('$f(x_1, x_2)$')
    ax1.set_title("Surface plot")
    
    ax2 = fig.add_subplot(122)
    cs = ax2.contour(xx, yy, ff, levels, alpha=0.6,
                     colors=sns.color_palette("Blues"))
    ax2.set_aspect(1)
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.clabel(cs, levels, fmt="%.2f")
    ax2.set_title("Contour lines")
    
    return fig