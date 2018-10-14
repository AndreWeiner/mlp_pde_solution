"""Collection of functions to visualize MLPs.

MLP - multi-layer-perceptron
"""
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'size': 32})

SMALL_SIZE = 10
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlepad=20)


def plot_field_and_error(x, y, field, reference, name, titles):
    """Plot figure with two sub-plots: field and deviation from reference."""
    diff = field - reference
    fig = plt.figure(figsize=(13, 4))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_trisurf(y, x, field, cmap=cm.coolwarm, linewidth=0.0,
                     antialiased=False)
    ax1.set_title(titles[0])
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_trisurf(y, x, diff, cmap=cm.coolwarm, linewidth=0.0,
                     antialiased=False)
    ax2.set_title(titles[1])

    for ax in [ax1, ax2]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.view_init(30, 120)
        ax.set_xlabel(r"$y$")
        ax.set_ylabel(r"$x$")

    fig.subplots_adjust(wspace=0.05, hspace=0.0)
    ax1.set_zlabel(r"$c_t$")
    ax2.set_zlabel(r"$c_t-c_{num}$")
    ax1.set_zlim(min(field), max(field))
    ax2.set_zlim(min(diff), max(diff))
    fig.savefig(name, bbox_inches='tight')
