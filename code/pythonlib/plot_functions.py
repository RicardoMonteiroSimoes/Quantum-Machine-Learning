import numpy as np
import matplotlib.pyplot as plt

# Plot Helper Funtions


def contourPlot(figureSize=(6.4, 4.8),  # Todo: refactor whole func
                axelLabels=['x', 'y'],
                pltTitle='Filled Contours Plot',
                withColorbar=False):
    """
    Function to create countour plots
    """

    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)

    X, Y = np.meshgrid(xlist, ylist)
    Z = np.sqrt(X**2 + Y**2)

    # figure
    fig, ax = plt.subplots(1, 1, figsize=figureSize)
    cp = ax.contourf(X, Y, Z)

    # colorbar
    if(withColorbar):
        fig.colorbar(cp)  # Add a colorbar to a plot

    ax.set_title(pltTitle)
    ax.set_xlabel(axelLabels[0])
    ax.set_ylabel(axelLabels[1])

    return plt
