import numpy as np
import matplotlib.pyplot as plt
from uncertainties.umath import cos
from scipy.optimize import curve_fit


# everything regarding fits
# makes a rough estimate for parameters of gaussian distributions
def initial_guess(x,y):
    A0 = (np.nanmax(y) - np.nanmin(y)) or 1.0
    mu0 = np.nanargmax(y)
    sigma0 = max(len(x) / 10, 1.0)
    C0 = float(np.nanmin(y))
    return A0, mu0, sigma0, C0

def single_gaussian(x, A, mu, sigma, C):
    return A * np.exp(-((x - mu)**2) / (2 * sigma**2)) + C

def single_gaussian_fit(x, y, uncertainties=None, p0=None, print=False):
    p0 = [initial_guess(x,y)] if p0 is None else p0
    popt, pcov = curve_fit(single_gaussian, x, y, p0=p0, sigma=uncertainties, absolute_sigma=True)

    if print is True:
        # unpack parameters
        A1, mu1, sigma1, C = popt

        # plot
        plt.figure(figsize=(7,4))
        plt.scatter(x, y, s=10, label="Data", color="black")

        # plot the individual components
        plt.plot(x, A1*np.exp(-((x - mu1)**2)/(2*sigma1**2))+C, '--', color="blue", label=f"Peak 1: μ={mu1:.2f} px")

        plt.xlabel("Pixel position")
        plt.ylabel("Normalized signal")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return popt