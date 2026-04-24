import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import correlated_values
from scipy.optimize import curve_fit
from scipy import odr


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
        plt.style.use('default')
        plt.scatter(x, y, s=10, label="Data", color="black")

        # plot the individual components
        plt.plot(x, A1*np.exp(-((x - mu1)**2)/(2*sigma1**2))+C, '--', color="blue", label=f"Peak 1: μ={mu1:.2f} px")

        plt.xlabel("Pixel position")
        plt.ylabel("Normalized signal")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return popt

def polynomial(beta, theta):
    a, b, c, d, e = beta
    return a + b*theta + c*theta**2 + d*theta**3 + e*theta**4

def polynomial_fit(x, y, sigma_x=None, sigma_y=None):
    model = odr.Model(polynomial)
    data = odr.RealData(x, y, sx=sigma_x, sy=sigma_y)
    p0 = np.array([1, 1, 1, 1, 1])
    odr_run = odr.ODR(data, model, beta0=p0).run()

    a_fit, b_fit, c_fit, d_fit, e_fit = odr_run.beta
    a_err, b_err, c_err, d_err, e_err = odr_run.sd_beta

    a = ufloat(a_fit, a_err)
    b = ufloat(b_fit, b_err)
    c = ufloat(c_fit, c_err)
    d = ufloat(d_fit, d_err)
    e = ufloat(e_fit, e_err)

    return a, b, c, d, e

def polyfit_peak(x, y, print=False):
    # fit
    coeff, cov = np.polyfit(x, y, 2, cov=True)
    a, b, c = coeff
    sigma_a = np.sqrt(cov[0, 0])
    sigma_b = np.sqrt(cov[1, 1])

    # store coefficients in ufloat
    a_u = ufloat(a, sigma_a)
    b_u = ufloat(b, sigma_b)

    # calculate peak of polynomial
    x_max = -b_u / (2 * a_u)

    y_hat = a*x**2 + b*x + c

    if print == True:
        plt.figure(figsize=(8, 5))
        plt.style.use('default')
        plt.plot(x, y, label="spectra")
        plt.plot(x, y_hat, label="fit")
        plt.xlabel("Pixel number")
        plt.ylabel("Intensity (normalized)")
        plt.axvline(x=x_max.n, color="red", linestyle="--")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return x_max