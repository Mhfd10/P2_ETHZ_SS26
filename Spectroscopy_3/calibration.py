import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties.umath import cos
from scipy import odr
from fits import single_gaussian_fit

# everything regarding calibration

# first calibration step
def k_lambda(beta, k):
    b, k0 = beta
    return b*(k-k0)

def k_lambda_fit(He_values, k_values, k_uncertainties):
    model = odr.Model(k_lambda)
    data = odr.RealData(k_values, He_values, sx=k_uncertainties, sy=None)
    odr_run = odr.ODR(data, model, beta0=[-0.3, 500]).run()
    b_fit, k0_fit = odr_run.beta
    b_err, k0_err = odr_run.sd_beta

    # Plot
    k_min, k_max = np.min(k_values), np.max(k_values)
    k_fit = np.linspace(k_min - 0.05*(k_max-k_min), k_max + 0.05*(k_max-k_min), 1000)
    lam_fit = b_fit * (k_fit - k0_fit)
    lam_pred = b_fit * (k_values - k0_fit)

    # RSME
    residuals = He_values - lam_pred
    rmse = np.sqrt(np.mean(residuals ** 2))


    plt.figure(figsize=(7.5,5))
    plt.style.use('default')
    plt.errorbar(k_values, He_values, xerr=k_uncertainties*10, yerr=None, fmt='o', capsize=3, label='He spectral lines (uncertainty 10x scaled)')
    plt.plot(k_fit, lam_fit, '-', label='Linear regression')

    plt.xlabel('Rotation position k')
    plt.ylabel('Wavelength [nm]')
    plt.title(f'Linear calibration fit wavelength in regards to k (RMSE = {rmse:.4f} nm)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('k_lambda_fit.png', dpi=600)
    plt.show()

    # convert to ufloat for error propagation
    b_u   = ufloat(b_fit,  b_err)
    k0_u  = ufloat(k0_fit, k0_err)

    return b_u, k0_u

def p_k(B, k, lambda_0, p0, k0, b):
    return ((lambda_0/b)-k+k0)*B+p0

# second calibration step
# fit through the origin taking into account uncertainties in b and k0
def fit_B_with_b_k0_uncertainties(img_dic_p_k,fit_results_p_k,b_u,k0_u,lambda0,p_central, B0=-12.0):
    k = k_measured_p = np.array([img_dic_p_k[name]['k_value'] for name in img_dic_p_k])
    p = np.array([fit_results_p_k[name]["mu"] for name in img_dic_p_k])
    p_uncertainty = np.abs([fit_results_p_k[name]["sigma"] for name in fit_results_p_k])
    y = p - p_central

    k_uncertainty = 1
    b, b_uncertainty   = b_u.n,  b_u.s
    k0, k0_uncertainty = k0_u.n, k0_u.s

    # Build x from nominal b,k0
    x = (lambda0 / b) - k + k0

    # gaussian error propagation
    dx_db = (lambda0 / (b**2)) * b_uncertainty
    sx  = np.sqrt(dx_db**2 + k0_uncertainty**2 + k_uncertainty**2)

    # fit through origin
    model = odr.Model(lambda beta, xdata: beta[0] * xdata)
    data  = odr.RealData(x, y, sx=sx, sy=p_uncertainty)
    out   = odr.ODR(data, model, beta0=[B0]).run()

    B_fit = out.beta[0]
    B_err = out.sd_beta
    B = ufloat(B_fit, B_err)

    y_hat = B_fit * x

    # RSME
    residuals = y - y_hat
    rmse = np.sqrt(np.mean(residuals ** 2))

    # plot fit
    kmin, kmax = float(np.min(k_measured_p)), float(np.max(k_measured_p))
    k_fit = np.linspace(kmin - 0.05 * (kmax - kmin), kmax + 0.05 * (kmax - kmin), 1000)
    p_fit = p_k(B.n, k_fit, lambda0, p_central, k0, b)

    plt.figure(figsize=(7.5, 5))
    plt.errorbar(k_measured_p, p, xerr=1,
                 yerr=np.abs([fit_results_p_k[name]["sigma"] * 10 for name in fit_results_p_k]), fmt='o', capsize=3,
                 label='He line (uncertainty in pixelposition 10x scaled)')
    plt.plot(k_fit, p_fit, '-', label=f'Linear Regression')
    plt.xlabel('Rotation position k')
    plt.ylabel('Pixel position')
    plt.title(f'Linear calibration fit pixel position in regrades to k (RMSE = {rmse:.4f} nm)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('p_k_fit.png', dpi=600)
    plt.show()

    return B

def fit_gaussian_intensity_profiles(img_dic):
    fit_results = {}

    for name in img_dic:
        y = img_dic[name]["pixel_values"]
        x = np.linspace(0, len(y), len(y))

        # Gaussian fit
        p_opt = single_gaussian_fit(x, y)
        A, mu, sigma, C = p_opt

        fit_results[name] = {
            "A": A,
            "mu": mu,
            "sigma": sigma,
            "C": C,
        }

    return fit_results

# calibrated wavelength function
def lambda_cal(b, k0, p0, B, k, p):
    return b*((k-k0)+(p-p0)/B)