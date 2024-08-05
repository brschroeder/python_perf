from time import time
import numpy as np
import scipy.constants as spc
from scipy.optimize import curve_fit
from scipy.integrate import quad

def ng(Dit, omega, tau_m, sigma_us):
    t1 = (spc.e * Dit) / (2 * omega * tau_m)
    t2 = np.invert(np.sqrt(2 *np.pi) * sigma_us)
    def integrand(omega):
        t3 = quad()