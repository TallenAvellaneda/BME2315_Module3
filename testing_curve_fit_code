import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Fact(L1, L2, Kd1, Kd2):
    """Fractional receptor activation with two agonists L1 and L2"""
    return (Kd2*L1+Kd1*L2) / (Kd1*Kd2 + Kd2*L1 + Kd1*L2)

def find_kd(Kd1, L1, L2_loaded, Fact_loaded):
    
    kd2_fit_list = []
    sse_vals = []
    Kd2_candidates = np.logspace(-3, 2, 10) # [nM]

    for testpoint in Kd2_candidates:
        popt, pcov = curve_fit(lambda L2, testpoint: Fact(L1, L2, Kd1, testpoint),
                            L2_loaded, Fact_loaded, p0=[0.1], bounds=(0, np.inf))
        Kd2_fit = popt[0]
        kd2_fit_list.append(float(popt[0]))
        sse_vals.append(np.sum((Fact_loaded - Fact(L1, L2_loaded, Kd1, Kd2_fit))**2))

    min = sse_vals[0]
    i = 0
    index = 0

    for val in sse_vals:
        if val < min:
            min = val
            index = i
        i += 1

    best_kd2 = kd2_fit_list[index]
    print(f"Fitted Kd2 = {best_kd2:.3f} nM (SSE = {sse_vals[index]:.4f})")
    return best_kd2


Kd1 = 0.17 # L1, GLP1
Kd2 = 0.38 # default 0.38 L2, semaglutide (to fit)
L1 = 0.015 # [nM] GLP1 (fasting)
L2 = np.logspace(-3, 2, 500) # [nM] semaglutide or cavalitide

data = np.loadtxt("cavalitide_data.csv", delimiter=",", skiprows=1)
L2_loaded, Fact_loaded = data[:, 0], data[:, 1]

best_kd2 = find_kd(Kd1, L1, L2_loaded, Fact_loaded)

plt.figure()
plt.semilogx(L2, Fact(L1, L2, Kd1, best_kd2), c = "blue", lw=2,
            label=f"fit curve (Kd2 = {best_kd2:.3f} nM)")
plt.semilogx(L2_loaded, Fact_loaded, "o", c="orange",
            label="experimental data")

plt.ylim(-0.1, 1.1)
plt.xlabel("[cavalitide] (nM)")
plt.ylabel("Fraction active (Fact)")
plt.title("GLP1R activation by cavalitide with GLP1 present")
plt.legend(loc="upper left")
plt.show()