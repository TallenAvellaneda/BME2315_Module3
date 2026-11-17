import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Fact(L1, L2, Kd1, Kd2):
    # Fractional receptor activation with two agonists L1 and L2
    return (Kd2*L1+Kd1*L2) / (Kd1*Kd2 + Kd2*L1 + Kd1*L2)

def find_kd(Kd1, L1, L2_loaded, Fact_loaded):

    # creates lists for comparing the best curve fit that gives the least SSE
    kd2_fit_list = []
    sse_vals = []
    Kd2_candidates = np.logspace(-4, 3, 10) # [nM]

    # adds each curve fit attempt into the the kd2_fit_list and its corresponding SSE
    for testpoint in Kd2_candidates:
        popt, pcov = curve_fit(lambda L2, testpoint: Fact(L1, L2, Kd1, testpoint),
                            L2_loaded, Fact_loaded, p0=[0.1], bounds=(0, np.inf))
        Kd2_fit = popt[0]
        kd2_fit_list.append(float(popt[0]))
        sse_vals.append(np.sum((Fact_loaded - Fact(L1, L2_loaded, Kd1, Kd2_fit))**2))

    # variables to compare SSEs
    min = sse_vals[0]
    i = 0
    index = 0

    # finds the index of the lowest SSE
    for val in sse_vals:
        if val < min:
            min = val
            index = i
        i += 1

    # returns the Kd2 that gives the best fit (lowest SSE)
    best_kd2 = kd2_fit_list[index]
    print(f"Fitted Kd2 = {best_kd2:.3f} nM (SSE = {sse_vals[index]:.4f})")
    return best_kd2


Kd1 = 0.17 # L1, GLP1
L1 = 0.015 # [nM] GLP1 (fasting)
L2 = np.logspace(-4, 3, 500) # [nM] semaglutide or cavalitide

cav_data = np.loadtxt("cavalitide_data.csv", delimiter=",", skiprows=1)
sema_data = np.loadtxt("semaglutide_data.csv", delimiter=",", skiprows=1)
cav_L2_loaded, cav_Fact_loaded = cav_data[:, 0], cav_data[:, 1]
sema_L2_loaded, sema_Fact_loaded = sema_data[:, 0], sema_data[:, 1]

print("Cavalitide Kd2:")
cav_best_kd2 = find_kd(Kd1, L1, cav_L2_loaded, cav_Fact_loaded)
print("-------------------------")
print("Semaglutide Kd2:")
sema_best_kd2 = find_kd(Kd1, L1, sema_L2_loaded, sema_Fact_loaded)
print("-------------------------")

plt.figure()
plt.semilogx(cav_L2_loaded, cav_Fact_loaded, "o", c="blue",
            label="Cavalitide Experimental Data")
plt.semilogx(sema_L2_loaded, sema_Fact_loaded, "o", c="orange",
            label="Semaglutide Experimental Data")
plt.semilogx(L2, Fact(L1, L2, Kd1, cav_best_kd2), "--", c = "blue", lw=2,
            label=f"Cavalitide Model (Kd2 = {cav_best_kd2:.3f} nM)")
plt.semilogx(L2, Fact(L1, L2, Kd1, sema_best_kd2), "--", c = "orange", lw=2,
            label=f"Semaglutide Model (Kd2 = {sema_best_kd2:.3f} nM)")

plt.axhline(0.5, ls=":", c="black", lw=1)
plt.text(0.0001, 0.525, "Fact = 0.5")
plt.ylim(-0.1, 1.1)
plt.xlabel("[Drug] (nM)")
plt.ylabel("Fraction Active (Fact)")
plt.title("Comparing GLP1R Activation between Semaglutide and Cavalitide")
plt.legend(loc="upper left", fontsize = 7)
plt.show()