import numpy as np
import matplotlib.pyplot as plt

def Fact(L, Kd):
    """Fractional receptor activation: L / (Kd + L)"""
    return L / (Kd + L)

def CoFact(L1, L2, Kd1, Kd2):
    """Fractional receptor activation with costimulation"""
    numerator = (Kd1 * L2) + (Kd2 * L1)
    denominator = (Kd1 * Kd2) + (Kd1 * L2) + (Kd2 * L1)
    return numerator/denominator

def findZero(L2, Kd1, Kd2):
    """Finds the minimum Fractional receptor activation with one set ligand concentration"""
    numerator = (Kd1 * L2)
    denominator = (Kd1 * Kd2) + (Kd1 * L2)
    return numerator/denominator

Lg = np.logspace(-2, 4, 500)    # GLP1 concentration, 0.01 to 10,000 [nM]
Ls = 1.0                        # semaglutide concentration is ~1.0 [nM]
kdg = 0.17                      # dissociation constant for GLP1 [nM]
kds = 0.38                      # dissociation constant for semaglutide [nM]

plt.figure()
plt.title("GLP1 Receptor Activation After Ozempic Dosage")
plt.semilogx(Lg, CoFact(Lg, Ls, kdg, kds), lw=2)
plt.axhline(119/138, ls="--", c="gray", lw=1)
plt.axhline(findZero(Ls, kdg, kds), ls="-", c="black", lw=1)    # creates a line at the minimum Fact value
plt.axvline(0.617, ls="--", c="gray", lw=1)
plt.ylim(0, 1)
plt.xlabel("[GLP1] (nM, log scale)")
plt.ylabel("Fraction active (Fact)")
plt.show()

Lgfasting = 0.015                   # GLP1 concentration is ~ 0.015 [nM]
Lsc = np.logspace(-3, 4, 500)       # semaglutide concentration, 0.01 to 10,000 [nM]

plt.figure()
plt.title("GLP1 Receptor Activation While Fasting Depending on Ozempic Dosage")
plt.semilogx(Lsc, CoFact(Lgfasting, Lsc, kdg, kds), lw=2)
plt.axhline(0.541, ls="--", c="gray", lw=1)
plt.axhline(findZero(Lgfasting, kds, kdg), ls="-", c="black", lw=1)    # creates a line at the minimum Fact value
plt.axvline(0.414, ls="--", c="gray", lw=1)
plt.ylim(0, 1)
plt.xlabel("[Semaglutide] (nM, log scale)")
plt.ylabel("Fraction active (Fact)")
plt.show()

Lgafter = 0.04                      # GLP1 concentration is ~ 0.04 [nM]

plt.figure()
plt.title("GLP1 Receptor Activation After Meal Depending on Ozempic Dosage")
plt.semilogx(Lsc, CoFact(Lgafter, Lsc, kdg, kds), lw=2)
plt.axhline(0.595, ls="--", c="gray", lw=1)
plt.axhline(findZero(Lgafter, kds, kdg), ls="-", c="black", lw=1)    # creates a line at the minimum Fact value
plt.axvline(0.470, ls="--", c="gray", lw=1)
plt.ylim(0, 1)
plt.xlabel("[Semaglutide] (nM, log scale)")
plt.ylabel("Fraction active (Fact)")
plt.show()