import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path

# ===== Cavalitide parameters =====
products = {
    'Cavalitide': dict(
        Favailable=0.01,
        ka=0.006,
        D=29,
        interval=24,
        n_doses=100,
        depot='gut (oral)'
    ),
}

# ===== Shared PK parameters =====
ke = 0.0039       # Elimination rate (1/h)
MW = 4113.6       # Molecular weight (g/mol)
Ffree = 0.01      # Free fraction of Cavalitide

dt = 0.1          # Output time spacing (h)

# ===== Shared PD parameters =====
Kd1 = 0.17       # GLP1 dissociation constant (nM)
Kd2 = 0.077      # Cavalitide fitted Kd (nM)
L1 = 0.015       # Fasting GLP1 concentration (nM)

# Calculated Blood Volumes by Sex/Weight (weight derived from BMI)
volumes = [5.0, 10.8] # blood volume

# ===== Pharmacodynamic model =====
def Fact(L1, L2, Kd1, Kd2):
    """Fraction of active GLP1 receptors"""

    return (Kd2 * L1 + Kd1 * L2) / (
        Kd1 * Kd2 + Kd2 * L1 + Kd1 * L2
    )

# ===== Pharmacokinetic model =====
def ODEfunc(t, y, ka):
    Asc, Cc = y

    dAsc = -ka * Asc
    dCc = ka * Asc / V - ke * Cc

    return [dAsc, dCc]


# ===== Simulate repeated dosing using solve_ivp =====
def simulate(D, n_doses, interval, ka, Favailable, dt):

    t_out, Asc_out, Cc_out = [], [], []

    y = np.array([0.0, 0.0])

    n_steps = int(round(interval / dt))
    t_eval = np.linspace(0, interval, n_steps + 1)

    for i in range(n_doses):

        # Add dose to the gut depot
        y[0] += Favailable * D

        t0 = i * interval

        # Solve differential equations
        sol = solve_ivp(
            ODEfunc,
            (0, interval),
            y,
            args=(ka,),
            t_eval=t_eval
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        t_out.append(t0 + sol.t)
        Asc_out.append(sol.y[0])
        Cc_out.append(sol.y[1])

        # Carry the final state into the next interval
        y = sol.y[:, -1].copy()

    # Combine all dosing intervals
    t = np.concatenate(t_out)
    Asc_out = np.concatenate(Asc_out)
    Cc_out = np.concatenate(Cc_out)

    # Convert mg/L to nM
    Cc_nM = Cc_out * 1e-3 / MW * 1e9

    return t, Asc_out, Cc_nM

# ===== Run Cavalitide PK simulation =====
results = []

for volume in volumes:
    V = volume      # volume of distribution (L)
    for drug, p in products.items():

        t, Asc, Cc = simulate(
            p['D'],
            p['n_doses'],
            p['interval'],
            p['ka'],
            p['Favailable'],
            dt
        )

        results.append([t, Asc, Cc])

t_cav = []
Asc_cav = []
Cc_cav = []
Fact_over_time = []

for result in results:
    # ===== Calculate receptor activation over time =====
    t_cav.append(result[0])
    Asc_cav.append(result[1])
    Cc_cav.append(result[2])

    # Convert total blood concentration to free concentration
    L2_free = Ffree * result[2] # result[2] == Cc_cav => L2_free = Ffree * Cc_cav

    # Calculate fraction of active receptors
    Fact_over_time.append(Fact(L1, L2_free, Kd1, Kd2))

index = 0
line_type = ["-", "--", ":", "-."]

# ===== Plot PKPD results =====
plt.figure(figsize=(10, 5))

while index < len(results):
    plt.plot(
        t_cav[index] / 24,
        Fact_over_time[index],
        label=f"Fraction of active GLP1 receptors (Volume = {volumes[index]:.3f})",
        color= "orange",
        linestyle = line_type[index]
        )

    index += 1

plt.xlabel("Time (days)")
plt.ylabel("Fraction active (Fact)")
plt.title("GLP1R Activation by Cavalitide Over Time (29 mg per 24 hr.)")

plt.axhline(0.9, ls=":", c="black", lw=1)
plt.text(0, 0.925, "Target Fact = 0.9")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.tight_layout()
plt.show()

# ===== Run Cavalitide PK Simulation Changing Dose Interval =====

results.clear()
t_cav.clear()
Asc_cav.clear()
Cc_cav.clear()
Fact_over_time.clear()

products['Cavalitide']['interval'] = 48
products['Cavalitide']['n_doses'] = 50

for volume in volumes:
    V = volume      # volume of distribution (L)
    for drug, p in products.items():

        t, Asc, Cc = simulate(
            p['D'],
            p['n_doses'],
            p['interval'],
            p['ka'],
            p['Favailable'],
            dt
        )

        results.append([t, Asc, Cc])

for result in results:
    # ===== Calculate receptor activation over time =====
    t_cav.append(result[0])
    Asc_cav.append(result[1])
    Cc_cav.append(result[2])

    # Convert total blood concentration to free concentration
    L2_free = Ffree * result[2] # result[2] == Cc_cav => L2_free = Ffree * Cc_cav

    # Calculate fraction of active receptors
    Fact_over_time.append(Fact(L1, L2_free, Kd1, Kd2))

index = 0

# ===== Plot PKPD results =====
plt.figure(figsize=(10, 5))

while index < len(results):
    plt.plot(
        t_cav[index] / 24,
        Fact_over_time[index],
        label=f"Fraction of active GLP1 receptors (Volume = {volumes[index]:.3f})",
        color= "orange",
        linestyle = line_type[index]
        )

    index += 1

plt.xlabel("Time (days)")
plt.ylabel("Fraction active (Fact)")
plt.title("GLP1R Activation by Cavalitide Over Time (29 mg per 48 hr.)")

plt.axhline(0.9, ls=":", c="black", lw=1)
plt.text(0, 0.925, "Target Fact = 0.9")
plt.ylim(0, 1.05)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.tight_layout()
plt.show()