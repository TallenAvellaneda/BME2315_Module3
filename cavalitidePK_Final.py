import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Pharmacokinetic parameters
ke = 0.0039          # Elimination rate (1/h)
V = 10.8             # Volume of distribution (L)
MW = 4113.6          # Molecular weight (g/mol)
Ffree = 0.01         # Free drug fraction


# Convert concentration from mg/L to free nM
def convert_to_nM(C):
    return C * (1e6 / MW) * Ffree


# Pharmacokinetic differential equations
def pk_model(t, y, ka):
    gut, blood = y

    d_gut = -ka * gut
    d_blood = (ka * gut / V) - (ke * blood)

    return [d_gut, d_blood]


# Simulate repeated doses using solve_ivp
def simulate_PK(dose, Favailable, ka, interval, n_doses):

    gut = 0
    blood = 0

    times = []
    gut_values = []
    blood_values = []

    for i in range(n_doses):

        # Add the next dose to the gut/depot compartment
        gut += dose * Favailable

        # Set the time interval for this dose
        t_start = i * interval
        t_end = (i + 1) * interval

        t_eval = np.linspace(t_start, t_end, 150)

        # Solve the differential equations
        sol = solve_ivp(
            pk_model,
            [t_start, t_end],
            [gut, blood],
            args=(ka,),
            t_eval=t_eval
        )

        # Store the results
        if i == 0:
            times.extend(sol.t)
            gut_values.extend(sol.y[0])
            blood_values.extend(sol.y[1])
        else:
            times.extend(sol.t[1:])
            gut_values.extend(sol.y[0][1:])
            blood_values.extend(sol.y[1][1:])

        # Carry remaining drug into the next dosing interval
        gut = sol.y[0][-1]
        blood = sol.y[1][-1]

    return (
        np.array(times),
        np.array(gut_values),
        convert_to_nM(np.array(blood_values))
    )


# ==================================================
# TASK 2A: OZEMPIC AND WEGOVY
# ==================================================

# Ozempic: 2 mg weekly subcutaneous injection
t_O, gut_O, blood_O = simulate_PK(
    dose=2,
    Favailable=0.89,
    ka=0.029,
    interval=168,
    n_doses=8
)

# Wegovy: 25 mg daily oral medication
t_W, gut_W, blood_W = simulate_PK(
    dose=25,
    Favailable=0.01,
    ka=0.0029,
    interval=24,
    n_doses=100
)

# Plot Ozempic and Wegovy separately
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Ozempic
axes[0].plot(t_O / 24, gut_O,
             label="subcutaneous depot [mg]")
axes[0].plot(t_O / 24, blood_O,
             label="free in blood [nM]")

axes[0].set_title("Ozempic: 2.0 mg every 168 h")
axes[0].set_xlabel("Time (days)")
axes[0].set_ylabel("Semaglutide")
axes[0].legend()
axes[0].grid(True)

# Wegovy
axes[1].plot(t_W / 24, gut_W,
             label="gut (oral) depot [mg]")
axes[1].plot(t_W / 24, blood_W,
             label="free in blood [nM]")

axes[1].set_title("Wegovy: 25 mg every 24 h")
axes[1].set_xlabel("Time (days)")
axes[1].set_ylabel("Semaglutide")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()


# ==================================================
# TASK 2B: CAVALITIDE DAILY ORAL DOSING
# ==================================================

# Cavalitide: 21 mg every 24 hours for 100 doses
t_C, gut_C, blood_C = simulate_PK(
    dose=21,
    Favailable=0.01,
    ka=0.006,
    interval=24,
    n_doses=100
)


# ==================================================
# TASK 2C: DOSE OPTIMIZATION
# ==================================================

# Initial Cavalitide dose: 25 mg daily
t_25, gut_25, blood_25 = simulate_PK(
    dose=25,
    Favailable=0.01,
    ka=0.006,
    interval=24,
    n_doses=100
)

# Optimized Cavalitide dose: 21 mg daily
t_21, gut_21, blood_21 = simulate_PK(
    dose=21,
    Favailable=0.01,
    ka=0.006,
    interval=24,
    n_doses=100
)

# Plot Cavalitide accumulation and dose comparison
plt.figure(figsize=(10, 6))

# Original 21 mg depot profile
plt.plot(t_C / 24, gut_C,
         label="gut (oral) depot: 21 mg [mg]")

# Free blood concentrations for both doses
plt.plot(t_25 / 24, blood_25,
         label="free in blood: 25 mg [nM]")

plt.plot(t_21 / 24, blood_21,
         label="free in blood: 21 mg [nM]")

# Horizontal target concentration
plt.axhline(
    y=0.5,
    color="black",
    linestyle="--",
    label="Target free concentration: 0.5 nM"
)

plt.title("Cavalitide: Daily Oral Dose Optimization")
plt.xlabel("Time (days)")
plt.ylabel("Cavalitide")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Print final free blood concentrations
print(f"25 mg daily: {blood_25[-1]:.3f} nM")
print(f"21 mg daily: {blood_21[-1]:.3f} nM")