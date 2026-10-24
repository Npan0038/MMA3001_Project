import numpy as np

import matplotlib.pyplot as plt

diameter = 18e-3
height = 65e-3
R = diameter / 2
volume = np.pi * R**2 * height

rho = 2720.0
cp = 900.0
k = 0.22
alpha = k / (rho * cp)

I_charge = 10.0
R_int = 0.05
q_gen = (I_charge**2 * R_int) / volume

h = 20.0
T_amb = 25.0
T_init = 25.0

n_nodes = 41
r = np.linspace(0, R, n_nodes)
dr = r[1] - r[0]

dt_max = 0.25 * dr**2 / alpha
dt = 0.9 * dt_max

total_time = 600.0
n_steps = int(np.ceil(total_time / dt))

T = np.zeros((n_steps + 1, n_nodes))
T[0, :] = T_init

for n in range(n_steps):
    T_old = T[n, :]
    T_new = np.empty(n_nodes)

    T_new[0] = T_old[0] + dt * (
        4 * alpha * (T_old[1] - T_old[0]) / dr**2
        + q_gen / (rho * cp)
    )

    for i in range(1, n_nodes - 1):
        ri = r[i]
        d2T_dr2 = (
            T_old[i + 1] - 2 * T_old[i] + T_old[i - 1]
        ) / dr**2
        dT_dr = (
            T_old[i + 1] - T_old[i - 1]
        ) / (2 * dr)

        T_new[i] = T_old[i] + dt * (
            alpha * (d2T_dr2 + dT_dr / ri)
            + q_gen / (rho * cp)
        )

    T_ghost = T_old[n_nodes - 2] - (
        2 * dr * h / k
    ) * (T_old[n_nodes - 1] - T_amb)

    ri = r[n_nodes - 1]

    d2T_dr2 = (
        T_ghost - 2 * T_old[n_nodes - 1] + T_old[n_nodes - 2]
    ) / dr**2

    dT_dr = (
        T_ghost - T_old[n_nodes - 2]
    ) / (2 * dr)

    T_new[n_nodes - 1] = T_old[n_nodes - 1] + dt * (
        alpha * (d2T_dr2 + dT_dr / ri)
        + q_gen / (rho * cp)
    )

    T[n + 1, :] = T_new

**Test 1**
plt.plot(r * 1000, T[-1, :])
plt.xlabel("radius (mm)")
plt.ylabel("Temperature (C)")
plt.title("Final temperature profile")
plt.show()
