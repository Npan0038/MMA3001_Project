import numpy as np
from analytical import analytical_solution

# ---- parameters (no heat generation, cell starts hot and cools) ----
R = 9e-3                      # m
rho, cp, k = 2720.0, 900.0, 0.22
alpha = k / (rho * cp)
h = 20.0
T_amb = 25.0
T_init = 80.0

# ---- mesh and time step ----
n_nodes = 41
r = np.linspace(0, R, n_nodes)
dr = r[1] - r[0]
dt = 0.9 * 0.25 * dr**2 / alpha
n_steps = int(np.ceil(600.0 / dt))

# ---- FTCS loop ----
T = np.zeros((n_steps + 1, n_nodes))
T[0, :] = T_init
for n in range(n_steps):
    T_old = T[n, :]
    T_new = np.empty(n_nodes)
    T_new[0] = T_old[0] + dt * 4 * alpha * (T_old[1] - T_old[0]) / dr**2
    for i in range(1, n_nodes - 1):
        d2 = (T_old[i+1] - 2*T_old[i] + T_old[i-1]) / dr**2
        d1 = (T_old[i+1] - T_old[i-1]) / (2*dr)
        T_new[i] = T_old[i] + dt * alpha * (d2 + d1 / r[i])
    ghost = T_old[-2] - (2*dr*h/k) * (T_old[-1] - T_amb)
    d2 = (ghost - 2*T_old[-1] + T_old[-2]) / dr**2
    d1 = (ghost - T_old[-2]) / (2*dr)
    T_new[-1] = T_old[-1] + dt * alpha * (d2 + d1 / r[-1])
    T[n+1, :] = T_new

# ---- compare with the exact answer ----
Bi = h * R / k
t_final = n_steps * dt
for idx in [0, n_nodes // 2, n_nodes - 1]:
    exact = analytical_solution(r[idx], t_final, R, alpha, Bi, T_init, T_amb)
    print(f"r = {r[idx]*1000:.1f} mm   FTCS = {T[-1, idx]:.4f}   exact = {exact:.4f}   diff = {T[-1, idx] - exact:.5f}")
