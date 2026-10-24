"""Mesh convergence study for the FTCS solver.

Runs the no-heating cooling problem on five meshes and measures the maximum error against
the exact solution in analytical.py. Halving the mesh spacing should cut the error by 4
(second-order convergence).
"""

import numpy as np
import matplotlib.pyplot as plt
from analytical import analytical_solution

# ---- same cooling problem as validate.py (no heat generation) ----
R = 9e-3
rho, cp, k = 2720.0, 900.0, 0.22
alpha = k / (rho * cp)
h = 20.0
T_amb = 25.0
T_init = 80.0
Bi = h * R / k

def run_ftcs(n_nodes):
    """Run FTCS for 600 s on a mesh with n_nodes points. Return dr and the max error vs exact."""
    r = np.linspace(0, R, n_nodes)
    dr = r[1] - r[0]
    dt = 0.9 * 0.25 * dr**2 / alpha          # stable time step for this mesh
    n_steps = int(np.ceil(600.0 / dt))
    T = np.full(n_nodes, T_init)
    for n in range(n_steps):
        T_new = np.empty(n_nodes)
        T_new[0] = T[0] + dt * 4 * alpha * (T[1] - T[0]) / dr**2
        d2 = (T[2:] - 2*T[1:-1] + T[:-2]) / dr**2
        d1 = (T[2:] - T[:-2]) / (2*dr)
        T_new[1:-1] = T[1:-1] + dt * alpha * (d2 + d1 / r[1:-1])
        ghost = T[-2] - (2*dr*h/k) * (T[-1] - T_amb)
        d2 = (ghost - 2*T[-1] + T[-2]) / dr**2
        d1 = (ghost - T[-2]) / (2*dr)
        T_new[-1] = T[-1] + dt * alpha * (d2 + d1 / r[-1])
        T = T_new
    t_final = n_steps * dt
    exact = np.array([analytical_solution(x, t_final, R, alpha, Bi, T_init, T_amb) for x in r])
    return dr, np.max(np.abs(T - exact))

mesh_sizes = [11, 21, 41, 81, 161]
drs, errs = [], []
for n in mesh_sizes:
    dr, err = run_ftcs(n)
    drs.append(dr)
    errs.append(err)
    print(f"nodes = {n:4d}   dr = {dr*1000:.4f} mm   max error = {err:.6f} C")

slope = np.polyfit(np.log(drs), np.log(errs), 1)[0]
print(f"Measured order of convergence (slope on log-log plot): {slope:.2f}")

plt.loglog(drs, errs, "o-", label="FTCS max error")
plt.loglog(drs, [errs[0] * (d / drs[0])**2 for d in drs], "--", label="slope 2 (reference)")
plt.xlabel("mesh spacing dr (m)")
plt.ylabel("max error vs exact solution (C)")
plt.title("Mesh convergence of FTCS solver")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig("convergence.png", dpi=150)
plt.show()
