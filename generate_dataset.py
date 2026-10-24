import time
import numpy as np

# ---- fixed cell properties (sources are listed in docs/problem_definition.md) ----
R = 9e-3                         # m, 18650 radius
H_CELL = 65e-3                   # m, cell height
V = np.pi * R**2 * H_CELL        # m^3
rho, cp, k = 2720.0, 900.0, 0.22
alpha = k / (rho * cp)
R_INT = 0.01923                  # ohm, Samsung INR18650-25R datasheet, DC typical
CHARGE_TIME = 3600.0             # s, one hour charge
N_NODES = 21                     # coarse mesh is fine: convergence study shows error ~0.008 C

def solve(current, h, T_amb):
    """FTCS solve of the cell for CHARGE_TIME. Returns centre temp, surface temp (C)."""
    r = np.linspace(0, R, N_NODES)
    dr = r[1] - r[0]
    dt = 0.9 * 0.25 * dr**2 / alpha
    n_steps = int(np.ceil(CHARGE_TIME / dt))
    q = current**2 * R_INT / V / (rho * cp)      # heating rate, K/s
    T = np.full(N_NODES, T_amb)
    for n in range(n_steps):
        T_new = np.empty(N_NODES)
        T_new[0] = T[0] + dt * (4 * alpha * (T[1] - T[0]) / dr**2 + q)
        d2 = (T[2:] - 2*T[1:-1] + T[:-2]) / dr**2
        d1 = (T[2:] - T[:-2]) / (2*dr)
        T_new[1:-1] = T[1:-1] + dt * (alpha * (d2 + d1 / r[1:-1]) + q)
        ghost = T[-2] - (2*dr*h/k) * (T[-1] - T_amb)
        d2 = (ghost - 2*T[-1] + T[-2]) / dr**2
        d1 = (ghost - T[-2]) / (2*dr)
        T_new[-1] = T[-1] + dt * (alpha * (d2 + d1 / r[-1]) + q)
        T = T_new
    return T[0], T[-1]

if __name__ == "__main__":
    rng = np.random.default_rng(42)              # fixed seed so results are repeatable
    n_samples = 300
    currents = rng.uniform(1.0, 4.0, n_samples)  # A, up to the datasheet max charge current
    hs = rng.uniform(5.0, 28.4, n_samples)       # W/m2K, 28.4 = sourced natural convection; lower end is an ASSUMED poorly cooled pack
    T_ambs = rng.uniform(15.0, 45.0, n_samples)  # C

    rows = []
    for i in range(n_samples):
        t0 = time.perf_counter()
        T_centre, T_surface = solve(currents[i], hs[i], T_ambs[i])
        runtime = time.perf_counter() - t0
        rows.append([currents[i], hs[i], T_ambs[i], T_centre, T_surface, runtime])
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{n_samples} runs done")

    header = "current_A,h_W_m2K,T_amb_C,T_centre_final_C,T_surface_final_C,solver_runtime_s"
    np.savetxt("dataset.csv", np.array(rows), delimiter=",", header=header, comments="")
    data = np.array(rows)
    print("saved dataset.csv")
    print(f"centre temp range: {data[:,3].min():.2f} to {data[:,3].max():.2f} C")
    print(f"mean solver runtime: {data[:,5].mean():.3f} s per run")
