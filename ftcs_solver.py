"""
FTCS solver for 1D radial heat conduction in a cylindrical 18650 cell.

Governing PDE:
    rho*cp * dT/dt = (1/r) * d/dr( k * r * dT/dr ) + q_gen

BCs:  dT/dr = 0 at r=0 (symmetry)
      -k dT/dr = h*(T_surf - T_amb) at r=R (convection)
IC:   T(r,0) = T_ambient
"""

import numpy as np

# ---------- 1. Cell geometry (18650 standard) ----------
diameter = 18e-3        # m
height   = 65e-3        # m
R        = diameter / 2 # outer radius, m
volume   = np.pi * R**2 * height   # m^3

# ---------- 2. Material properties (sourced — see problem_definition.md) ----------
rho = 2720.0   # kg/m^3  (density)
cp  = 900.0    # J/(kg*K) (specific heat)
k   = 0.22     # W/(m*K)  (radial thermal conductivity)
alpha = k / (rho * cp)   # thermal diffusivity, m^2/s

# ---------- 3. Heat generation (Joule heating, I^2 R) ----------
I_charge = 10.0     # A, constant fast-charge current (placeholder — source this)
R_int    = 0.05     # ohm, internal resistance (placeholder — source this)
q_gen = (I_charge**2 * R_int) / volume   # W/m^3

# ---------- 4. Boundary condition parameters ----------
h      = 20.0   # W/(m^2*K), convection coefficient (placeholder — source this)
T_amb  = 25.0   # deg C
T_init = 25.0   # deg C

# ---------- 5. Mesh ----------
n_nodes = 41
r = np.linspace(0, R, n_nodes)
dr = r[1] - r[0]

# ---------- 6. Stability-limited timestep ----------
# Centre node (r=0) needs the tighter bound alpha*dt/dr^2 <= 0.25
# (see explanation below for why this is different from the standard 0.5 bound)
dt_max = 0.25 * dr**2 / alpha
dt = 0.9 * dt_max   # 10% safety margin

total_time = 600.0   # seconds (10 min fast charge)
n_steps = int(np.ceil(total_time / dt))

# ---------- 7. Initialise temperature array ----------
T = np.zeros((n_steps + 1, n_nodes))
T[0, :] = T_init

# ---------- 8. Time-marching loop ----------
for n in range(n_steps):
    T_old = T[n, :]
    T_new = np.empty(n_nodes)

    # --- Centre node (r=0): symmetry + L'Hopital's rule ---
    # As r->0, (1/r)*d/dr(r*dT/dr) -> 2*d2T/dr2, and with the extra
    # factor from the cylindrical term this becomes 4*alpha overall.
    T_new[0] = T_old[0] + dt * (4*alpha*(T_old[1] - T_old[0])/dr**2 + q_gen/(rho*cp))

    # --- Interior nodes ---
    for i in range(1, n_nodes - 1):
        ri = r[i]
        d2T_dr2 = (T_old[i+1] - 2*T_old[i] + T_old[i-1]) / dr**2
        dT_dr   = (T_old[i+1] - T_old[i-1]) / (2*dr)
        T_new[i] = T_old[i] + dt * (alpha*(d2T_dr2 + dT_dr/ri) + q_gen/(rho*cp))

    # --- Surface node (r=R): convective BC via ghost node ---
    # Ghost point T_ghost found from discretising -k dT/dr = h(T_surf - T_amb)
    T_ghost = T_old[n_nodes-2] - (2*dr*h/k) * (T_old[n_nodes-1] - T_amb)
    ri = r[n_nodes - 1]
    d2T_dr2 = (T_ghost - 2*T_old[n_nodes-1] + T_old[n_nodes-2]) / dr**2
    dT_dr   = (T_ghost - T_old[n_nodes-2]) / (2*dr)
    T_new[n_nodes-1] = T_old[n_nodes-1] + dt * (alpha*(d2T_dr2 + dT_dr/ri) + q_gen/(rho*cp))

    T[n+1, :] = T_new

# T now holds the full solution: T[timestep, node]

print("Centre temperature at start:", T[0, 0])
print("Centre temperature at end:", T[-1, 0])
print("Surface temperature at end:", T[-1, -1])
print("Any weird values (NaN)?", np.isnan(T).any())

import matplotlib.pyplot as plt

plt.plot(r*1000, T[-1, :])  # temperature across radius, at the final time
plt.xlabel("radius (mm)")
plt.ylabel("Temperature (C)")
plt.title("Final temperature profile")
plt.show()


