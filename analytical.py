"""Exact (analytical) solution for a cooling solid cylinder with convection at the surface.

This is the reference solution for validating the FTCS solver. With no heat generation, a solid
cylinder that starts at a uniform temperature and sits in air at a different temperature
has a known series solution built from Bessel functions. The functions here compute it
using only basic maths (no scipy needed):

    theta(r, t) = sum over n of  C_n * exp(-lam_n**2 * Fo) * J0(lam_n * r / R)

where lam_n are the eigenvalues, found from  lam * J1(lam) = Bi * J0(lam).
"""
import math


def bessel_j0(x):
    """Bessel function of the first kind, order 0, as a power series.

    Accurate for the arguments used here (x up to about 30).
    """
    term = 1.0
    total = 1.0
    for k in range(60):
        term *= -(x / 2) ** 2 / ((k + 1) ** 2)
        total += term
    return total


def bessel_j1(x):
    """Bessel function of the first kind, order 1, as a power series.

    Accurate for the arguments used here (x up to about 30).
    """
    term = x / 2
    total = term
    for k in range(60):
        term *= -(x / 2) ** 2 / ((k + 1) * (k + 2))
        total += term
    return total


def f(lam, Bi):
    """Eigenvalue equation lam*J1(lam) - Bi*J0(lam); the eigenvalues are where this is zero.

    Args:
        lam: trial eigenvalue.
        Bi: Biot number, h*R/k (surface convection compared with internal conduction).
    """
    return lam * bessel_j1(lam) - Bi * bessel_j0(lam)


def find_eigenvalues(Bi, n_terms):
    """Find the first n_terms eigenvalues for a given Biot number.

    Walks along in small steps looking for a sign change of f (a root is hiding in that
    gap), then halves the gap 60 times (bisection) to pin the root down.

    Args:
        Bi: Biot number, h*R/k.
        n_terms: how many eigenvalues to return. Keep this at 8 or fewer, because the
            power-series Bessel functions lose accuracy for large arguments.

    Returns:
        List of eigenvalues in increasing order.
    """
    roots = []
    step = 0.01
    left = 1e-9
    while len(roots) < n_terms:
        right = left + step
    
        if f(left, Bi) * f(right, Bi) < 0:
            lo, hi = left, right
            for _ in range(60):          
                mid = (lo + hi) / 2
                if f(lo, Bi) * f(mid, Bi) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2)
        left = right
    return roots


def analytical_solution(r, t, R, alpha, Bi, T_i, T_inf, n_terms=8):
    """Exact temperature in a cooling cylinder at radius r and time t.

    Args:
        r: radial position (m), 0 at the centre and R at the surface.
        t: time since the cell was placed in the air (s).
        R: cylinder radius (m).
        alpha: thermal diffusivity k/(rho*cp) (m^2/s).
        Bi: Biot number, h*R/k.
        T_i: initial uniform temperature (C).
        T_inf: ambient air temperature (C).
        n_terms: number of series terms (8 is plenty).

    Returns:
        Temperature in C.
    """
    lams = find_eigenvalues(Bi, n_terms)
    Fo = alpha * t / R**2                     
    theta = 0.0
    for lam in lams:
        J0 = bessel_j0(lam)
        J1 = bessel_j1(lam)
        C = 2 * J1 / (lam * (J0**2 + J1**2))   
        theta += C * math.exp(-lam**2 * Fo) * bessel_j0(lam * r / R)
    return T_inf + (T_i - T_inf) * theta       


if __name__ == "__main__":
    Bi = 20 * 0.009 / 0.22
    print(find_eigenvalues(Bi, 6))
