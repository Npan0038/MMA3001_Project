
def bessel_j0(x):
    # J0 as a simple sum of terms (a power series)
    term = 1.0
    total = 1.0
    for k in range(60):
        term *= -(x / 2) ** 2 / ((k + 1) ** 2)
        total += term
    return total

def bessel_j1(x):
    # J1, same idea
    term = x / 2
    total = term
    for k in range(60):
        term *= -(x / 2) ** 2 / ((k + 1) * (k + 2))
        total += term
    return total

def f(lam, Bi):
    # we want the lam values where this equals zero
    return lam * bessel_j1(lam) - Bi * bessel_j0(lam)

def find_eigenvalues(Bi, n_terms):
    roots = []
    step = 0.01
    left = 1e-9
    while len(roots) < n_terms:
        right = left + step
        # a sign change means a root is hiding between left and right
        if f(left, Bi) * f(right, Bi) < 0:
            lo, hi = left, right
            for _ in range(60):          # bisection: keep halving the gap
                mid = (lo + hi) / 2
                if f(lo, Bi) * f(mid, Bi) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2)
        left = right
    return roots

Bi = 20 * 0.009 / 0.22
print(find_eigenvalues(Bi, 6))

"_________________________________________________________________________________________"

import math

def bessel_j0(x):
    # J0 as a simple sum of terms (a power series)
    term = 1.0
    total = 1.0
    for k in range(60):
        term *= -(x / 2) ** 2 / ((k + 1) ** 2)
        total += term
    return total

def bessel_j1(x):
    # J1, same idea
    term = x / 2
    total = term
    for k in range(60):
        term *= -(x / 2) ** 2 / ((k + 1) * (k + 2))
        total += term
    return total

def f(lam, Bi):
    # we want the lam values where this equals zero
    return lam * bessel_j1(lam) - Bi * bessel_j0(lam)

def find_eigenvalues(Bi, n_terms):
    roots = []
    step = 0.01
    left = 1e-9
    while len(roots) < n_terms:
        right = left + step
        # a sign change means a root is hiding between left and right
        if f(left, Bi) * f(right, Bi) < 0:
            lo, hi = left, right
            for _ in range(60):          # bisection: keep halving the gap
                mid = (lo + hi) / 2
                if f(lo, Bi) * f(mid, Bi) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append((lo + hi) / 2)
        left = right
    return roots

def analytical_solution(r, t, R, alpha, Bi, T_i, T_inf, n_terms=8):
    # n_terms=8 is plenty here and is as high as this simple Bessel code stays accurate
    lams = find_eigenvalues(Bi, n_terms)
    Fo = alpha * t / R**2                      # dimensionless time
    theta = 0.0
    for lam in lams:
        J0 = bessel_j0(lam)
        J1 = bessel_j1(lam)
        C = 2 * J1 / (lam * (J0**2 + J1**2))   # weight of this term
        theta += C * math.exp(-lam**2 * Fo) * bessel_j0(lam * r / R)
    return T_inf + (T_i - T_inf) * theta       # convert back to real temperature

if __name__ == "__main__":
    Bi = 20 * 0.009 / 0.22
    print(find_eigenvalues(Bi, 6))
