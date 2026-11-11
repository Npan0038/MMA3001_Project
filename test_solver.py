"""Tests for the analytical solution, the FTCS solver and the mesh convergence.

Run from the project folder with:  python -m pytest tests -v
"""
import numpy as np
import pytest

from analytical import bessel_j0, bessel_j1, f, find_eigenvalues, analytical_solution
from generate_dataset import solve
from convergence import run_ftcs


# ---------- Bessel functions and eigenvalues ----------

def test_bessel_known_values():
    """J0(0)=1, J1(0)=0, J1(1)=0.4400505857, and J0 is zero at 2.404825557695773."""
    assert bessel_j0(0.0) == pytest.approx(1.0)
    assert bessel_j1(0.0) == pytest.approx(0.0)
    assert bessel_j1(1.0) == pytest.approx(0.4400505857, abs=1e-9)
    assert bessel_j0(2.404825557695773) == pytest.approx(0.0, abs=1e-9)


def test_first_eigenvalue_matches_textbook_for_Bi_1():
    """For Bi = 1 the first cylinder eigenvalue is about 1.2558 (standard heat transfer table value)."""
    assert find_eigenvalues(1.0, 1)[0] == pytest.approx(1.2558, abs=1e-3)


def test_eigenvalues_satisfy_the_equation():
    """Every eigenvalue must make f(lam) = 0, so the residual should be tiny."""
    Bi = 0.818
    for lam in find_eigenvalues(Bi, 6):
        assert abs(f(lam, Bi)) < 1e-8


def test_eigenvalues_are_increasing():
    """Eigenvalues come out in increasing order, one per interval."""
    lams = find_eigenvalues(0.818, 6)
    assert all(b > a for a, b in zip(lams, lams[1:]))


def test_small_Bi_limit():
    """For very small Bi the first eigenvalue approaches sqrt(2*Bi)."""
    assert find_eigenvalues(0.01, 1)[0] == pytest.approx(np.sqrt(2 * 0.01), rel=0.01)


# ---------- analytical solution ----------

R, ALPHA, BI = 9e-3, 0.22 / (2720.0 * 900.0), 20 * 9e-3 / 0.22

def test_analytical_cools_towards_ambient():
    """A hot cell in cooler air must approach the ambient temperature at long times."""
    T = analytical_solution(0.0, 1e5, R, ALPHA, BI, 80.0, 25.0)
    assert T == pytest.approx(25.0, abs=1e-6)


def test_analytical_centre_hotter_than_surface_while_cooling():
    """While cooling, the centre is hotter than the surface."""
    t = 300.0
    centre = analytical_solution(0.0, t, R, ALPHA, BI, 80.0, 25.0)
    surface = analytical_solution(R, t, R, ALPHA, BI, 80.0, 25.0)
    assert 25.0 < surface < centre < 80.0


def test_analytical_temperature_falls_with_time():
    """The centre temperature must fall steadily as time passes."""
    temps = [analytical_solution(0.0, t, R, ALPHA, BI, 80.0, 25.0) for t in (60, 120, 300, 600)]
    assert all(b < a for a, b in zip(temps, temps[1:]))


# ---------- FTCS solver (heating case used for the dataset) ----------

def test_no_current_means_no_heating():
    """With zero current there is no heat source, so the cell stays at ambient temperature."""
    centre, surface = solve(0.0, 20.0, 30.0)
    assert centre == pytest.approx(30.0, abs=1e-9)
    assert surface == pytest.approx(30.0, abs=1e-9)


def test_charging_heats_the_cell_centre_hottest():
    """With current flowing, ambient < surface < centre."""
    centre, surface = solve(4.0, 10.0, 25.0)
    assert 25.0 < surface < centre


def test_more_current_gives_higher_temperature():
    """Doubling the charging current must raise the temperature."""
    low, _ = solve(2.0, 15.0, 25.0)
    high, _ = solve(4.0, 15.0, 25.0)
    assert high > low


def test_better_cooling_gives_lower_temperature():
    """A higher convection coefficient must keep the cell cooler."""
    poor, _ = solve(4.0, 5.0, 25.0)
    good, _ = solve(4.0, 28.4, 25.0)
    assert good < poor


def test_solver_results_are_finite():
    """No NaN or infinity across the whole range used in the dataset."""
    for current in (1.0, 4.0):
        for h in (5.0, 28.4):
            for T_amb in (15.0, 45.0):
                centre, surface = solve(current, h, T_amb)
                assert np.isfinite(centre) and np.isfinite(surface)


# ---------- validation against the exact solution ----------

def test_ftcs_matches_analytical_solution():
    """On a 41-node mesh the FTCS result agrees with the exact solution to within 0.01 C."""
    _, err = run_ftcs(41)
    assert err < 0.01


def test_ftcs_converges_at_second_order():
    """Halving the mesh spacing should cut the error by about 4 (second-order convergence)."""
    _, err_coarse = run_ftcs(21)
    _, err_fine = run_ftcs(41)
    assert err_coarse / err_fine == pytest.approx(4.0, rel=0.15)