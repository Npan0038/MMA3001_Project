# MMA3001_Project
# EV Battery Cell Thermal Model with ML Surrogate (MMA3001)

A finite-difference model of an 18650 lithium-ion cell heating up during charging. The solver is validated against an exact solution, then used to train machine-learning surrogates (decision tree and neural network), which are compared with the solver for speed and accuracy.

## What it does
1. **FTCS solver** for 1D radial heat conduction in the cell, with Joule heating (I²R) inside and convective cooling at the surface.
2. **Validation** against an exact Bessel-series solution, plus a mesh convergence study (second-order convergence, measured slope 2.00).
3. **Dataset** of 300 solver runs over charging current, cooling and ambient temperature.
4. **Surrogates** (linear baseline, decision tree, neural network) trained on that dataset and compared with the solver.

## Files
| File | Purpose |
|---|---|
| `ftcs_solver.py` | Main FTCS solver script (heating case) |
| `analytical.py` | Exact solution used for validation (importable module) |
| `validate.py` | Compares FTCS against the exact solution |
| `convergence.py` | Mesh convergence study, saves `convergence.png` |
| `generate_dataset.py` | Runs the solver 300 times, saves `dataset.csv` |
| `train_surrogate.py` | Trains and compares the surrogates, saves `parity_plots.png` |
| `tests/test_solver.py` | 15 pytest tests; report in `test_report.txt` |
| `docs/` | HTML documentation generated with pdoc |
| `results.txt` | Printed output of the surrogate training run |

## How to run
```
pip install -r requirements.txt
python validate.py
python convergence.py
python generate_dataset.py
python -m pytest tests -v
python -m pdoc analytical generate_dataset convergence -o docs
```
`train_surrogate.py` needs scikit-learn. It was run on Google Colab: upload `dataset.csv`, `generate_dataset.py` and `train_surrogate.py`, then run `%cd /` and `%run train_surrogate.py`. The solver, tests and docs were run locally on Python 3.15 with numpy, matplotlib, pytest and pdoc.

## Results (test set of 60 runs, temperatures in °C)
| Model | RMSE | MAE | Max error | R² | Speed vs solver |
|---|---|---|---|---|---|
| Linear regression (baseline) | 1.051 | 0.637 | 4.792 | 0.9877 | 1,261x faster |
| Decision tree | 1.710 | 1.224 | 6.468 | 0.9675 | 771x faster |
| Neural network | 0.516 | 0.367 | 2.241 | 0.9970 | 172x faster |

The solver took 121 ms per run on Colab. The neural network is the most accurate surrogate and still about 170 times faster than the solver, but its worst-case error is 2.2 °C.

## Inputs and sources
- Internal resistance 19.23 mΩ (DC, typical), maximum charge current 4.0 A and maximum cell weight 45 g: Samsung SDI, "Introduction of INR18650-25R" datasheet (Mar 2013).
- Convection coefficient 28.4 W/m²K (natural convection in air): Tripathi and Marconnet, "Multi-Physics Numerical Analysis of Single-phase Immersion Cooling for Thermal Management of Li-Ion Batteries", arXiv:2404.14428, Section II-C.
- Specific heat 900 J/kg·K (averaged from the reported 814 to 972 range) and radial conductivity 0.22 W/m·K: J. Electrochem. Soc. 162(14), A2789 to A2795 (2015), doi:10.1149/2.0871514jes.
- Density 2720 kg/m³ is derived from the 45 g datasheet weight and the 18 mm x 65 mm cell volume.
- The low end of the cooling range (h = 5 W/m²K) is an assumed poorly ventilated pack and is not from a source.

## Limitations
- 1D radial conduction only, with constant material properties and constant charging current.
- The conductivity and specific heat come from a LiCoO₂ cell study, while the resistance comes from an NCA cell datasheet.
- Validation shows the numerics are correct, not that the model matches a real cell. There is no experimental data.
- The surrogates are only valid inside the sampled ranges (1 to 4 A, h of 5 to 28.4, ambient 15 to 45 °C). The test set is only 60 runs from one random split.
- Crank-Nicolson was planned but not implemented.

## **AI acknowledgement**
This project was developed with help from Claude (Anthropic). The full reflection is in the written report.
