import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = np.genfromtxt("dataset.csv", delimiter=",", names=True)
X = np.column_stack([data["current_A"], data["h_W_m2K"], data["T_amb_C"]])   
y = data["T_centre_final_C"]                                                 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"train runs: {len(X_train)}, test runs: {len(X_test)}")


models = {
    "Linear regression (baseline)": LinearRegression(),
    "Decision tree": DecisionTreeRegressor(max_depth=8, random_state=42),
    "Neural network": make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=3000, random_state=42),
    ),
}


results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    error = pred - y_test
    results[name] = {
        "pred": pred,
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "MAE": mean_absolute_error(y_test, pred),
        "max_err": np.max(np.abs(error)),
        "R2": r2_score(y_test, pred),
    }

print("\nAccuracy on the test set (temperatures in deg C):")
print(f"{'model':32s} {'RMSE':>8s} {'MAE':>8s} {'max err':>8s} {'R2':>8s}")
for name, r in results.items():
    print(f"{name:32s} {r['RMSE']:8.3f} {r['MAE']:8.3f} {r['max_err']:8.3f} {r['R2']:8.4f}")


try:
    from generate_dataset import solve
    t0 = time.perf_counter()
    for i in range(20):
        solve(X_test[i % len(X_test), 0], X_test[i % len(X_test), 1], X_test[i % len(X_test), 2])
    solver_time = (time.perf_counter() - t0) / 20
    print(f"\nSolver time per run (timed here): {solver_time * 1000:.2f} ms")
except ImportError:
    solver_time = np.mean(data["solver_runtime_s"])
    print(f"\nWARNING: generate_dataset.py not found, using the runtime saved in the CSV "
          f"({solver_time * 1000:.2f} ms). That was timed on a different machine, so the comparison is not fair.")

print("Surrogate time per single prediction:")
for name, model in models.items():
    t0 = time.perf_counter()
    for _ in range(500):
        model.predict(X_test[:1])
    t = (time.perf_counter() - t0) / 500
    results[name]["time"] = t
    print(f"  {name:30s} {t * 1000:.3f} ms   ({solver_time / t:,.0f}x faster than the solver)")


fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
lo, hi = y_test.min() - 1, y_test.max() + 1
for ax, (name, r) in zip(axes, results.items()):
    ax.scatter(y_test, r["pred"], s=14)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_title(f"{name}\nRMSE {r['RMSE']:.3f} C")
    ax.set_xlabel("solver temperature (C)")
    ax.set_ylabel("surrogate prediction (C)")
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("parity_plots.png", dpi=150)
plt.show()
