import numpy as np
import io
from contextlib import redirect_stdout
from group_25_circuit_simulator import CircuitSimulator

# data in csv sorted as: V1, V2, V3, IE
x_test = np.loadtxt("measurements.csv", delimiter=",")


amplitude = 5.0
f = 60
delta_t = 1e-4
T = x_test.shape[0] * delta_t

# Initial state for simulation
x_init = np.zeros((4,))

guesses = [
    (500.0, 0.5e-6),
    (1000.0, 1e-6),
    (2000.0, 2e-6),
    (3000.0, 5e-6),
    (4500.0, 8e-6),
    (988.8775, 1.4015e-06)  #from GBT
]

print(f"Loaded measurement data with shape: {x_test.shape}")
print()

results = []

for R_guess, C_guess in guesses:
    sim = CircuitSimulator(amplitude, f, R_guess, C_guess)

    try:
        with redirect_stdout(io.StringIO()):
            R_est, C_est, cost = sim.GaussNewton(
                R_init=R_guess,
                C_init=C_guess,
                x_init=x_init,
                x_test=x_test,
                delta_t=delta_t,
                T=T,
                max_iter=100,
                noise=True
            )

        results.append([
            f"{R_guess:.4f}",
            f"{C_guess:.4e}",
            f"{R_est:.4f}",
            f"{C_est:.4e}",
            f"{cost:.6e}",
            "OK",
        ])

    except Exception as e:
        results.append([
            f"{R_guess:.4f}",
            f"{C_guess:.4e}",
            "-",
            "-",
            "-",
            f"ERROR: {e}",
        ])

headers = ["R_guess (ohms)", "C_guess (F)", "R_est (ohms)", "C_est (F)", "Final cost", "Status"]
widths = []

for i, header in enumerate(headers):
    column_values = [row[i] for row in results]
    widths.append(max(len(header), *(len(value) for value in column_values)))

header_row = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
separator_row = "-+-".join("-" * widths[i] for i in range(len(headers)))

print(header_row)
print(separator_row)
for row in results:
    print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
