import numpy as np
from circuit_simulator import CircuitSimulator

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
    (4500.0, 8e-6)
]

print(f"Loaded measurement data with shape: {x_test.shape}")
print()

for R_guess, C_guess in guesses:
    sim = CircuitSimulator(amplitude, f, R_guess, C_guess)

    try:
        R_est, C_est, cost = sim.GaussNewton(
            R_init=R_guess,
            C_init=C_guess,
            x_init=x_init,
            x_test=x_test,
            delta_t=delta_t,
            T=T,
            max_iter=30,
            noise=False
        )

        print("--------------------------------------------------")
        print(f"Initial guess: R_guess = {R_guess:.4f} ohms, C_guess = {C_guess:.4e} F")
        print(f"Estimated:     R_est   = {R_est:.4f} ohms, C_est   = {C_est:.4e} F")
        print(f"Final cost:    {cost:.6e}")

    except Exception as e:
        print("--------------------------------------------------")
        print(f"Initial guess: R_guess = {R_guess:.4f} ohms, C_guess = {C_guess:.4e} F")
        print(f"Failed with error: {e}")