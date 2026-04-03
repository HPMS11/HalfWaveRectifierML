import numpy as np
import matplotlib.pyplot as plt
from circuit_simulator import CircuitSimulator
from helper_functions import plot_data

# range for R: [1, 2500], C: [0.1e-6, 5e-6]
R = 983.7770         # ohms
C = 1.5314e-06      # farads

# Choose source parameters
amplitude = 5     # volts
f = 60             # Hz

# Simulation settings
delta_t = 1e-4    # seconds
T = 5e-2         # total simulation time

# Initial condition: [V1, V2, V3, IE]
x_init = np.zeros((4,))

# Create simulator
mna = CircuitSimulator(amplitude, f, R, C)

# Run transient simulation
x_test, tpoints = mna.BEuler(x_init, delta_t, T, noise=False)

# Plot results
plot_data(x_test, tpoints)