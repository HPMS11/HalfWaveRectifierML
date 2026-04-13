import numpy as np
import matplotlib.pyplot as plt
from circuit_simulator import CircuitSimulator

# range for R: [1, 2500], C: [0.1e-6, 5e-6]
R = 892.8388        # ohms
C = 1.6883e-06     # farads

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
x_test, tpoints = mna.BEuler(x_init, delta_t, T, noise=True)

# Load measured data and build a matching time axis for overlay
x_measured = np.loadtxt("measurements.csv", delimiter=",")

if x_measured.ndim == 1:
    x_measured = x_measured.reshape(-1, 4)

if x_measured.shape[1] != 4:
    raise ValueError(
        f"Expected measurements.csv to have 4 columns, got {x_measured.shape[1]}."
    )

if len(x_measured) == len(tpoints):
    measured_tpoints = tpoints
else:
    measured_tpoints = np.linspace(tpoints[0], tpoints[-1], len(x_measured))

# Create the figure and the first axis (for Volts)
_, ax1 = plt.subplots(figsize=(10, 6))

# Simulated voltages
ax1.plot(tpoints, x_test[:, 0], label="Sim $V_1$")
ax1.plot(tpoints, x_test[:, 1], label="Sim $V_2$", linestyle="--")
ax1.plot(tpoints, x_test[:, 2], label="Sim $V_3$", linestyle=":")

# Measured voltages
ax1.plot(measured_tpoints, x_measured[:, 0], label="Measured $V_1$", alpha=0.75)
ax1.plot(
    measured_tpoints,
    x_measured[:, 1],
    label="Measured $V_2$",
    linestyle="--",
    alpha=0.75,
)
ax1.plot(
    measured_tpoints,
    x_measured[:, 2],
    label="Measured $V_3$",
    linestyle=":",
    alpha=0.75,
)

ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Volt (V)")
ax1.grid(True, linestyle="--", alpha=0.7)

# Create a twin axis for I_E (for mA)
ax2 = ax1.twinx()
ax2.plot(tpoints, x_test[:, 3] * 1000, label="Sim $I_E$", color="red", linewidth=2)
ax2.plot(
    measured_tpoints,
    x_measured[:, 3] * 1000,
    label="Measured $I_E$",
    color="darkred",
    linestyle="--",
    alpha=0.75,
)
ax2.set_ylabel("Current (mA)")


def align_zeros(ax_ref, ax_target):
    ymin_ref, ymax_ref = ax_ref.get_ylim()
    rat = ymax_ref / (ymax_ref - ymin_ref)
    ymin_tar, ymax_tar = ax_target.get_ylim()
    if abs(ymin_tar) > abs(ymax_tar):
        new_ymax = ymin_tar * rat / (rat - 1)
        ax_target.set_ylim(ymin_tar, new_ymax)
    else:
        new_ymin = ymax_tar * (rat - 1) / rat
        ax_target.set_ylim(new_ymin, ymax_tar)


align_zeros(ax1, ax2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Simulated vs Measured Voltage and Current")
plt.tight_layout()
plt.show()
