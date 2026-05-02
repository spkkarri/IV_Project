import matlab.engine
import matplotlib.pyplot as plt
import numpy as np

print("Starting Engine...")
eng = matlab.engine.start_matlab()

# Define System
num = matlab.double([1])
den = matlab.double([1, 3, 2])

# Push to workspace and create system
eng.workspace['num'] = num
eng.workspace['den'] = den
eng.eval("sys = tf(num, den);", nargout=0)

# --- THE FIX ---
print("Calculating step response data...")

# 1. Calculate y and t INSIDE MATLAB's workspace
# We use eval so MATLAB resolves 'sys' correctly
eng.eval("[y, t] = step(sys);", nargout=0)

# 2. Pull the resulting arrays from the MATLAB workspace to Python
y = eng.workspace['y']
t = eng.workspace['t']

# Convert to Numpy arrays and flatten (MATLAB returns column vectors)
t_py = np.array(t).flatten()
y_py = np.array(y).flatten()

# Plot using Python
print("Plotting in Python...")
plt.figure()
plt.plot(t_py, y_py)
plt.title("Step Response (Calculated in MATLAB, Plotted in Python)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.savefig("tf_plot_python_native.jpeg", dpi=300)

print("Plot saved as tf_plot_python_native.jpeg")
eng.quit()