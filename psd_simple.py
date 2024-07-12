import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import linregress

# Generate some example data
np.random.seed(0)
fs = 1000  # Sampling frequency
N = 1e5  # Number of data points
time = np.arange(N) / fs
data = np.sin(2 * np.pi * 10 * time) + np.random.normal(scale=0.5, size=time.shape)

# Compute the Power Spectral Density (PSD) using Welch's method
frequencies, psd = welch(data, fs, nperseg=1024)

# Fit a line to the log-log plot of the PSD
log_frequencies = np.log10(frequencies[1:])  # Skip the zero frequency component
log_psd = np.log10(psd[1:])
slope, intercept, r_value, p_value, std_err = linregress(log_frequencies, log_psd)

# Create the best fit line
best_fit_line = 10**(slope * log_frequencies + intercept)

# Plot the PSD and the best fit line
plt.figure(figsize=(10, 6))
plt.loglog(frequencies, psd, label='Power Spectral Density')
plt.loglog(frequencies[1:], best_fit_line, label='Best Fit Line', linestyle='--')

# Add labels and title
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Spectral Density [V^2/Hz]')
plt.title('Power Spectral Density with Best Fit Line')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
