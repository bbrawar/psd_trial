import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import linregress

def generate_noise(N, alpha=1):
    """Generate red noise (1/f noise) using a power law.
    
    Parameters:
    N (int): Number of data points
    alpha (float): Exponent of the power law. Alpha=1 corresponds to red noise.
    
    Returns:
    np.array: Red noise time series
    """
    freqs = np.fft.fftfreq(N)
    psd = np.zeros_like(freqs)
    psd[freqs != 0] = np.abs(freqs[freqs != 0]) ** (-alpha / 2.0)
    phases = np.random.uniform(0, 2 * np.pi, len(freqs))
    noise = np.fft.ifft(psd * (np.cos(phases) + 1j * np.sin(phases))).real
    return noise

# Parameters
N = 10000  # Number of data points
alpha = 1  # Exponent for red noise

# Generate red noise
red_noise = generate_red_noise(N, alpha)

# Compute the Power Spectral Density (PSD) using Welch's method
fs = 1  # Sampling frequency
frequencies, psd = welch(red_noise, fs, nperseg=1024)

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
plt.title('Power Spectral Density of Red Noise with Best Fit Line')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
