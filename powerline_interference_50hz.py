import numpy as np
from src.data_loader import load_ecg_signal, FS
from src.svd_methods import standard_svd, hankel_svd, recursive_svd, randomized_svd
from src.metrics import calculate_all_metrics
from src.plotting import plot_noise_comparison
import os

# Load clean ECG
clean = load_ecg_signal('wave.csv')
n = len(clean)

# Find R-wave amplitude (peak of clean signal)
r_wave_amplitude = np.max(np.abs(clean))
print(f"R-wave amplitude: {r_wave_amplitude:.2f}")

# Generate 50Hz interference with amplitude close to R-wave
# Use 80% of R-wave amplitude
noise_amplitude = 0.8 * r_wave_amplitude
t = np.arange(n) / FS
phase = np.random.uniform(0, 2 * np.pi)
interference = noise_amplitude * np.sin(2 * np.pi * 50 * t + phase)

# Add interference to clean signal
noisy = clean + interference

# Save noisy signal to CSV
import pandas as pd
noisy_df = pd.DataFrame({'noisy_signal': noisy})
noisy_df.to_csv('results/noisy_50hz_interference.csv', index=False)
print(f"Saved noisy signal to results/noisy_50hz_interference.csv")

# Apply 4 SVD methods
denoised_dict = {}
methods = {
    'standard': standard_svd,
    'hankel': hankel_svd,
    'recursive': recursive_svd,
    'randomized': randomized_svd
}

for name, method in methods.items():
    denoised = method(noisy, rank=5)
    denoised_dict[name] = denoised
    metrics = calculate_all_metrics(clean, noisy, denoised)
    print(f"{name}: SNR improvement = {metrics['snr_improvement']:.2f} dB, correlation = {metrics['correlation']:.4f}")

# Normalize for plotting
def normalize_to_range(x):
    xmin, xmax = x.min(), x.max()
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

clean_norm = normalize_to_range(clean)
noisy_norm = normalize_to_range(noisy)
denoised_dict_norm = {k: normalize_to_range(v) for k, v in denoised_dict.items()}

# Generate comparison plot
os.makedirs('results', exist_ok=True)
plot_noise_comparison(
    clean_norm, noisy_norm, denoised_dict_norm,
    '50Hz_Powerline_Interference', 0,  # SNR not applicable here
    'results/50hz_interference_comparison.png'
)
print("Saved: results/50hz_interference_comparison.png")
