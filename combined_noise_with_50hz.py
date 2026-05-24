"""
Combined noise test: 50Hz interference + baseline wander + motion artifacts + white noise + measurement error.

Tests all 5 SVD methods (standard, hankel, recursive, randomized, notch_svd_50hz)
on a signal degraded with multiple simultaneous noise sources.
"""

import numpy as np
import pandas as pd
from src.data_loader import load_ecg_signal, FS
from src.noise import (
    generate_baseline_wander,
    generate_motion_artifact,
    generate_measurement_error,
    generate_white_noise,
)
from src.svd_methods import standard_svd, hankel_svd, recursive_svd, randomized_svd, notch_svd_50hz
from src.metrics import calculate_all_metrics
from src.plotting import plot_noise_comparison
import os

np.random.seed(42)

# Load clean signal
clean = load_ecg_signal('wave.csv')
n = len(clean)

# Load the 50Hz-interfered signal (already has 50Hz at 80% R-wave amplitude)
noisy_50hz = pd.read_csv('results/noisy_50hz_interference.csv')['noisy_signal'].values

# Generate additional noise types
baseline = generate_baseline_wander(n, FS)  # Baseline wander
motion = generate_motion_artifact(n, FS)    # Motion artifacts
measurement = generate_measurement_error(n, FS)  # Equipment noise
white = generate_white_noise(n)             # White noise

# Scale additional noises to be significant but not overwhelming
# Use 30% of R-wave amplitude for each
r_wave_amp = np.max(np.abs(clean))
scale = 0.3 * r_wave_amp

baseline_scaled = baseline / np.std(baseline) * scale
motion_scaled = motion / np.std(motion) * scale
measurement_scaled = measurement / np.std(measurement) * scale
white_scaled = white / np.std(white) * scale

# Combine: 50Hz + baseline + motion + measurement + white
noisy_combined = noisy_50hz + baseline_scaled + motion_scaled + measurement_scaled + white_scaled

print("=" * 60)
print("Combined noise: 50Hz + baseline + motion + measurement + white")
print(f"R-wave amplitude: {r_wave_amp:.2f}")
print(f"Additional noise scale: {scale:.2f} (30% of R-wave)")
print("=" * 60)

# Test all 5 SVD methods
methods = {
    'standard': lambda x: standard_svd(x, rank=5),
    'hankel': lambda x: hankel_svd(x, rank=5),
    'recursive': lambda x: recursive_svd(x, rank=5),
    'randomized': lambda x: randomized_svd(x, rank=5),
    'notch_svd_50hz': lambda x: notch_svd_50hz(x, fs=FS),
}

results = []
denoised_dict = {}

for name, method in methods.items():
    denoised = method(noisy_combined)
    denoised_dict[name] = denoised
    metrics = calculate_all_metrics(clean, noisy_combined, denoised)
    results.append({
        'method': name,
        'snr_improvement': metrics['snr_improvement'],
        'mse': metrics['mse'],
        'correlation': metrics['correlation'],
    })
    print(f"{name:20s}: SNR improvement = {metrics['snr_improvement']:+.2f} dB, correlation = {metrics['correlation']:.4f}")

# Normalize for plotting
def normalize_to_range(x):
    xmin, xmax = x.min(), x.max()
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

clean_norm = normalize_to_range(clean)
noisy_norm = normalize_to_range(noisy_combined)
denoised_dict_norm = {k: normalize_to_range(v) for k, v in denoised_dict.items()}

# Generate comparison plot
os.makedirs('results', exist_ok=True)
plot_noise_comparison(
    clean_norm, noisy_norm, denoised_dict_norm,
    'Combined_Noise_50Hz_Baseline_Motion_White', 0,
    'results/combined_noise_with_50hz_comparison.png'
)
print("\nSaved: results/combined_noise_with_50hz_comparison.png")

# Save metrics
results_df = pd.DataFrame(results)
results_df.to_csv('results/combined_noise_with_50hz_metrics.csv', index=False)
print("Saved: results/combined_noise_with_50hz_metrics.csv")

# Save noisy signal
noisy_df = pd.DataFrame({'noisy_combined': noisy_combined})
noisy_df.to_csv('results/noisy_combined_with_50hz.csv', index=False)
print("Saved: results/noisy_combined_with_50hz.csv")
