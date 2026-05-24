import numpy as np
import pandas as pd
from src.data_loader import load_ecg_signal, FS
from src.svd_methods import standard_svd, hankel_svd, recursive_svd, randomized_svd, notch_svd_50hz
from src.metrics import calculate_all_metrics
from src.plotting import plot_noise_comparison
import os

# Load clean signal and noisy signal
clean = load_ecg_signal('wave.csv')
noisy_df = pd.read_csv('results/noisy_50hz_interference.csv')
noisy = noisy_df['noisy_signal'].values

print("=" * 60)
print("Testing SVD methods on 50Hz interference (amplitude ~ R-wave)")
print("=" * 60)

# Test all 5 methods
methods = {
    'standard': standard_svd,
    'hankel': hankel_svd,
    'recursive': recursive_svd,
    'randomized': randomized_svd,
    'notch_svd_50hz': notch_svd_50hz,
}

results = []
for name, method in methods.items():
    if name == 'notch_svd_50hz':
        denoised = method(noisy, fs=FS)
    else:
        denoised = method(noisy, rank=5)

    metrics = calculate_all_metrics(clean, noisy, denoised)
    results.append({
        'method': name,
        'snr_improvement': metrics['snr_improvement'],
        'mse': metrics['mse'],
        'correlation': metrics['correlation'],
    })
    print(f"{name:20s}: SNR improvement = {metrics['snr_improvement']:+.2f} dB, correlation = {metrics['correlation']:.4f}")

# Generate comparison plot
def normalize_to_range(x):
    xmin, xmax = x.min(), x.max()
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

clean_norm = normalize_to_range(clean)
noisy_norm = normalize_to_range(noisy)
denoised_dict = {}
for name, method in methods.items():
    if name == 'notch_svd_50hz':
        denoised_dict[name] = method(noisy, fs=FS)
    else:
        denoised_dict[name] = method(noisy, rank=5)

denoised_dict_norm = {k: normalize_to_range(v) for k, v in denoised_dict.items()}

os.makedirs('results', exist_ok=True)
plot_noise_comparison(
    clean_norm, noisy_norm, denoised_dict_norm,
    '50Hz_Notch_SVD_Comparison', 0,
    'results/50hz_notch_svd_comparison.png'
)
print("\nSaved: results/50hz_notch_svd_comparison.png")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('results/50hz_notch_svd_metrics.csv', index=False)
print("Saved: results/50hz_notch_svd_metrics.csv")
