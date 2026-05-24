"""
Combined noise experiment: all 6 noise types simultaneously.

Tests 4 SVD methods on ECG signal corrupted by the sum of all 6 noise types,
at 4 SNR levels. Generates comparison plots, metrics summary, and scree plot.
"""

import os
import time

import numpy as np
import pandas as pd

from src.data_loader import load_ecg_signal, FS
from src.noise import (
    generate_baseline_wander,
    generate_powerline_interference,
    generate_emg_noise,
    generate_motion_artifact,
    generate_measurement_error,
    generate_white_noise,
    add_noise,
)
from src.svd_methods import standard_svd, hankel_svd, recursive_svd, randomized_svd
from src.metrics import calculate_all_metrics
from src.plotting import plot_noise_comparison, plot_metrics_summary, plot_singular_values

# Reproducibility
np.random.seed(42)

# Experiment configuration
SNR_LEVELS = [-20, -15, -10, -5]
SVD_METHODS = {
    'standard': standard_svd,
    'hankel': hankel_svd,
    'recursive': recursive_svd,
    'randomized': randomized_svd,
}


def normalize_to_range(x):
    """Normalize signal to [-1, 1] range."""
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-10:
        return x
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0


def generate_combined_noise(n, fs=250):
    """
    Generate the sum of all 6 noise types.

    Parameters
    ----------
    n : int
        Number of samples.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Combined noise signal of shape (n,).
    """
    noise1 = generate_baseline_wander(n, fs)
    noise2 = generate_powerline_interference(n, fs)
    noise3 = generate_emg_noise(n, fs)
    noise4 = generate_motion_artifact(n, fs)
    noise5 = generate_measurement_error(n, fs)
    noise6 = generate_white_noise(n)

    combined = noise1 + noise2 + noise3 + noise4 + noise5 + noise6
    return combined


def main():
    """Run the combined noise denoising experiment."""
    # ── Load clean ECG signal ──
    print("Loading ECG signal...")
    clean = load_ecg_signal('wave.csv')
    n = len(clean)
    fs = FS
    print(f"  Loaded {n} samples at {fs} Hz")

    # ── Generate combined noise (once, reused across SNR levels) ──
    print("Generating combined noise (6 types)...")
    combined_noise = generate_combined_noise(n, fs)
    print(f"  Combined noise variance: {np.var(combined_noise):.4f}")

    # ── Experiment loop ──
    results = []
    total = len(SNR_LEVELS) * len(SVD_METHODS)
    iteration = 0

    for snr_db in SNR_LEVELS:
        # Scale combined noise to target SNR
        noisy, scaled_noise = add_noise(clean, combined_noise, snr_db)
        actual_snr = 10 * np.log10(np.var(clean) / np.var(clean - noisy))
        print(f"\nSNR target={snr_db}dB, actual={actual_snr:.2f}dB")

        # Denoise with each SVD method
        denoised_dict = {}
        for method_name, method_fn in SVD_METHODS.items():
            iteration += 1
            print(f"  [{iteration}/{total}] {method_name} SVD...", end=" ")

            start = time.perf_counter()
            denoised = method_fn(noisy, rank=5)
            runtime = time.perf_counter() - start

            denoised_dict[method_name] = denoised

            metrics = calculate_all_metrics(clean, noisy, denoised)
            print(f"ΔSNR={metrics['snr_improvement']:+.2f}dB  "
                  f"r={metrics['correlation']:.4f}  "
                  f"MSE={metrics['mse']:.2f}  "
                  f"({runtime:.3f}s)")

            results.append({
                'noise_type': 'combined',
                'snr_db': snr_db,
                'method': method_name,
                'snr_improvement': metrics['snr_improvement'],
                'mse': metrics['mse'],
                'correlation': metrics['correlation'],
                'runtime': runtime,
            })

        # ── Save comparison plot for this SNR level ──
        # Normalize all signals for fair visual comparison
        clean_norm = normalize_to_range(clean)
        noisy_norm = normalize_to_range(noisy)
        denoised_dict_norm = {}
        for method_name, denoised in denoised_dict.items():
            denoised_dict_norm[method_name] = normalize_to_range(denoised)

        save_path = f'results/combined_noise_comparison_SNR{snr_db}.png'
        plot_noise_comparison(
            clean_norm, noisy_norm, denoised_dict_norm,
            noise_type='combined_noise',
            snr_db=snr_db,
            save_path=save_path,
        )
        print(f"  Saved {save_path}")

    # ── Save metrics CSV ──
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = 'results/combined_noise_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path}")

    # ── Metrics summary plot ──
    metrics_plot_path = 'results/combined_noise_metrics.png'
    plot_metrics_summary(df, metrics_plot_path)
    print(f"Saved {metrics_plot_path}")

    # ── Scree plot (use combined noisy signal at SNR=0dB) ──
    noisy_0db, _ = add_noise(clean, combined_noise, 0)
    scree_path = 'results/combined_noise_scree.png'
    plot_singular_values(noisy_0db, L=500, save_path=scree_path)
    print(f"Saved {scree_path}")

    print("\nDone! All results in results/")


if __name__ == '__main__':
    main()
