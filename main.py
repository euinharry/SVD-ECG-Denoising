"""
Main simulation pipeline for ECG denoising with SVD methods.

Runs a full factorial experiment:
  6 noise types x 4 SNR levels x 4 SVD methods = 96 conditions

Collects SNR improvement, MSE, correlation, and runtime for each condition.
Saves results to results/metrics.csv.
"""

import os
import time

import numpy as np
import pandas as pd

from src.data_loader import load_ecg_signal
from src.noise import generate_noise_by_type, add_noise
from src.svd_methods import standard_svd, hankel_svd, recursive_svd, randomized_svd
from src.metrics import calculate_all_metrics
from src.plotting import plot_noise_comparison, plot_metrics_summary, plot_singular_values

# Reproducibility
np.random.seed(42)

# Experiment configuration
NOISE_TYPES = ['baseline', 'powerline', 'emg', 'motion', 'measurement', 'white']
NOISE_TYPE_LABELS = [
    'baseline_wander', 'powerline', 'emg',
    'motion_artifact', 'measurement_error', 'white_noise',
]
SNR_LEVELS = [-5, 0, 5, 10]
SVD_METHODS = {
    'standard': standard_svd,
    'hankel': hankel_svd,
    'recursive': recursive_svd,
    'randomized': randomized_svd,
}


def main():
    """Run the full ECG denoising simulation pipeline."""
    # Load clean ECG signal
    print("Loading ECG signal...")
    clean = load_ecg_signal('wave.csv')
    n = len(clean)
    print(f"  Loaded {n} samples")

    # Total iterations for progress
    total = len(NOISE_TYPES) * len(SNR_LEVELS) * len(SVD_METHODS)
    iteration = 0
    results = []

    # Triple loop: noise_type x snr_db x method
    for noise_type, noise_label in zip(NOISE_TYPES, NOISE_TYPE_LABELS):
        # Generate noise once per type (reused across SNR levels)
        noise = generate_noise_by_type(n, noise_type)

        for snr_db in SNR_LEVELS:
            # Add noise at this SNR level
            noisy, _ = add_noise(clean, noise, snr_db)

            for method_name, method_fn in SVD_METHODS.items():
                iteration += 1
                print(f"[{iteration}/{total}] {noise_label} SNR={snr_db:+d}dB method={method_name}")

                # Denoise and measure runtime
                start = time.perf_counter()
                denoised = method_fn(noisy, rank=5)
                runtime = time.perf_counter() - start

                # Calculate metrics
                metrics = calculate_all_metrics(clean, noisy, denoised)

                # Record result
                results.append({
                    'noise_type': noise_label,
                    'snr_db': snr_db,
                    'method': method_name,
                    'snr_improvement': metrics['snr_improvement'],
                    'mse': metrics['mse'],
                    'correlation': metrics['correlation'],
                    'runtime': runtime,
                })

    # Save results
    print("\nSaving results...")
    os.makedirs('results', exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv('results/metrics.csv', index=False)
    print(f"  Saved {len(df)} rows to results/metrics.csv")

    # Generate comparison plots
    print("\nGenerating plots...")
    for noise_type, noise_label in zip(NOISE_TYPES, NOISE_TYPE_LABELS):
        noise = generate_noise_by_type(n, noise_type)
        noisy, _ = add_noise(clean, noise, 5)  # Use SNR=5dB for comparison
        denoised_dict = {}
        for method_name, method_fn in SVD_METHODS.items():
            denoised_dict[method_name] = method_fn(noisy, rank=5)
        save_path = f'results/{noise_label}_comparison.png'
        plot_noise_comparison(clean, noisy, denoised_dict, noise_label, 5, save_path)
        print(f"  Saved {save_path}")

    # Generate metrics summary
    print("Generating metrics summary...")
    plot_metrics_summary(df, 'results/metrics_summary.png')
    print("  Saved results/metrics_summary.png")

    # Generate singular values plot
    print("Generating singular values plot...")
    plot_singular_values(clean, L=500, save_path='results/singular_values.png')
    print("  Saved results/singular_values.png")

    print("Done!")


if __name__ == '__main__':
    main()
