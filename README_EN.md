# SVD-ECG-Denoising

[![Python](https://img.shields.io/badge/Python-3.8%2b-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![SciPy](https://img.shields.io/badge/SciPy-1.7%2b-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2b-013243?logo=numpy&logoColor=white)](https://numpy.org/)

---

## Project Overview

A systematic comparison of **12 SVD-based ECG denoising methods** covering both traditional approaches and advanced variants with signal preprocessing. The project simulates **6 real-world noise types** at **4 SNR levels** and evaluates denoising performance using multiple quantitative metrics.

This framework lets researchers and practitioners compare SVD-based denoising strategies side by side under controlled, reproducible conditions. All methods operate on the same input data and are evaluated with identical metrics, ensuring a fair comparison.

---

## Noise Types

Six clinically relevant noise types commonly found in ECG recordings:

| # | Noise Type | Frequency Range | Description |
|---|-----------|-----------------|-------------|
| 1 | Baseline Wander | 0.05 - 0.5 Hz | Low-frequency drift from respiration, body movement, or electrode impedance changes |
| 2 | Powerline Interference | 50 Hz | Sinusoidal interference from mains electricity coupling into the recording circuit |
| 3 | EMG (Electromyography) | 20 - 500 Hz | Muscle contraction artifacts, modeled as bandpass-filtered white noise |
| 4 | Motion Artifact | Transient | Random step and ramp transients from electrode displacement or cable motion |
| 5 | Measurement Error | 1/f pink noise | Instrumentation noise with power spectral density inversely proportional to frequency |
| 6 | White Noise | Broadband | Gaussian thermal noise from electronic components, uniform across all frequencies |

---

## SVD Denoising Methods

### Traditional Methods (1-6)

These methods operate directly on the raw noisy signal without any preprocessing.

| # | Method | Description |
|--|--------|-------------|
| 1 | **Basic SVD Truncation** | Hankel matrix (L = N/2), full SVD, retain top 3 singular values, reconstruct via anti-diagonal averaging |
| 2 | **SSA (L=60)** | Singular Spectrum Analysis with trajectory matrix of window length 60, keep top 5 components |
| 3 | **Optimal Rank SVD** | Automatically detects optimal truncation rank using curvature-based knee detection on the singular value spectrum |
| 4 | **Block-SVD** | Processes signal in 128-sample overlapping blocks (64-sample hop) with rank-2 approximation and overlap-add reconstruction |
| 5 | **Targeted SVD** | Multi-stage: baseline removal via SSA, periodic interference cancellation via period-reshape, then SSA with component selection by peak-ratio and autocorrelation scoring |
| 6 | **Multichannel SVD** | Wiener-like soft thresholding on Hankel singular values using noise variance estimated from the tail of the SV spectrum |

### New Methods (7-12)

These methods apply notch filtering (50/60 Hz) and bandpass filtering (0.5 - 40 Hz) before SVD denoising.

| # | Method | Description |
|--|--------|-------------|
| 7 | **Notch + SSA** | Preprocessing then SSA (L=80) with component selection by correlation (> 0.3) against a QRS template extracted from the signal |
| 8 | **Beat-Aligned SVD** | Detect R-peaks, extract beat windows ([-0.3s, +0.5s]), stack into matrix, apply rank-3 SVD, reconstruct via overlap-add with Hanning windowing |
| 9 | **Iterative SSA** | 5 iterations of SSA with progressive component pruning (12 down to 5), keeping components most correlated with the current estimate |
| 10 | **Freq-Banded SVD** | Decompose signal into 5 sub-bands ([0.5-4, 4-8, 8-15, 15-25, 25-40] Hz), apply SSA (L=32) to each band independently, sum the reconstructions |
| 11 | **Cross-Validated SVD** | SSA (L=60) with automatic rank selection (2-15) by maximizing a composite score of R-peak height preservation and autocorrelation at the R-R interval |
| 12 | **Combined Best** | Use Beat-Aligned SVD result as guide signal, select SSA components (L=80) whose correlation with the guide exceeds 0.4 |

---

## Experiment Design

A full factorial experimental design evaluates all combinations of noise type, SNR level, and SVD method:

```
6 noise types x 4 SNR levels x 4 SVD methods = 96 conditions
```

### Parameters

| Parameter | Values |
|-----------|--------|
| Noise Types | Baseline wander, Powerline, EMG, Motion artifact, Measurement error, White noise |
| SNR Levels | -5 dB, 0 dB, 5 dB, 10 dB |
| SVD Methods | Standard SVD, Hankel SVD (SSA), Recursive SVD (Brand 2006), Randomized SVD |
| Sampling Rate | 250 Hz |
| Signal Length | 2000 samples (8 seconds) |
| Random Seed | 42 |

### Pipeline

1. Load the clean ECG signal from wave.csv
2. For each noise type, generate the noise pattern
3. For each SNR level, scale the noise and add it to the clean signal
4. For each SVD method, apply denoising with rank = 5
5. Compute evaluation metrics against the clean reference
6. Save numerical results to results/metrics.csv
7. Generate comparison plots and diagnostic visualizations

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **SNR Improvement** (dB) | SNR_after - SNR_before | Positive values mean successful noise reduction |
| **RMSE** | sqrt(mean((clean - denoised)^2)) | Lower is better; penalizes large deviations |
| **Correlation Coefficient** | Pearson r between clean and denoised | Closer to 1.0 means better morphological preservation |
| **PRD** (%) | 100 x sqrt(sum((d - c)^2) / sum(c^2)) | Percentage root-mean-square difference; lower is better |

---

## Project Structure

```
SVD-ECG-Denoising/
+-- main.py                         # Full factorial experiment pipeline (6x4x4 = 96 conditions)
+-- svd_ecg_denoise.py              # 12-method comparison (6 traditional + 6 new)
+-- add_all_noise.py                # Add all noise types to clean signal
+-- add_50hz_noise.py               # Add 50 Hz powerline interference
+-- combined_noise_experiment.py    # Combined noise experiment runner
+-- combined_noise_with_50hz.py     # Combined noise with 50 Hz interference
+-- plot_noisy_50hz.py              # Visualization of 50 Hz noisy signals
+-- powerline_interference_50hz.py  # Powerline interference analysis
+-- test_combined_noise.py          # Tests for combined noise experiments
+-- test_new_svd.py                 # Tests for new SVD methods (7-12)
+-- requirements.txt                # Python dependencies
+-- kai4.csv                        # Example ECG dataset (Kai4)
+-- LICENSE                         # MIT License
|
+-- src/                            # Source modules
|   +-- __init__.py
|   +-- data_loader.py              # ECG signal loading utilities
|   +-- metrics.py                  # Evaluation metrics (SNR, MSE, correlation)
|   +-- noise.py                    # Noise generation (6 types)
|   +-- plotting.py                 # Visualization and comparison plots
|   +-- svd_methods.py              # Core SVD denoising implementations (4 methods)
|
+-- results/                        # Output directory for results and figures
|   +-- 1/ ... 7/                   # Per-method detailed results
|   +-- metrics.csv                 # Numerical results table (96 rows)
|   +-- metrics_summary.png         # Aggregate metrics visualization
|   +-- singular_values.png         # Singular value spectrum plot
|   +-- *_comparison.png            # Per-noise-type comparison plots
|   +-- kai4_exploration/           # Kai4 dataset exploration outputs
|   +-- realtime_denoise/           # Realtime denoising demonstrations
|
+-- draft/                          # Experimental and draft code
```

---

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- **numpy** - Numerical computing and array operations
- **scipy** - Signal processing filters and linear algebra
- **pandas** - Data handling and CSV output
- **scikit-learn** - Randomized SVD implementation
- **matplotlib** - Visualization and plotting
- **kneed** (optional) - Automatic knee/elbow detection for optimal rank estimation

---

## Usage

### Run the full factorial experiment (4 methods)

```bash
python main.py
```

This runs 96 conditions and saves results to results/metrics.csv, plus generates comparison plots for each noise type.

### Run the 12-method comparison

```bash
python svd_ecg_denoise.py
```

This runs all 12 methods on a pre-mixed noisy signal (wave_with_all_noise.csv) and prints per-method metrics including SNR improvement, RMSE, correlation, and PRD.

### Explore individual noise types

```bash
python add_all_noise.py
python add_50hz_noise.py
python powerline_interference_50hz.py
python plot_noisy_50hz.py
```

---

## Results

Results are stored in the results/ directory:

- **metrics.csv** - Full numerical results table with all 96 conditions
- **metrics_summary.png** - Aggregate summary visualization across all conditions
- **singular_values.png** - Singular value spectrum of the clean ECG signal
- ***_comparison.png** - Per-noise-type visual comparisons of all methods
- **1/ through 7/** - Per-method detailed output directories

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this project in your research, please consider citing:

```bibtex
@software{svd_ecg_denoising_2025,
  title = {SVD-ECG-Denoising: A Systematic Comparison of 12 SVD-Based ECG Denoising Methods},
  author = {SVD-ECG-Denoising Contributors},
  year = {2025},
  url = {https://github.com/your-username/SVD-ECG-Denoising}
}
```
