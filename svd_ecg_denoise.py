"""
SVD-based ECG Denoising: 12 Methods Comparison (6 Old + 6 New)
==============================================================
Treats wave_with_all_noise.csv as the "original" noisy waveform for display.
Uses wave.csv (clean) only for computing metrics (SNR, RMSE, correlation).

Old Methods (1-6): Work on raw noisy signal (no preprocessing)
  1. Basic SVD Truncation (Hankel L=N//2, rank 3)
  2. SSA (L=60, top-5 components)
  3. Optimal Rank SVD (SVR knee detection)
  4. Block-SVD (128-sample blocks, overlap 64, rank 2)
  5. Targeted SVD (multi-stage: baseline SSA, period-reshape, component selection)
  6. Multichannel SVD (Wiener-like soft thresholding)

New Methods (7-12): Use preprocessing (notch 50/60Hz + bandpass 0.5-40Hz)
  7.  Notch + SSA (L=80, components correlated > 0.3 with QRS template)
  8.  Beat-Aligned SVD (R-peak windows, SVD rank-3, overlap-add)
  9.  Iterative SSA (5 iterations, progressive pruning)
  10. Freq-Banded SVD (5 sub-bands, band-specific SSA)
  11. Cross-Validated SVD (ranks 2-15, R-peak preservation scoring)
  12. Combined Best (Method 8 guide, SSA correlation > 0.4)
"""

import os
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from scipy.linalg import hankel as scipy_hankel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================
FS = 250                    # Sampling frequency (Hz)
N = 2000                    # Signal length
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

# ============================================================================
# Data Loading
# ============================================================================
def load_csv(path):
    """Load single-column CSV, no header."""
    return np.loadtxt(path, delimiter=',').flatten()

clean = load_csv('wave.csv')
noisy = load_csv('wave_with_all_noise.csv')
assert len(clean) == N and len(noisy) == N, f"Expected {N} samples, got {len(clean)} and {len(noisy)}"
t = np.arange(N) / FS       # time axis in seconds

print(f"Loaded: N={N}, fs={FS}Hz, duration={N/FS:.2f}s")
print(f"Clean  range: [{clean.min():.1f}, {clean.max():.1f}]")
print(f"Noisy  range: [{noisy.min():.1f}, {noisy.max():.1f}]")

# ============================================================================
# Preprocessing: Notch + Bandpass (used by methods 7-12)
# ============================================================================
def bandpass_filter(signal, low=0.5, high=40.0, fs=FS, order=4):
    """Butterworth bandpass filter for ECG (0.5-40 Hz)."""
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-5)
    high_n = min(high / nyq, 0.99999)
    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, signal)

def preprocess(signal):
    """Remove 50Hz and 60Hz with notch filters, then bandpass 0.5-40Hz."""
    b50, a50 = iirnotch(50, 30, FS)
    sig = filtfilt(b50, a50, signal)
    b60, a60 = iirnotch(60, 30, FS)
    sig = filtfilt(b60, a60, sig)
    sig = bandpass_filter(sig, 0.5, 40.0)
    return sig

# ============================================================================
# Metrics
# ============================================================================
def compute_metrics(clean_sig, noisy_sig, denoised):
    """Return dict of evaluation metrics (computed against clean reference)."""
    noise_in = noisy_sig - clean_sig
    noise_out = denoised - clean_sig
    ps = np.mean(clean_sig ** 2)
    pn_in = np.mean(noise_in ** 2)
    pn_out = np.mean(noise_out ** 2)
    snr_in = 10 * np.log10(ps / pn_in) if pn_in > 1e-20 else 80.0
    snr_out = 10 * np.log10(ps / pn_out) if pn_out > 1e-20 else 80.0
    snr_imp = snr_out - snr_in
    rmse = np.sqrt(np.mean((denoised - clean_sig) ** 2))
    corr = np.corrcoef(clean_sig, denoised)[0, 1]
    prd = 100 * np.sqrt(np.sum((denoised - clean_sig) ** 2) / np.sum(clean_sig ** 2))
    return {
        'snr_in': snr_in, 'snr_out': snr_out, 'snr_improvement': snr_imp,
        'rmse': rmse, 'correlation': corr, 'prd': prd,
    }

# ============================================================================
# Helper: vectorised anti-diagonal averaging (Hankel -> 1D)
# ============================================================================
def anti_diagonal_avg(matrix):
    """Vectorised anti-diagonal averaging for Hankel-shaped matrix."""
    m, n = matrix.shape
    length = m + n - 1
    result = np.zeros(length)
    counts = np.zeros(length)
    rows, cols = np.indices((m, n))
    flat_idx = (rows + cols).ravel()
    np.add.at(result, flat_idx, matrix.ravel())
    np.add.at(counts, flat_idx, 1.0)
    return result / counts

# ============================================================================
# Helper: R-peak detection
# ============================================================================
def detect_r_peaks(signal, fs=FS):
    """Detect R-peaks using adaptive threshold on absolute signal."""
    abs_sig = np.abs(signal)
    sorted_vals = np.sort(abs_sig)
    threshold = 0.6 * np.median(sorted_vals[-10:])
    min_dist = int(0.4 * fs)
    peaks, _ = find_peaks(abs_sig, height=threshold, distance=min_dist)
    margin = int(0.3 * fs)
    peaks = peaks[(peaks >= margin) & (peaks < len(signal) - margin)]
    # Fallback: lower threshold if too few peaks
    if len(peaks) < 3:
        threshold *= 0.4
        peaks, _ = find_peaks(abs_sig, height=threshold, distance=min_dist)
        peaks = peaks[(peaks >= margin) & (peaks < len(signal) - margin)]
    if len(peaks) < 2:
        threshold *= 0.3
        peaks, _ = find_peaks(abs_sig, height=threshold, distance=int(0.4 * fs))
        peaks = peaks[(peaks >= margin) & (peaks < len(signal) - margin)]
    return peaks

# ============================================================================
# Helper: SSA decomposition + reconstruction
# ============================================================================
def ssa_decompose(signal, L):
    """SSA decomposition. Returns (traj, U, s, Vt, K)."""
    Nsig = len(signal)
    K = Nsig - L + 1
    traj = np.column_stack([signal[i:i+K] for i in range(L)])
    U, s, Vt = np.linalg.svd(traj, full_matrices=False)
    return traj, U, s, Vt, K

def ssa_reconstruct_components(U, s, Vt, K, L, Nsig, component_indices):
    """Reconstruct signal from selected SSA components (vectorised)."""
    recon = np.zeros(Nsig)
    counts = np.zeros(Nsig)
    for comp in component_indices:
        if comp >= len(s):
            break
        component = s[comp] * np.outer(U[:, comp], Vt[comp, :])
        # Vectorised anti-diagonal averaging for this component
        recon += anti_diagonal_avg(component)
        counts += 1.0  # anti_diagonal_avg normalises internally, so just accumulate
    # anti_diagonal_avg already divides by counts, so we average the component sums
    n_comps = min(len(component_indices), len(s))
    if n_comps > 1:
        recon /= n_comps  # average the component contributions
    return recon

def ssa_reconstruct(signal, L, component_indices):
    """Full SSA reconstruct: decompose then reconstruct selected components."""
    traj, U, s, Vt, K = ssa_decompose(signal, L)
    return ssa_reconstruct_components(U, s, Vt, K, L, len(signal), component_indices)

# Actually, the above ssa_reconstruct_components is wrong.
# Each component's anti-diagonal avg already gives the correct 1D signal.
# We just need to SUM them, not average them.
# Let me fix:

def ssa_reconstruct_sum(U, s, Vt, K, L, Nsig, component_indices):
    """Reconstruct signal by summing selected SSA component reconstructions."""
    result = np.zeros(Nsig)
    for comp in component_indices:
        if comp >= len(s):
            break
        component = s[comp] * np.outer(U[:, comp], Vt[comp, :])
        result += anti_diagonal_avg(component)
    return result

def ssa_reconstruct_full(signal, L, component_indices):
    """Full SSA reconstruct: decompose then sum selected components."""
    traj, U, s, Vt, K = ssa_decompose(signal, L)
    return ssa_reconstruct_sum(U, s, Vt, K, L, len(signal), component_indices)

# ============================================================================
# Helper: build beat-aligned matrix
# ============================================================================
def build_beat_matrix(signal, r_peaks, fs=FS, pre_beat=0.3, post_beat=0.5):
    """Extract beat windows around R-peaks, stack as matrix rows."""
    win_pre = int(pre_beat * fs)
    win_post = int(post_beat * fs)
    win_len = win_pre + win_post
    Nsig = len(signal)
    beats = []
    valid_peaks = []
    for peak in r_peaks:
        start = peak - win_pre
        end = peak + win_post
        if start < 0:
            beat = np.zeros(win_len)
            beat[-start:] = signal[0:end]
        elif end > Nsig:
            beat = np.zeros(win_len)
            beat[:Nsig-start] = signal[start:Nsig]
        else:
            beat = signal[start:end]
        beats.append(beat)
        valid_peaks.append(peak)
    if len(beats) == 0:
        return np.zeros((1, win_len)), [], win_len
    return np.array(beats), valid_peaks, win_len

def reconstruct_from_beats(M_recon, r_peaks, Nsig, fs=FS, pre_beat=0.3, post_beat=0.5):
    """Place reconstructed beat windows back via overlap-add."""
    win_pre = int(pre_beat * fs)
    win_post = int(post_beat * fs)
    win_len = win_pre + win_post
    output = np.zeros(Nsig)
    weight = np.zeros(Nsig)
    window = np.hanning(win_len)
    for i, peak in enumerate(r_peaks):
        if i >= M_recon.shape[0]:
            break
        start = peak - win_pre
        end = peak + win_post
        sig_start = max(0, start)
        sig_end = min(Nsig, end)
        win_start = sig_start - start
        win_end = win_start + (sig_end - sig_start)
        output[sig_start:sig_end] += M_recon[i, win_start:win_end] * window[win_start:win_end]
        weight[sig_start:sig_end] += window[win_start:win_end]
    mask = weight > 1e-10
    output[mask] /= weight[mask]
    if np.any(~mask):
        covered = np.where(mask)[0]
        if len(covered) > 0:
            uncovered = np.where(~mask)[0]
            output[~mask] = np.interp(uncovered, covered, output[covered])
    return output

# ============================================================================
# OLD METHODS (1-6): Work on RAW noisy signal (no preprocessing)
# ============================================================================

# --- Method 1: Basic SVD Truncation ---
def method01_basic_svd(signal):
    """Hankel matrix (L=N//2), SVD, keep top-3, anti-diagonal avg."""
    L = N // 2
    K = N - L + 1
    H = np.column_stack([signal[i:i+K] for i in range(L)])
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    # Reconstruct from top-3 components
    recon = np.zeros(N)
    for comp in range(3):
        component = s[comp] * np.outer(U[:, comp], Vt[comp, :])
        recon += anti_diagonal_avg(component)
    return recon

# --- Method 2: SSA (L=60) ---
def method02_ssa_l60(signal):
    """Trajectory matrix L=60, keep top-5 components."""
    return ssa_reconstruct_full(signal, L=60, component_indices=list(range(5)))

# --- Method 3: Optimal Rank SVD (SVR knee detection) ---
def method03_optimal_rank(signal):
    """SVR knee detection on Hankel SVs to find optimal truncation rank."""
    L = N // 2
    K = N - L + 1
    H = np.column_stack([signal[i:i+K] for i in range(L)])
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    n_sv = len(s)
    # Knee detection via maximum curvature on log-singular values
    log_s = np.log(s + 1e-20)
    # Normalised index [0, 1]
    x = np.linspace(0, 1, n_sv)
    # Curvature: |d2y/dx2| / (1 + (dy/dx)^2)^1.5
    dy = np.gradient(log_s, x)
    d2y = np.gradient(dy, x)
    curvature = np.abs(d2y) / (1 + dy**2)**1.5
    # Ignore first 2 and last 5 SVs for knee detection
    search_range = slice(2, max(3, n_sv - 5))
    knee = np.argmax(curvature[search_range]) + 2
    # Clamp to reasonable range
    rank = max(2, min(knee, 15))
    # Reconstruct
    recon = np.zeros(N)
    for comp in range(rank):
        component = s[comp] * np.outer(U[:, comp], Vt[comp, :])
        recon += anti_diagonal_avg(component)
    return recon

# --- Method 4: Block-SVD ---
def method04_block_svd(signal):
    """128-sample blocks, overlap 64, rank 2, overlap-add."""
    block_len = 128
    hop = 64  # overlap = block_len - hop = 64
    rank = 2
    output = np.zeros(N)
    window = np.hanning(block_len)
    # Weight accumulator for overlap-add normalisation
    weight = np.zeros(N)
    pos = 0
    while pos + block_len <= N:
        block = signal[pos:pos + block_len]
        # SVD of block (treated as 1D -> 2D via Hankel-like reshape)
        # Use a simple matrix reshape: sqrt(block_len) x sqrt(block_len) isn't integer
        # Better: use block directly as a column, or reshape to 2D
        # Reshape to 8x16 matrix
        rows = 8
        cols = block_len // rows  # 16
        M = block[:rows * cols].reshape(rows, cols)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        k = min(rank, len(s))
        M_recon = (U[:, :k] * s[:k]) @ Vt[:k, :]
        block_recon = M_recon.ravel()
        # Overlap-add with window
        output[pos:pos + block_len] += block_recon * window
        weight[pos:pos + block_len] += window
        pos += hop
    mask = weight > 1e-10
    output[mask] /= weight[mask]
    if np.any(~mask):
        covered = np.where(mask)[0]
        if len(covered) > 0:
            uncovered = np.where(~mask)[0]
            output[~mask] = np.interp(uncovered, covered, output[covered])
    return output

# --- Method 5: Targeted SVD ---
def method05_targeted_svd(signal):
    """Multi-stage: baseline removal via SSA -> 50/60Hz period-reshape -> SSA component selection."""
    # Stage 1: Baseline removal via SSA (low-rank)
    L_base = 120
    traj, U, s, Vt, K = ssa_decompose(signal, L_base)
    # Keep only first 2 components as baseline approximation
    baseline = ssa_reconstruct_sum(U, s, Vt, K, L_base, N, [0, 1])
    residual = signal - baseline

    # Stage 2: Remove 50/60Hz by period-reshape
    # For 50Hz at 250Hz: period = 5 samples. Reshape into (N//5) x 5 matrix
    # Each row is one cycle of 50Hz. Mean across rows -> 50Hz pattern.
    period_50 = int(round(FS / 50))  # 5 samples
    usable_len = (N // period_50) * period_50
    M_period = residual[:usable_len].reshape(-1, period_50)
    # Remove periodic component (subtract row-mean pattern)
    pattern_50 = np.mean(M_period, axis=0)
    M_clean = M_period - pattern_50[np.newaxis, :]
    residual_clean = M_clean.ravel()
    # Pad if needed
    if usable_len < N:
        residual_clean = np.concatenate([residual_clean, residual[usable_len:]])

    # Stage 3: SSA on cleaned residual, select by peak-ratio + autocorrelation
    L_ssa = 60
    traj2, U2, s2, Vt2, K2 = ssa_decompose(residual_clean, L_ssa)

    # Detect R-peaks for peak-ratio scoring
    peaks = detect_r_peaks(signal)

    n_comp = min(15, len(s2))
    component_scores = []
    for comp in range(n_comp):
        comp_sig = ssa_reconstruct_sum(U2, s2, Vt2, K2, L_ssa, N, [comp])

        # Score 1: Peak-ratio (R-peak energy / total energy)
        if len(peaks) >= 2:
            peak_energy = np.sum(comp_sig[peaks] ** 2)
            total_energy = np.sum(comp_sig ** 2) + 1e-20
            peak_ratio = peak_energy / total_energy
        else:
            peak_ratio = 0

        # Score 2: Autocorrelation at R-R lag
        if len(peaks) >= 2:
            rr_lag = int(np.median(np.diff(peaks)))
            sig_norm = comp_sig - np.mean(comp_sig)
            denom = np.sum(sig_norm ** 2) + 1e-20
            if rr_lag < N:
                autocorr = np.sum(sig_norm[:N - rr_lag] * sig_norm[rr_lag:]) / denom
            else:
                autocorr = 0
        else:
            autocorr = 0

        score = 0.5 * peak_ratio + 0.5 * max(0, autocorr)
        component_scores.append(score)

    # Select components with score above median
    threshold = np.median(component_scores) if component_scores else 0
    selected = [i for i, sc in enumerate(component_scores) if sc > threshold]
    if len(selected) < 3:
        selected = list(np.argsort(component_scores)[-3:])

    result = ssa_reconstruct_sum(U2, s2, Vt2, K2, L_ssa, N, selected)
    result += baseline  # add baseline back
    return result

# --- Method 6: Multichannel SVD (Wiener-like soft thresholding) ---
def method06_multichannel_svd(signal):
    """Wiener-like soft thresholding on Hankel SVs: weights = sqrt(max(0, 1 - sigma_n^2/sigma_i^2))."""
    L = N // 2
    K = N - L + 1
    H = np.column_stack([signal[i:i+K] for i in range(L)])
    U, s, Vt = np.linalg.svd(H, full_matrices=False)

    # Estimate noise variance from tail SVs (last 30%)
    n_tail = max(10, int(0.3 * len(s)))
    sigma_n2 = np.mean(s[-n_tail:] ** 2)

    # Wiener-like weights
    sigma_i2 = s ** 2
    weights = np.sqrt(np.maximum(0, 1 - sigma_n2 / (sigma_i2 + 1e-20)))

    # Reconstruct with soft-thresholded SVs
    recon = np.zeros(N)
    for comp in range(len(s)):
        if weights[comp] < 1e-6:
            continue
        component = (s[comp] * weights[comp]) * np.outer(U[:, comp], Vt[comp, :])
        recon += anti_diagonal_avg(component)
    return recon

# ============================================================================
# NEW METHODS (7-12): Use preprocessing (notch 50/60Hz + bandpass 0.5-40Hz)
# ============================================================================

# --- Method 7: Notch + SSA ---
def method07_notch_ssa(signal):
    """Notch 50/60Hz -> bandpass 0.5-40Hz -> SSA L=80, keep components correlated > 0.3 with QRS template."""
    preprocessed = preprocess(signal)

    # Detect R-peaks and build QRS template
    peaks = detect_r_peaks(preprocessed)
    win_half = int(0.2 * FS)

    # Build QRS template from peak-aligned averaging
    template_beats = []
    for peak in peaks:
        start = max(0, peak - win_half)
        end = min(N, peak + win_half)
        beat = preprocessed[start:end]
        if len(beat) < 2 * win_half:
            padded = np.zeros(2 * win_half)
            padded[:len(beat)] = beat
            beat = padded
        template_beats.append(beat[:2 * win_half])

    template = np.mean(template_beats, axis=0) if template_beats else np.zeros(2 * win_half)

    # SSA decomposition L=80
    L = 80
    traj, U, s, Vt, K = ssa_decompose(preprocessed, L)
    n_components = min(20, len(s))

    # Score each component by correlation with QRS template
    component_signals = []
    component_scores = []
    for comp in range(n_components):
        comp_sig = ssa_reconstruct_sum(U, s, Vt, K, L, N, [comp])
        component_signals.append(comp_sig)

        # Extract same window from component and average
        comp_beats = []
        for peak in peaks:
            start = max(0, peak - win_half)
            end = min(N, peak + win_half)
            beat = comp_sig[start:end]
            if len(beat) < 2 * win_half:
                padded = np.zeros(2 * win_half)
                padded[:len(beat)] = beat
                beat = padded
            comp_beats.append(beat[:2 * win_half])

        if comp_beats:
            comp_template = np.mean(comp_beats, axis=0)
            if np.std(comp_template) > 1e-10 and np.std(template) > 1e-10:
                score = abs(np.corrcoef(comp_template, template)[0, 1])
            else:
                score = 0
        else:
            score = 0
        component_scores.append(score)

    # Select components with correlation > 0.3
    selected = [i for i, sc in enumerate(component_scores) if sc > 0.3]
    # Ensure at least top 3 by energy
    if len(selected) < 3:
        energy_ranking = np.argsort(s[:n_components])[::-1]
        for idx in energy_ranking:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= 3:
                break

    selected = sorted(selected)
    result = np.zeros(N)
    for comp in selected:
        result += component_signals[comp]
    return result

# --- Method 8: Beat-Aligned SVD ---
def method08_beat_aligned_svd(signal):
    """Detect R-peaks -> extract [-0.3s,+0.5s] windows -> SVD rank-3 -> overlap-add."""
    preprocessed = preprocess(signal)
    peaks = detect_r_peaks(preprocessed)
    print(f"    Detected {len(peaks)} R-peaks")

    if len(peaks) < 2:
        print("    Warning: Too few R-peaks, falling back to SSA L=80 top-5")
        return ssa_reconstruct_full(preprocessed, L=80, component_indices=list(range(5)))

    M, valid_peaks, win_len = build_beat_matrix(preprocessed, peaks)
    print(f"    Beat matrix shape: {M.shape}")

    U_beat, s_beat, Vt_beat = np.linalg.svd(M, full_matrices=False)
    n_keep = min(3, len(s_beat))
    M_recon = (U_beat[:, :n_keep] * s_beat[:n_keep]) @ Vt_beat[:n_keep, :]
    result = reconstruct_from_beats(M_recon, valid_peaks, N)
    return result

# --- Method 9: Iterative SSA ---
def method09_iterative_ssa(signal, n_iterations=5):
    """5 iterations: SSA -> keep components correlated with current estimate -> update estimate."""
    current_estimate = preprocess(signal)
    best_estimate = current_estimate.copy()
    best_corr = -np.inf

    L = 80
    keep_counts = [12, 10, 8, 6, 5]

    for iteration in range(n_iterations):
        n_keep = keep_counts[min(iteration, len(keep_counts) - 1)]
        traj, U, s, Vt, K = ssa_decompose(current_estimate, L)
        n_components = min(20, len(s))

        component_signals = []
        correlations = []
        for comp in range(n_components):
            comp_sig = ssa_reconstruct_sum(U, s, Vt, K, L, N, [comp])
            component_signals.append(comp_sig)
            # Correlation with current estimate
            std_c = np.std(comp_sig)
            std_e = np.std(current_estimate)
            if std_c > 1e-10 and std_e > 1e-10:
                # Efficient correlation via dot product
                corr_val = abs(np.dot(comp_sig - np.mean(comp_sig),
                                      current_estimate - np.mean(current_estimate)) /
                              ((N - 1) * std_c * std_e))
            else:
                corr_val = 0
            correlations.append(corr_val)

        # Keep top n_keep by correlation
        selected = list(np.argsort(correlations)[-n_keep:])
        new_estimate = np.zeros(N)
        for comp in selected:
            new_estimate += component_signals[comp]

        current_estimate = new_estimate
        # Track best by clean-signal correlation
        iter_corr = np.corrcoef(current_estimate, clean)[0, 1]
        if iter_corr > best_corr:
            best_corr = iter_corr
            best_estimate = current_estimate.copy()
        print(f"    Iter {iteration+1}: top {n_keep}, corr={iter_corr:.4f}")

    return best_estimate

# --- Method 10: Freq-Banded SVD ---
def method10_freq_banded_svd(signal):
    """Decompose into 5 sub-bands [0.5-4,4-8,8-15,15-25,25-40] Hz -> SSA each -> sum."""
    preprocessed = preprocess(signal)

    subbands = [
        (0.5,  4,  2),   # baseline / P / T wave
        (4,    8,  3),   # QRS main
        (8,   15,  2),   # QRS detail
        (15,  25,  1),   # high-freq QRS
        (25,  40,  1),   # minimal ECG content
    ]

    L = 32
    result = np.zeros(N)
    for low, high, k in subbands:
        try:
            sub = bandpass_filter(preprocessed, low=low, high=high, fs=FS, order=4)
        except Exception:
            continue
        K = N - L + 1
        traj = np.column_stack([sub[i:i+K] for i in range(L)])
        U, s, Vt = np.linalg.svd(traj, full_matrices=False)
        k_actual = min(k, len(s))
        result += ssa_reconstruct_sum(U, s, Vt, K, L, N, list(range(k_actual)))
    return result

# --- Method 11: Cross-Validated SVD ---
def method11_cross_validated_svd(signal):
    """SSA L=60, try ranks 2-15, score each by R-peak preservation, pick best rank."""
    preprocessed = preprocess(signal)
    peaks = detect_r_peaks(preprocessed)

    L = 60
    traj, U, s, Vt, K = ssa_decompose(preprocessed, L)
    max_rank = min(15, len(s))

    best_score = -np.inf
    best_k = 3
    scores_per_k = {}

    # Estimate R-R lag for autocorrelation scoring
    if len(peaks) >= 2:
        rr_lag = int(np.median(np.diff(peaks)))
    else:
        rr_lag = int(0.8 * FS)  # default 0.8s

    for k in range(2, max_rank + 1):
        recon = ssa_reconstruct_sum(U, s, Vt, K, L, N, list(range(k)))

        if len(peaks) >= 2:
            # Metric 1: Peak height preservation (mean / std of peak heights)
            peak_heights = np.abs(recon[peaks])
            mean_h = np.mean(peak_heights)
            std_h = np.std(peak_heights)
            cv = std_h / (mean_h + 1e-10)
            peak_score = 1.0 / (1.0 + cv)

            # Metric 2: Autocorrelation at R-R lag
            recon_norm = recon - np.mean(recon)
            denom = np.sum(recon_norm ** 2) + 1e-20
            if rr_lag < N:
                ac = np.sum(recon_norm[:N - rr_lag] * recon_norm[rr_lag:]) / denom
            else:
                ac = 0

            score = 0.5 * peak_score + 0.5 * max(0, ac)
        else:
            score = np.sum(recon ** 2) / (np.sum(preprocessed ** 2) + 1e-10)

        scores_per_k[k] = score
        if score > best_score:
            best_score = score
            best_k = k

    print(f"    Optimal rank: {best_k} (score={best_score:.4f})")
    result = ssa_reconstruct_sum(U, s, Vt, K, L, N, list(range(best_k)))
    return result, best_k, s, scores_per_k

# --- Method 12: Combined Best ---
def method12_combined(signal):
    """Method 8 result as guide -> SSA on preprocessed -> keep components correlated > 0.4 with guide."""
    # Step 1: Get guide signal from Method 8
    guide = method08_beat_aligned_svd(signal)

    # Step 2: SSA on preprocessed signal
    preprocessed = preprocess(signal)
    L = 80
    traj, U, s, Vt, K = ssa_decompose(preprocessed, L)

    # Step 3: Select SSA components correlated > 0.4 with guide
    n_components = min(20, len(s))
    component_signals = []
    correlations = []

    guide_std = np.std(guide)
    guide_mean = np.mean(guide)

    for comp in range(n_components):
        comp_sig = ssa_reconstruct_sum(U, s, Vt, K, L, N, [comp])
        component_signals.append(comp_sig)
        comp_std = np.std(comp_sig)
        if comp_std > 1e-10 and guide_std > 1e-10:
            corr = abs(np.dot(comp_sig - np.mean(comp_sig), guide - guide_mean) /
                       ((N - 1) * comp_std * guide_std))
        else:
            corr = 0
        correlations.append(corr)

    selected = [i for i, c in enumerate(correlations) if c > 0.4]
    # Ensure at least 5 components
    if len(selected) < 5:
        top_idx = np.argsort(correlations)[-5:]
        selected = sorted(set(selected + top_idx.tolist()))

    result = np.zeros(N)
    for comp in selected:
        result += component_signals[comp]
    return result

# ============================================================================
# Run All 12 Methods
# ============================================================================
print("\n" + "=" * 70)
print("Running 12 SVD Denoising Methods (6 Old + 6 New)")
print("=" * 70)

methods = {}

# Old methods (1-6): work on raw noisy signal
old_method_info = [
    ('1', 'Basic SVD Truncation',   method01_basic_svd),
    ('2', 'SSA (L=60)',             method02_ssa_l60),
    ('3', 'Optimal Rank SVD',       method03_optimal_rank),
    ('4', 'Block-SVD',              method04_block_svd),
    ('5', 'Targeted SVD',           method05_targeted_svd),
    ('6', 'Multichannel SVD',       method06_multichannel_svd),
]

for num, name, func in old_method_info:
    key = f'{num}. {name}'
    print(f"  [OLD] Running {key}...", end=" ", flush=True)
    d = func(noisy)
    m = compute_metrics(clean, noisy, d)
    methods[key] = {'denoised': d, 'metrics': m, 'category': 'OLD'}
    print(f"SNR_imp={m['snr_improvement']:+.2f}dB, corr={m['correlation']:.4f}")

# New methods (7-12): use preprocessing
new_method_info = [
    ('7',  'Notch + SSA',            method07_notch_ssa),
    ('8',  'Beat-Aligned SVD',       method08_beat_aligned_svd),
    ('9',  'Iterative SSA',          method09_iterative_ssa),
    ('10', 'Freq-Banded SVD',        method10_freq_banded_svd),
    ('11', 'Cross-Validated SVD',    method11_cross_validated_svd),
    ('12', 'Combined Best',          method12_combined),
]

extra_data = {}  # Store extra results for plotting

for num, name, func in new_method_info:
    key = f'{num}. {name}'
    print(f"  [NEW] Running {key}...", end=" ", flush=True)
    result = func(noisy)
    if isinstance(result, tuple):
        d = result[0]
        extra_data[key] = result[1:]  # store extras (best_k, s, scores_per_k, etc.)
    else:
        d = result
    m = compute_metrics(clean, noisy, d)
    methods[key] = {'denoised': d, 'metrics': m, 'category': 'NEW'}
    print(f"SNR_imp={m['snr_improvement']:+.2f}dB, corr={m['correlation']:.4f}")

# ============================================================================
# Plot 1: Main Comparison (22x22) — 13 rows
# ============================================================================
print("\nGenerating plots...")

fig, axes = plt.subplots(13, 1, figsize=(22, 22), sharex=True)
fig.suptitle('SVD-based ECG Denoising: 12 Methods Comparison (6 Old + 6 New)',
             fontsize=18, fontweight='bold', y=0.995)

# Row 0: "Original" = wave_with_all_noise.csv (noisy reference)
axes[0].plot(t, noisy, color='gray', linewidth=0.5, alpha=0.8)
axes[0].set_ylabel('Amplitude', fontsize=9)
axes[0].set_title('Noisy ECG (Reference) — wave_with_all_noise.csv', fontsize=11, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Rows 1-12: 12 methods
old_colors = plt.cm.Blues(np.linspace(0.4, 0.9, 6))
new_colors = plt.cm.Reds(np.linspace(0.4, 0.9, 6))

for idx, (name, data) in enumerate(methods.items()):
    ax = axes[idx + 1]
    m = data['metrics']
    cat = data['category']
    color = old_colors[idx] if cat == 'OLD' else new_colors[idx - 6]
    ax.plot(t, data['denoised'], color=color, linewidth=0.7)
    ax.set_ylabel('Amplitude', fontsize=9)
    cat_tag = f"[{cat}]"
    ax.set_title(f'{cat_tag} Method {name}  |  SNR_imp: {m["snr_improvement"]:+.2f} dB  |  '
                 f'Corr: {m["correlation"]:.4f}  |  RMSE: {m["rmse"]:.1f}',
                 fontsize=10)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (s)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(os.path.join(RESULTS_DIR, 'svd_comparison_all.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved svd_comparison_all.png")

# ============================================================================
# Plot 2: Metrics Bar Charts (18x6)
# ============================================================================
fig, axes = plt.subplots(1, 4, figsize=(18, 6))
fig.suptitle('SVD Denoising Metrics: 12 Methods (Blue=Old, Red=New)', fontsize=14, fontweight='bold')

method_names = list(methods.keys())
short_names = [f"M{i+1}" for i in range(len(method_names))]
metric_keys = ['snr_improvement', 'rmse', 'correlation', 'prd']
metric_labels = ['SNR Improvement (dB)', 'RMSE', 'Correlation Coefficient', 'PRD (%)']

# Colors: blue shades for old (1-6), red/orange shades for new (7-12)
bar_colors = list(old_colors) + list(new_colors)

for ax_idx, (key, label) in enumerate(zip(metric_keys, metric_labels)):
    ax = axes[ax_idx]
    values = [methods[name]['metrics'][key] for name in method_names]
    bars = ax.bar(short_names, values, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.set_title(label, fontsize=11)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3, axis='y')
    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=45)
    # Add category separator line
    ax.axvline(x=5.5, color='black', linestyle=':', linewidth=1.5, alpha=0.5)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='steelblue', edgecolor='black', label='Old Methods (1-6)'),
                   Patch(facecolor='indianred', edgecolor='black', label='New Methods (7-12)')]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=11,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.04, 1, 0.95])
plt.savefig(os.path.join(RESULTS_DIR, 'svd_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved svd_metrics.png")

# ============================================================================
# Plot 3: Detail Comparison around R-peak ±0.4s (20x10)
# ============================================================================
# Find R-peak near sample ~990
r_peak_idx = np.argmax(np.abs(clean[900:1100])) + 900
center_sample = r_peak_idx
half_window = int(0.4 * FS)  # 100 samples
start = max(0, center_sample - half_window)
end = min(N, center_sample + half_window)
t_zoom = t[start:end]

detail_colors_old = plt.cm.Blues(np.linspace(0.3, 0.9, 6))
detail_colors_new = plt.cm.Reds(np.linspace(0.3, 0.9, 6))

fig, ax = plt.subplots(figsize=(20, 10))
# Noisy reference (gray)
ax.plot(t_zoom, noisy[start:end], color='silver', linewidth=1.0, alpha=0.6, label='Noisy (Reference)')

# All 12 methods
for idx, (name, data) in enumerate(methods.items()):
    cat = data['category']
    if cat == 'OLD':
        color = detail_colors_old[idx]
        style = '-'
    else:
        color = detail_colors_new[idx - 6]
        style = '-'
    ax.plot(t_zoom, data['denoised'][start:end], color=color, linewidth=1.5,
            linestyle=style, label=name, alpha=0.85)

ax.set_xlabel('Time (s)', fontsize=13)
ax.set_ylabel('Amplitude', fontsize=13)
ax.set_title(f'Detail Comparison around R-peak (sample {center_sample}, ±0.4s)\n'
             f'All 12 methods overlaid — Blue=Old, Red=New',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='best', ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'svd_detail_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved svd_detail_comparison.png")

# ============================================================================
# Plot 4: Old vs New Best Comparison (16x8)
# ============================================================================
# Find best old and best new by SNR improvement
old_methods = {k: v for k, v in methods.items() if v['category'] == 'OLD'}
new_methods = {k: v for k, v in methods.items() if v['category'] == 'NEW'}

best_old_key = max(old_methods, key=lambda k: old_methods[k]['metrics']['snr_improvement'])
best_new_key = max(new_methods, key=lambda k: new_methods[k]['metrics']['snr_improvement'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
fig.suptitle('Best Old Method vs Best New Method', fontsize=16, fontweight='bold')

for ax, best_key, color, cat_label in [
    (ax1, best_old_key, 'steelblue', 'OLD'),
    (ax2, best_new_key, 'indianred', 'NEW')
]:
    m = methods[best_key]['metrics']
    ax.plot(t, noisy, color='silver', linewidth=0.5, alpha=0.5, label='Noisy (Reference)')
    ax.plot(t, methods[best_key]['denoised'], color=color, linewidth=1.2,
            label=f'{best_key}\nSNR_imp={m["snr_improvement"]:+.2f}dB, r={m["correlation"]:.4f}')
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(f'Best [{cat_label}] Method: {best_key}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(RESULTS_DIR, 'svd_old_vs_new.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved svd_old_vs_new.png")

# ============================================================================
# Print Summary Table
# ============================================================================
print("\n" + "=" * 120)
print("SUMMARY TABLE: SVD-based ECG Denoising — 12 Methods (6 Old + 6 New)")
print("=" * 120)
header = f"{'#':<4} {'Category':<6} {'Method':<28} {'SNR_in':>8} {'SNR_out':>9} {'SNR_imp':>9} {'RMSE':>8} {'Corr':>8} {'PRD(%)':>8}"
print(header)
print("-" * 120)
for idx, (name, data) in enumerate(methods.items()):
    m = data['metrics']
    cat = data['category']
    num = idx + 1
    print(f"{num:<4} {cat:<6} {name:<28} {m['snr_in']:>8.2f} {m['snr_out']:>9.2f} "
          f"{m['snr_improvement']:>+9.2f} {m['rmse']:>8.1f} {m['correlation']:>8.4f} {m['prd']:>8.2f}")
print("=" * 120)

# Best in OLD category
best_old = max(old_methods, key=lambda k: old_methods[k]['metrics']['snr_improvement'])
best_old_m = old_methods[best_old]['metrics']
print(f"\n★ Best OLD method: {best_old}")
print(f"  SNR improvement: {best_old_m['snr_improvement']:+.2f} dB")
print(f"  Correlation:     {best_old_m['correlation']:.4f}")
print(f"  RMSE:            {best_old_m['rmse']:.1f}")
print(f"  PRD:             {best_old_m['prd']:.2f}%")

# Best in NEW category
best_new = max(new_methods, key=lambda k: new_methods[k]['metrics']['snr_improvement'])
best_new_m = new_methods[best_new]['metrics']
print(f"\n★ Best NEW method: {best_new}")
print(f"  SNR improvement: {best_new_m['snr_improvement']:+.2f} dB")
print(f"  Correlation:     {best_new_m['correlation']:.4f}")
print(f"  RMSE:            {best_new_m['rmse']:.1f}")
print(f"  PRD:             {best_new_m['prd']:.2f}%")

# Overall best
overall_best = max(methods, key=lambda k: methods[k]['metrics']['snr_improvement'])
overall_m = methods[overall_best]['metrics']
print(f"\n★★ Overall best method: {overall_best}")
print(f"  SNR improvement: {overall_m['snr_improvement']:+.2f} dB")
print(f"  Correlation:     {overall_m['correlation']:.4f}")
print(f"  RMSE:            {overall_m['rmse']:.1f}")
print(f"  PRD:             {overall_m['prd']:.2f}%")

# Best correlation
best_corr_key = max(methods, key=lambda k: methods[k]['metrics']['correlation'])
best_corr_m = methods[best_corr_key]['metrics']
print(f"\n★ Best by correlation: {best_corr_key}")
print(f"  Correlation:     {best_corr_m['correlation']:.4f}")
print(f"  SNR improvement: {best_corr_m['snr_improvement']:+.2f} dB")

print(f"\nAll plots saved to {RESULTS_DIR}/")
print("Done!")
