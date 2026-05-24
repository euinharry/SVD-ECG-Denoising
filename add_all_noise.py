import numpy as np
import matplotlib.pyplot as plt

# 1. Load ECG with existing 50Hz noise
ecg_noisy_50hz = np.loadtxt(r'C:\Users\34118\Desktop\SVD\wave_with_50hz.csv')
ecg = np.loadtxt(r'C:\Users\34118\Desktop\SVD\wave.csv')
N = len(ecg_noisy_50hz)
fs = 250
t = np.arange(N) / fs
print(f"Loaded: {N} samples, fs={fs}Hz")
print(f"Original ECG range: [{ecg.min():.0f}, {ecg.max():.0f}]")
print(f"50Hz noisy range:   [{ecg_noisy_50hz.min():.0f}, {ecg_noisy_50hz.max():.0f}]")

# 2. Extract 50Hz noise component (original noisy - clean)
noise_50hz = ecg_noisy_50hz - ecg

# ============================================================
# 3. Generate additional noise components
# ============================================================

# --- EMG (肌电噪音) - high frequency muscle artifact ---
np.random.seed(42)
emg_raw = 3000 * np.random.randn(N)
# Highpass effect: subtract moving average (window=5)
kernel = np.ones(5) / 5
emg_baseline = np.convolve(emg_raw, kernel, mode='same')
emg = emg_raw - emg_baseline
print(f"EMG noise:         mean={emg.mean():.2f}, std={emg.std():.2f}, range [{emg.min():.0f}, {emg.max():.0f}]")

# --- Motion artifact (运动伪影) - low frequency transient bursts ---
np.random.seed(123)
n_bursts = 3
burst_centers = np.random.choice(np.arange(100, N - 100), size=n_bursts, replace=False)
burst_amplitudes = np.random.uniform(5000, 8000, size=n_bursts)
motion = np.zeros(N)
for center, amp in zip(burst_centers, burst_amplitudes):
    # Create local time array for the burst (1 second window)
    half_win = int(0.5 * fs)  # 125 samples each side
    start = max(0, center - half_win)
    end = min(N, center + half_win)
    t_burst = np.arange(end - start) / fs
    # Shift so burst peaks at center
    t_burst_shifted = t_burst - (center - start) / fs
    damped = amp * np.sin(2 * np.pi * 0.5 * t_burst_shifted) * np.exp(-np.abs(t_burst_shifted) / 0.3)
    motion[start:end] += damped
print(f"Motion artifact:   mean={motion.mean():.2f}, std={motion.std():.2f}, range [{motion.min():.0f}, {motion.max():.0f}]")
print(f"  Burst centers at samples: {burst_centers}, amplitudes: {[f'{a:.0f}' for a in burst_amplitudes]}")

# --- Baseline wander (基线漂移) - slow drift ---
wander = 2500 * np.sin(2 * np.pi * 0.1 * t) + \
         1500 * np.sin(2 * np.pi * 0.3 * t + 1.2) + \
         1000 * np.sin(2 * np.pi * 0.05 * t + 0.7)
print(f"Baseline wander:   mean={wander.mean():.2f}, std={wander.std():.2f}, range [{wander.min():.0f}, {wander.max():.0f}]")

# --- Equipment noise (设备噪音) - electronic white noise ---
np.random.seed(99)
equipment = 1200 * np.random.randn(N)
print(f"Equipment noise:   mean={equipment.mean():.2f}, std={equipment.std():.2f}, range [{equipment.min():.0f}, {equipment.max():.0f}]")

# --- Environment noise (环境噪音) - 60Hz powerline ---
np.random.seed(7)
env_phase = np.random.uniform(0, 2 * np.pi)
environment = 800 * np.sin(2 * np.pi * 60 * t + env_phase)
print(f"60Hz environment:  mean={environment.mean():.2f}, std={environment.std():.2f}, range [{environment.min():.0f}, {environment.max():.0f}]")

# ============================================================
# 4. Combine all noise
# ============================================================
other_noise = emg + motion + wander + equipment + environment
ecg_final = ecg_noisy_50hz + other_noise

print(f"\n--- Final Signal ---")
print(f"Other noises combined: mean={other_noise.mean():.2f}, std={other_noise.std():.2f}, range [{other_noise.min():.0f}, {other_noise.max():.0f}]")
print(f"Final noisy ECG:       mean={ecg_final.mean():.2f}, std={ecg_final.std():.2f}, range [{ecg_final.min():.0f}, {ecg_final.max():.0f}]")

# ============================================================
# 5. Save
# ============================================================
np.savetxt(r'C:\Users\34118\Desktop\SVD\wave_with_all_noise.csv', ecg_final, fmt='%.7f')
print(f"\nSaved: wave_with_all_noise.csv")

# ============================================================
# 6. Plot (5 subplots)
# ============================================================
fig, axes = plt.subplots(5, 1, figsize=(16, 14), dpi=150, sharex=True)

axes[0].plot(t, ecg, 'b', linewidth=0.5)
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Original ECG Signal')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, noise_50hz, 'r', linewidth=0.5)
axes[1].set_ylabel('Amplitude')
axes[1].set_title('50Hz Power Line Noise')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, other_noise, color='orange', linewidth=0.5)
axes[2].set_ylabel('Amplitude')
axes[2].set_title('Other Noises Combined (EMG + Motion + Wander + Equipment + 60Hz)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(t, ecg_final, 'g', linewidth=0.5)
axes[3].set_ylabel('Amplitude')
axes[3].set_xlabel('Time (s)')
axes[3].set_title('Final Noisy ECG (all noise types)')
axes[3].grid(True, alpha=0.3)

# Zoomed view: 0.5s window around sample 500
zoom_center = 500
zoom_half = int(0.25 * fs)  # 0.25s each side = 0.5s total
zoom_start = max(0, zoom_center - zoom_half)
zoom_end = min(N, zoom_center + zoom_half)
axes[4].plot(t[zoom_start:zoom_end], ecg_final[zoom_start:zoom_end], 'g', linewidth=1.0)
axes[4].set_ylabel('Amplitude')
axes[4].set_xlabel('Time (s)')
axes[4].set_title(f'Zoomed View: samples {zoom_start}-{zoom_end} ({t[zoom_start]:.3f}-{t[zoom_end]:.3f}s)')
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\34118\Desktop\SVD\wave_all_noise_comparison.png')
print("Saved plot: wave_all_noise_comparison.png")

# ============================================================
# 7. Statistics summary
# ============================================================
print(f"\n{'='*60}")
print(f"{'Component':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print(f"{'='*60}")
print(f"{'Original ECG':<25} {ecg.mean():>10.2f} {ecg.std():>10.2f} {ecg.min():>10.0f} {ecg.max():>10.0f}")
print(f"{'50Hz Noise':<25} {noise_50hz.mean():>10.2f} {noise_50hz.std():>10.2f} {noise_50hz.min():>10.0f} {noise_50hz.max():>10.0f}")
print(f"{'EMG':<25} {emg.mean():>10.2f} {emg.std():>10.2f} {emg.min():>10.0f} {emg.max():>10.0f}")
print(f"{'Motion Artifact':<25} {motion.mean():>10.2f} {motion.std():>10.2f} {motion.min():>10.0f} {motion.max():>10.0f}")
print(f"{'Baseline Wander':<25} {wander.mean():>10.2f} {wander.std():>10.2f} {wander.min():>10.0f} {wander.max():>10.0f}")
print(f"{'Equipment Noise':<25} {equipment.mean():>10.2f} {equipment.std():>10.2f} {equipment.min():>10.0f} {equipment.max():>10.0f}")
print(f"{'60Hz Environment':<25} {environment.mean():>10.2f} {environment.std():>10.2f} {environment.min():>10.0f} {environment.max():>10.0f}")
print(f"{'='*60}")
print(f"{'Final Noisy ECG':<25} {ecg_final.mean():>10.2f} {ecg_final.std():>10.2f} {ecg_final.min():>10.0f} {ecg_final.max():>10.0f}")

snr = 10 * np.log10(np.mean(ecg**2) / np.mean((ecg_final - ecg)**2))
print(f"\nSNR (original / all noise): {snr:.2f} dB")
