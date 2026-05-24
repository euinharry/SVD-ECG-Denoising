import numpy as np
import matplotlib.pyplot as plt

# 1. Load ECG data
ecg = np.loadtxt(r'C:\Users\34118\Desktop\SVD\wave.csv')
print(f"Loaded ECG: {len(ecg)} samples, range [{ecg.min():.2f}, {ecg.max():.2f}]")

# 2. Generate 50Hz sine noise with amplitude modulation
np.random.seed(42)
phase = np.random.uniform(0, 2 * np.pi)
t = np.arange(2000) / 250
N = 2000
envelope = 1.0 + 0.15 * np.sin(2 * np.pi * 0.5 * t) + 0.08 * np.random.randn(N)
envelope = np.clip(envelope, 0.7, 1.3)
noise = 50000 * envelope * np.sin(2 * np.pi * 50 * t + phase)
print(f"50Hz noise: base amplitude=50000, envelope range [{envelope.min():.3f}, {envelope.max():.3f}], phase={phase:.4f} rad")

# 3. Add noise
noisy_ecg = ecg + noise

# 4. Save noisy signal
np.savetxt(r'C:\Users\34118\Desktop\SVD\wave_with_50hz.csv', noisy_ecg, fmt='%.7f')
print(f"Saved noisy ECG: wave_with_50hz.csv")

# 5. Plot comparison
fig, axes = plt.subplots(3, 1, figsize=(16, 10), dpi=150, sharex=True)

axes[0].plot(t, ecg, 'b', linewidth=0.5)
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Original ECG Signal')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, noise, 'r', linewidth=0.5)
axes[1].set_ylabel('Amplitude')
axes[1].set_title('50Hz Power Line Noise (base amplitude=50000, modulated)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, noisy_ecg, 'g', linewidth=0.5)
axes[2].set_ylabel('Amplitude')
axes[2].set_xlabel('Time (s)')
axes[2].set_title('ECG with 50Hz Noise')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'C:\Users\34118\Desktop\SVD\wave_comparison.png')
print("Saved plot: wave_comparison.png")

# 6. Print stats
print(f"\n--- Statistics ---")
print(f"Original ECG  : mean={ecg.mean():.2f}, std={ecg.std():.2f}, min={ecg.min():.2f}, max={ecg.max():.2f}")
print(f"50Hz Noise    : mean={noise.mean():.2f}, std={noise.std():.2f}, min={noise.min():.2f}, max={noise.max():.2f}")
print(f"Noisy ECG     : mean={noisy_ecg.mean():.2f}, std={noisy_ecg.std():.2f}, min={noisy_ecg.min():.2f}, max={noisy_ecg.max():.2f}")
print(f"SNR (original/noise power): {10*np.log10(np.mean(ecg**2)/np.mean(noise**2)):.2f} dB")
