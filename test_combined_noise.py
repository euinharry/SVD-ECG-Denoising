"""
在noisy_50hz_interference.csv基础上叠加多种噪声，测试SVD算法效果
"""
import pandas as pd
import numpy as np
from src.data_loader import load_ecg_signal, FS
from src.noise import generate_baseline_wander, generate_motion_artifact, generate_measurement_error, generate_white_noise, generate_powerline_interference, add_noise
from src.svd_methods import standard_svd, hankel_svd, recursive_svd, randomized_svd, notch_svd_50hz
from src.metrics import calculate_all_metrics
from src.plotting import plot_noise_comparison
import os

np.random.seed(42)

# 1. 加载原始干净信号
clean = load_ecg_signal('wave.csv')
n = len(clean)

# 2. 加载或生成50Hz干扰信号
csv_path = 'results/noisy_50hz_interference.csv'
if os.path.exists(csv_path):
    noisy_50hz = pd.read_csv(csv_path)['noisy_signal'].values
    print('=' * 60)
    print('基础信号: noisy_50hz_interference.csv (已含50Hz干扰)')
    print('=' * 60)
else:
    print('=' * 60)
    print('noisy_50hz_interference.csv 不存在, 自动生成50Hz干扰信号')
    print('=' * 60)
    powerline = generate_powerline_interference(n, FS, freq=50)
    noisy_50hz, _ = add_noise(clean, powerline, snr_db=0)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame({'noisy_signal': noisy_50hz}).to_csv(csv_path, index=False)
    print(f'已保存: {csv_path}')

# 3. 生成其他噪声类型
r_wave_amp = np.max(np.abs(clean))
noise_scale = 0.3 * r_wave_amp

baseline = generate_baseline_wander(n, FS)
motion = generate_motion_artifact(n, FS)
measurement = generate_measurement_error(n, FS)
white = generate_white_noise(n)

# 归一化后缩放到目标幅度
baseline = baseline / np.std(baseline) * noise_scale
motion = motion / np.std(motion) * noise_scale
measurement = measurement / np.std(measurement) * noise_scale
white = white / np.std(white) * noise_scale

# 4. 叠加所有噪声到50Hz干扰信号上
noisy_combined = noisy_50hz + baseline + motion + measurement + white

print(f'R波幅度: {r_wave_amp:.2f}')
print(f'叠加噪声幅度: {noise_scale:.2f} (R波的30%)')
print()

# 5. 测试5种SVD方法
methods = {
    'standard': lambda x: standard_svd(x, rank=5),
    'hankel': lambda x: hankel_svd(x, rank=5),
    'recursive': lambda x: recursive_svd(x, rank=5),
    'randomized': lambda x: randomized_svd(x, rank=5),
    'notch_svd_50hz': lambda x: notch_svd_50hz(x, fs=FS),
}

print('测试结果:')
print('-' * 60)
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
    print(f"{name}: SNR改善={metrics['snr_improvement']:+.2f}dB, 相关系数={metrics['correlation']:.4f}")

# 6. 保存叠加后的信号
noisy_df = pd.DataFrame({'noisy_combined': noisy_combined})
noisy_df.to_csv('results/noisy_combined_with_50hz.csv', index=False)
print('\n已保存: results/noisy_combined_with_50hz.csv')

# 7. 生成对比图
def normalize_to_range(x):
    xmin, xmax = x.min(), x.max()
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

clean_norm = normalize_to_range(clean)
noisy_norm = normalize_to_range(noisy_combined)
denoised_dict_norm = {k: normalize_to_range(v) for k, v in denoised_dict.items()}

os.makedirs('results', exist_ok=True)
plot_noise_comparison(
    clean_norm, noisy_norm, denoised_dict_norm,
    '50Hz_Baseline_Motion_Measurement_White', 0,
    'results/combined_noise_with_50hz_comparison.png'
)
print('已保存: results/combined_noise_with_50hz_comparison.png')

# 8. 保存指标
results_df = pd.DataFrame(results)
results_df.to_csv('results/combined_noise_with_50hz_metrics.csv', index=False)
print('已保存: results/combined_noise_with_50hz_metrics.csv')
