"""
绘制noisy_50hz_interference.csv的波形
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 加载数据
df = pd.read_csv('results/noisy_50hz_interference.csv')
signal = df['noisy_signal'].values

# 绘制波形
fig, ax = plt.subplots(figsize=(12, 4), dpi=300)
ax.plot(signal, color='#DC2626', linewidth=0.8)
ax.set_title('Noisy ECG with 50Hz Interference (Amplitude ≈ R-wave)', fontsize=12, fontweight='bold')
ax.set_xlabel('Sample', fontsize=10)
ax.set_ylabel('Amplitude', fontsize=10)
ax.grid(True, alpha=0.3)

# 保存
os.makedirs('results', exist_ok=True)
plt.tight_layout()
plt.savefig('results/noisy_50hz_waveform.png', dpi=300, bbox_inches='tight')
print('Saved: results/noisy_50hz_waveform.png')
