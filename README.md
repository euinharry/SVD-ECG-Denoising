# SVD-ECG-Denoising

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![SciPy](https://img.shields.io/badge/SciPy-1.7+-orange?logo=scipy)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-blue?logo=numpy)

---

## 项目概述

本项目系统研究了 12 种基于奇异值分解（SVD）的心电信号（ECG）去噪方法。ECG 信号在采集和传输过程中极易受到各类噪声的干扰，影响医学诊断的准确性。我们模拟了 6 种真实世界中常见的噪声类型，并在统一的实验框架下对传统 SVD 方法和新型改进方法进行了全面的对比评估。

项目的核心贡献在于：
- 构建了完整的噪声模拟管线，覆盖从低频基线漂移到宽带白噪声的各类干扰
- 实现了 4 种经典 SVD 去噪方法，以及 8 种带预处理的改进方案
- 设计了全因子实验，涵盖 6 种噪声 x 4 种信噪比 x 4 种 SVD 方法的共 96 种工况
- 提供了 SNR 改善、RMSE、相关系数和 PRD 四种定量评价指标

---

## 噪声类型

| 编号 | 噪声名称 | 频率范围 | 说明 |
|------|----------|----------|------|
| 1 | 基线漂移 | 0.05-0.5 Hz | 呼吸、电极移动引起的低频干扰 |
| 2 | 电源线干扰 | 50 Hz | 工频电磁场耦合产生的正弦噪声 |
| 3 | 肌电噪声 | 20-500 Hz | 肌肉收缩产生的高频随机信号 |
| 4 | 运动伪影 | 瞬态低频 | 电极与皮肤相对位移产生的突发干扰 |
| 5 | 测量误差 | 1/f 粉红噪声 | 量化误差和放大器噪声的等效模型 |
| 6 | 白噪声 | 全频段 | 热噪声、散粒噪声等宽谱干扰 |

---

## SVD 去噪方法

### 传统方法（Old Methods）

| 编号 | 方法 | 核心策略 |
|------|------|----------|
| 1 | Basic SVD Truncation | 构建 Hankel 矩阵，保留前 3 个奇异值重构 |
| 2 | SSA (L=60) | 奇异谱分析，窗口长度 60，分组重构 |
| 3 | Optimal Rank SVD | 通过 SVR 曲线拐点自动确定最优截断秩 |
| 4 | Block-SVD | 128 样本分块处理，相邻块 64 样本重叠 |
| 5 | Targeted SVD | 多级去噪，逐级定位并移除噪声分量 |
| 6 | Multichannel SVD | 多导联联合处理，结合 Wiener 软阈值滤波 |

### 新方法（New Methods，含预处理）

| 编号 | 方法 | 核心策略 |
|------|------|----------|
| 7 | Notch + SSA | 50 Hz 陷波滤波器预处理，再结合模板相关系数筛选 SSA 分量 |
| 8 | Beat-Aligned SVD | 基于 R 峰检测的心拍对齐，对每个心拍独立进行 SVD |
| 9 | Iterative SSA | 5 轮迭代，每轮逐步剪枝噪声分量 |
| 10 | Freq-Banded SVD | 将信号分解为 5 个子频带分别去噪后重构 |
| 11 | Cross-Validated SVD | 对 Rank 2-15 逐级评分，通过交叉验证选择最优秩 |
| 12 | Combined Best | 融合 Beat-Aligned 和 SSA 模板相关，取各方法最优 |

---

## 实验设计

采用全因子实验设计，覆盖以下维度：

| 维度 | 取值 |
|------|------|
| 噪声类型 | 基线漂移、电源线干扰、肌电噪声、运动伪影、测量误差、白噪声 |
| 信噪比 (SNR) | -5 dB, 0 dB, 5 dB, 10 dB |
| SVD 方法 | Basic SVD Truncation, SSA, Optimal Rank SVD, Block-SVD |
| 总工况数 | 6 x 4 x 4 = **96 种** |

信号参数：采样率 250 Hz，信号长度 2000 个采样点（8 秒），提取标准 12 导联 ECG 记录进行实验。

---

## 评价指标

| 指标 | 缩写 | 含义 |
|------|------|------|
| SNR 改善 | SNR Imp. | 去噪前后信噪比提升量 (dB) |
| 均方根误差 | RMSE | 重构信号与纯净信号之间的误差 |
| 相关系数 | CC | 重构信号与纯净信号之间的线性相关度 |
| 百分比均方根差 | PRD | 归一化的失真百分比，衡量信号保真度 |

这些指标从不同角度反映了去噪效果：SNR 改善关注噪声抑制能力，RMSE 衡量绝对误差，CC 反映波形保真度，PRD 则综合评估信号失真程度。

---

## 项目结构

```
SVD-ECG-Denoising/
├── main.py                      # 全因子实验主程序（96 种工况）
├── svd_ecg_denoise.py           # 12 种 SVD 方法对比实验
├── src/
│   ├── data_loader.py           # ECG 数据加载与预处理
│   ├── noise.py                 # 6 种噪声生成器
│   ├── svd_methods.py           # 4 种核心 SVD 去噪算法
│   ├── metrics.py               # 评价指标计算
│   └── plotting.py              # 可视化绘图工具
├── draft/
│   ├── wave.csv                 # 纯净 ECG 信号
│   ├── wave_with_50hz.csv       # 叠加 50 Hz 工频干扰
│   ├── wave_with_all_noise.csv  # 叠加多种噪声
│   ├── wave_comparison.png      # 去噪前后对比图
│   └── wave_all_noise_comparison.png
├── results/                     # 实验结果输出目录
├── requirements.txt             # Python 依赖
├── README.md                    # 本文件
└── LICENSE                      # MIT 许可证
```

---

## 安装与使用

### 环境要求

- Python 3.8 或更高版本
- 建议使用虚拟环境运行

### 安装步骤

```bash
git clone https://github.com/your-username/SVD-ECG-Denoising.git
cd SVD-ECG-Denoising
pip install -r requirements.txt
```

### 运行实验

```bash
python main.py
python svd_ecg_denoise.py
```

运行完成后，结果将保存在 `results/` 目录下，包括各项评价指标的汇总表和各方法去噪效果的对比图。

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源。您可以自由使用、修改和分发本项目的代码，但需保留原版权声明。

---

## 引用

如果您的研究使用了本项目，请考虑引用：

```bibtex
@software{svd_ecg_denoising,
  title = {SVD-ECG-Denoising: 基于奇异值分解的心电信号去噪方法系统对比},
  year = {2026},
  url = {https://github.com/your-username/SVD-ECG-Denoising}
}
```
