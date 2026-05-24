"""
Metrics module for evaluating ECG denoising performance.

Provides three standard metrics:
1. SNR improvement (dB) — how much the denoising improved signal quality
2. Mean Squared Error (MSE) — average squared difference from clean signal
3. Pearson correlation — linear similarity between clean and denoised signals
"""

import numpy as np


def calculate_snr_improvement(clean, noisy, denoised):
    """
    Calculate SNR improvement in dB after denoising.

    SNR improvement = SNR_after - SNR_before, where
    SNR = 10*log10(var(signal) / var(noise_residual)).

    Parameters
    ----------
    clean : np.ndarray
        Original clean signal of shape (N,).
    noisy : np.ndarray
        Noisy signal of shape (N,).
    denoised : np.ndarray
        Denoised signal of shape (N,).

    Returns
    -------
    float
        SNR improvement in dB. Positive means denoising helped.
    """
    snr_before = 10 * np.log10(np.var(clean) / np.var(clean - noisy))
    snr_after = 10 * np.log10(np.var(clean) / np.var(clean - denoised))
    return snr_after - snr_before


def calculate_mse(clean, denoised):
    """
    Calculate Mean Squared Error between clean and denoised signals.

    MSE = mean((clean - denoised)^2).

    Parameters
    ----------
    clean : np.ndarray
        Original clean signal of shape (N,).
    denoised : np.ndarray
        Denoised signal of shape (N,).

    Returns
    -------
    float
        Mean squared error (non-negative).
    """
    return np.mean((clean - denoised) ** 2)


def calculate_correlation(clean, denoised):
    """
    Calculate Pearson correlation coefficient between clean and denoised signals.

    Parameters
    ----------
    clean : np.ndarray
        Original clean signal of shape (N,).
    denoised : np.ndarray
        Denoised signal of shape (N,).

    Returns
    -------
    float
        Pearson correlation coefficient in [-1, 1]. Higher is better.
    """
    return np.corrcoef(clean, denoised)[0, 1]


def calculate_all_metrics(clean, noisy, denoised):
    """
    Calculate all three denoising metrics at once.

    Parameters
    ----------
    clean : np.ndarray
        Original clean signal of shape (N,).
    noisy : np.ndarray
        Noisy signal of shape (N,).
    denoised : np.ndarray
        Denoised signal of shape (N,).

    Returns
    -------
    dict
        Dictionary with keys: 'snr_improvement', 'mse', 'correlation'.
    """
    return {
        'snr_improvement': calculate_snr_improvement(clean, noisy, denoised),
        'mse': calculate_mse(clean, denoised),
        'correlation': calculate_correlation(clean, denoised),
    }
