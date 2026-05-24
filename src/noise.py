"""
Noise generation module for ECG signal simulation.

Provides 6 noise types commonly found in ECG recordings:
1. Baseline wander (0.05-0.5 Hz)
2. Powerline interference (50/60 Hz)
3. EMG noise (20-500 Hz)
4. Motion artifacts (transient)
5. Measurement error (1/f pink noise)
6. White noise (broadband)

All generators return 1D numpy arrays of specified length.
"""

import numpy as np
from scipy.signal import butter, filtfilt

np.random.seed(42)


def generate_baseline_wander(n, fs=250):
    """
    Generate baseline wander noise (low-frequency respiratory/cardiac motion).

    Sum of sinusoids in 0.05-0.5 Hz range with random phases.

    Parameters
    ----------
    n : int
        Number of samples
    fs : int
        Sampling frequency in Hz (default: 250)

    Returns
    -------
    np.ndarray
        Baseline wander noise of shape (n,)
    """
    t = np.arange(n) / fs
    noise = np.zeros(n)
    # Sum of low-frequency sinusoids
    for freq in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.5, 1.5)
        noise += amplitude * np.sin(2 * np.pi * freq * t + phase)
    return noise


def generate_powerline_interference(n, fs=250, freq=50):
    """
    Generate powerline interference noise.

    Single sinusoid at specified frequency with random phase.

    Parameters
    ----------
    n : int
        Number of samples
    fs : int
        Sampling frequency in Hz (default: 250)
    freq : float
        Powerline frequency in Hz (default: 50)

    Returns
    -------
    np.ndarray
        Powerline interference noise of shape (n,)
    """
    t = np.arange(n) / fs
    phase = np.random.uniform(0, 2 * np.pi)
    noise = np.sin(2 * np.pi * freq * t + phase)
    return noise


def generate_emg_noise(n, fs=250):
    """
    Generate EMG (electromyography) noise.

    Bandpass filtered white noise in 20-500 Hz range using Butterworth filter.

    Parameters
    ----------
    n : int
        Number of samples
    fs : int
        Sampling frequency in Hz (default: 250)

    Returns
    -------
    np.ndarray
        EMG noise of shape (n,)
    """
    # Generate white noise
    white = np.random.randn(n)

    # Design bandpass filter (20-500 Hz)
    nyq = fs / 2.0
    low = 20.0 / nyq
    high = min(500.0 / nyq, 0.99)  # Ensure below Nyquist
    b, a = butter(4, [low, high], btype='band')

    # Apply zero-phase filtering
    noise = filtfilt(b, a, white)
    return noise


def generate_motion_artifact(n, fs=250):
    """
    Generate motion artifact noise.

    Random step and ramp transients at random positions.

    Parameters
    ----------
    n : int
        Number of samples
    fs : int
        Sampling frequency in Hz (default: 250)

    Returns
    -------
    np.ndarray
        Motion artifact noise of shape (n,)
    """
    noise = np.zeros(n)

    # Add 2-5 random transients
    num_transients = np.random.randint(2, 6)
    for _ in range(num_transients):
        # Random position and duration
        pos = np.random.randint(0, n - 100)
        duration = np.random.randint(10, 50)
        amplitude = np.random.uniform(-2, 2)

        # Randomly choose step or ramp
        if np.random.random() > 0.5:
            # Step transient
            noise[pos:pos + duration] += amplitude
        else:
            # Ramp transient
            ramp = np.linspace(0, amplitude, duration)
            noise[pos:pos + duration] += ramp

    return noise


def generate_measurement_error(n, fs=250):
    """
    Generate measurement error noise (1/f pink noise).

    Generated via frequency domain shaping: white noise multiplied by 1/sqrt(f).

    Parameters
    ----------
    n : int
        Number of samples
    fs : int
        Sampling frequency in Hz (default: 250)

    Returns
    -------
    np.ndarray
        Measurement error noise of shape (n,)
    """
    # Generate white noise in frequency domain
    white_freq = np.fft.rfft(np.random.randn(n))

    # Create 1/f shaping
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    freqs[0] = 1.0  # Avoid division by zero at DC
    shaping = 1.0 / np.sqrt(freqs)

    # Apply shaping
    shaped_freq = white_freq * shaping

    # Transform back to time domain
    noise = np.fft.irfft(shaped_freq, n=n)
    return noise


def generate_white_noise(n):
    """
    Generate standard Gaussian white noise.

    Parameters
    ----------
    n : int
        Number of samples

    Returns
    -------
    np.ndarray
        White noise of shape (n,)
    """
    return np.random.randn(n)


def add_noise(signal, noise, snr_db):
    """
    Add noise to signal at specified SNR level.

    Scales noise to achieve target SNR and returns noisy signal.

    Parameters
    ----------
    signal : np.ndarray
        Clean signal
    noise : np.ndarray
        Noise signal (same length as signal)
    snr_db : float
        Target signal-to-noise ratio in dB

    Returns
    -------
    tuple of np.ndarray
        (noisy_signal, scaled_noise)
    """
    # Calculate signal and noise power
    p_signal = np.var(signal)
    p_noise = np.var(noise)

    # Scale noise to achieve target SNR
    # SNR_dB = 10*log10(P_signal/P_noise_scaled)
    # P_noise_scaled = P_signal / 10^(SNR_dB/10)
    # scale^2 = P_noise_scaled / P_noise
    scale = np.sqrt(p_signal / (p_noise * 10 ** (snr_db / 10)))
    scaled_noise = noise * scale

    return (signal + scaled_noise, scaled_noise)


def generate_noise_by_type(n, noise_type, fs=250):
    """
    Router function to generate noise by type name.

    Parameters
    ----------
    n : int
        Number of samples
    noise_type : str
        One of: 'baseline', 'powerline', 'emg', 'motion', 'measurement', 'white'
    fs : int
        Sampling frequency in Hz (default: 250)

    Returns
    -------
    np.ndarray
        Generated noise of shape (n,)

    Raises
    ------
    ValueError
        If noise_type is not recognized
    """
    if noise_type == 'baseline':
        return generate_baseline_wander(n, fs)
    elif noise_type == 'powerline':
        return generate_powerline_interference(n, fs)
    elif noise_type == 'emg':
        return generate_emg_noise(n, fs)
    elif noise_type == 'motion':
        return generate_motion_artifact(n, fs)
    elif noise_type == 'measurement':
        return generate_measurement_error(n, fs)
    elif noise_type == 'white':
        return generate_white_noise(n)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
