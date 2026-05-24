"""ECG signal data loading and preprocessing module."""

import numpy as np

# Sampling frequency in Hz
FS = 250


def load_ecg_signal(filepath: str) -> np.ndarray:
    """Load ECG signal from a single-column CSV file.

    Reads the CSV, validates that the data has shape (2000,), removes the
    DC offset by subtracting the mean, and returns a 1-D NumPy array.

    Parameters
    ----------
    filepath : str
        Path to a single-column CSV file with 2000 rows and no header.

    Returns
    -------
    np.ndarray
        1-D array of shape (2000,) with zero mean (DC offset removed).

    Raises
    ------
    ValueError
        If the loaded data does not have shape (2000,).
    """
    x = np.loadtxt(filepath, delimiter=",")
    if x.shape != (2000,):
        raise ValueError(
            f"Expected signal shape (2000,), got {x.shape}"
        )
    x = x - x.mean()
    return x


def normalize_signal(x: np.ndarray) -> np.ndarray:
    """Normalize a signal to the range [-1, 1].

    Divides the signal by the maximum absolute value so that the resulting
    signal peaks at exactly -1 and +1.

    Parameters
    ----------
    x : np.ndarray
        Input 1-D signal.

    Returns
    -------
    np.ndarray
        Normalized signal in the range [-1, 1].
    """
    xmin, xmax = x.min(), x.max()
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0
