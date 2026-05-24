"""
SVD-based ECG denoising methods.

Implements four SVD denoising variants:
1. Standard SVD with overlapping windows
2. Hankel SVD (Singular Spectrum Analysis)
3. Recursive SVD (Brand 2006)
4. Randomized SVD via sklearn

Plus automatic rank estimation with elbow detection.
"""

import numpy as np
from scipy.linalg import svd, hankel
from sklearn.utils.extmath import randomized_svd as sklearn_randomized_svd


def standard_svd(signal, rank=5, window_len=50, overlap=0.44):
    """
    Standard SVD denoising with overlapping windows.

    Frames the signal into overlapping windows, performs SVD on each frame,
    keeps top `rank` components, and reconstructs via overlap-add.

    Parameters
    ----------
    signal : np.ndarray
        Input signal of shape (N,).
    rank : int
        Number of singular values to keep.
    window_len : int
        Length of each window/frame.
    overlap : float
        Overlap ratio between consecutive windows (0 to 1).

    Returns
    -------
    np.ndarray
        Denoised signal of shape (N,).
    """
    N = len(signal)
    step = int(window_len * (1 - overlap))
    if step < 1:
        step = 1

    # Frame signal into overlapping windows
    frames = []
    start = 0
    while start + window_len <= N:
        frames.append(signal[start:start + window_len])
        start += step

    if len(frames) == 0:
        return signal.copy()

    # Each frame = one row of matrix
    X = np.array(frames)

    # SVD decomposition
    U, s, Vt = svd(X, full_matrices=False)

    # Low-rank approximation
    rank = min(rank, len(s))
    X_denoised = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]

    # Overlap-add reconstruction
    output = np.zeros(N)
    count = np.zeros(N)
    start = 0
    for i, frame in enumerate(X_denoised):
        output[start:start + window_len] += frame
        count[start:start + window_len] += 1
        start += step

    # Average overlapping regions
    mask = count > 0
    output[mask] /= count[mask]

    return output


def hankel_svd(signal, rank=5, L=500):
    """
    Hankel SVD denoising (Singular Spectrum Analysis).

    Constructs a Hankel matrix from the signal, performs SVD decomposition,
    keeps top `rank` components, and reconstructs via diagonal averaging.

    Parameters
    ----------
    signal : np.ndarray
        Input signal of shape (N,).
    rank : int
        Number of singular values to keep.
    L : int
        Window length for Hankel matrix construction (N/4 recommended).

    Returns
    -------
    np.ndarray
        Denoised signal of shape (N,).
    """
    N = len(signal)
    L = min(L, N // 2)  # Ensure valid window length
    K = N - L + 1

    # Construct Hankel matrix
    # hankel(c, r) where c = first column, r = last row
    first_col = signal[:L]
    last_row = signal[L - 1:]
    H = hankel(first_col, last_row)

    # SVD decomposition
    U, s, Vt = svd(H, full_matrices=False)

    # Low-rank approximation
    rank = min(rank, len(s))
    H_denoised = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]

    # Diagonal averaging (inverse Hankel) to reconstruct signal
    output = _diagonal_averaging(H_denoised, N)

    return output


def recursive_svd(signal, rank=5, window_len=50):
    """
    Recursive SVD denoising using Brand 2006 rank-1 update algorithm.

    Processes signal in overlapping windows, updating SVD incrementally
    using the fast rank-1 modification algorithm from Brand (2006).
    Uses overlap-add reconstruction for high-quality output.

    Parameters
    ----------
    signal : np.ndarray
        Input signal of shape (N,).
    rank : int
        Number of singular values to keep.
    window_len : int
        Length of each window for incremental processing.

    Returns
    -------
    np.ndarray
        Denoised signal of shape (N,).
    """
    N = len(signal)
    rank = min(rank, window_len)
    overlap = 0.44
    step = max(1, int(window_len * (1 - overlap)))

    # Collect all frames
    frames = []
    start = 0
    while start + window_len <= N:
        frames.append(signal[start:start + window_len])
        start += step

    if len(frames) == 0:
        return signal.copy()

    # Initialize SVD with first frame
    col = frames[0].reshape(-1, 1)
    U, s, Vt = svd(col, full_matrices=False)
    U = U[:, :rank]
    s = s[:rank]
    Vt = Vt[:rank, :]

    # Incrementally add remaining frames using Brand 2006 update
    for i in range(1, len(frames)):
        new_col = frames[i].reshape(-1, 1)
        U, s, Vt = _brand_rank1_update(U, s, Vt, new_col, rank)

    # Reconstruct matrix from low-rank factors
    X_denoised = U @ np.diag(s) @ Vt

    # Overlap-add reconstruction
    output = np.zeros(N)
    count = np.zeros(N)
    start = 0
    for i in range(X_denoised.shape[1]):
        if start + window_len <= N:
            output[start:start + window_len] += X_denoised[:, i]
            count[start:start + window_len] += 1
        start += step

    mask = count > 0
    output[mask] /= count[mask]

    return output


def randomized_svd(signal, rank=5, L=500):
    """
    Randomized SVD denoising via sklearn.

    Uses Hankel matrix embedding (same as hankel_svd) but performs
    decomposition using randomized SVD from sklearn for efficiency.

    Parameters
    ----------
    signal : np.ndarray
        Input signal of shape (N,).
    rank : int
        Number of singular values to keep.
    L : int
        Window length for Hankel matrix construction.

    Returns
    -------
    np.ndarray
        Denoised signal of shape (N,).
    """
    N = len(signal)
    L = min(L, N // 2)

    # Construct Hankel matrix
    first_col = signal[:L]
    last_row = signal[L - 1:]
    H = hankel(first_col, last_row)

    # Randomized SVD decomposition
    rank = min(rank, min(H.shape) - 1)
    U, s, Vt = sklearn_randomized_svd(H, n_components=rank, random_state=42)

    # Low-rank approximation
    H_denoised = U @ np.diag(s) @ Vt

    # Diagonal averaging to reconstruct signal
    output = _diagonal_averaging(H_denoised, N)

    return output


def notch_svd_50hz(signal, fs=250, L=500):
    """
    SVD-based denoising specifically targeting 50Hz powerline interference.

    Algorithm:
    1. Construct Hankel matrix from noisy signal
    2. Compute SVD
    3. Identify components corresponding to 50Hz interference:
       - A pure sinusoid at frequency f creates rank-2 structure in Hankel matrix
       - These components have specific frequency content in their singular vectors
    4. Remove 50Hz components from the low-rank approximation
    5. Reconstruct signal via diagonal averaging

    Parameters
    ----------
    signal : np.ndarray
        Input signal of shape (N,).
    fs : int
        Sampling frequency (default: 250 Hz).
    L : int
        Hankel window length (default: 500).

    Returns
    -------
    np.ndarray
        Denoised signal with 50Hz interference removed.
    """
    N = len(signal)
    L = min(L, N // 2)

    # Step 1: Construct Hankel matrix
    H = hankel(signal[:L], signal[L - 1:])

    # Step 2: Compute SVD
    U, s, Vt = svd(H, full_matrices=False)

    # Step 3: Identify 50Hz components
    # A sinusoid at 50Hz with fs=250 has period = 5 samples
    # In the Hankel matrix, this creates a specific pattern in singular vectors
    target_freq = 50  # Hz

    # For each singular vector pair, check if it corresponds to 50Hz
    # by analyzing the frequency content of the left singular vectors
    n_components = len(s)
    keep_mask = np.ones(n_components, dtype=bool)

    for i in range(min(n_components, 20)):  # Check first 20 components
        # Get the left singular vector
        u = U[:, i]

        # Compute its dominant frequency using FFT
        fft_u = np.fft.rfft(u)
        freqs = np.fft.rfftfreq(len(u), d=1.0 / fs)
        magnitudes = np.abs(fft_u)

        # Find dominant frequency (excluding DC)
        if len(magnitudes) > 1:
            dominant_idx = np.argmax(magnitudes[1:]) + 1
            dominant_freq = freqs[dominant_idx]

            # If dominant frequency is close to 50Hz, mark for removal
            if abs(dominant_freq - target_freq) < 2:  # Within 2Hz tolerance
                keep_mask[i] = False

    # Step 4: Reconstruct without 50Hz components
    # Zero out the 50Hz components
    s_filtered = s.copy()
    s_filtered[~keep_mask] = 0

    # Reconstruct Hankel matrix
    H_filtered = U @ np.diag(s_filtered) @ Vt

    # Step 5: Diagonal averaging to reconstruct signal
    output = _diagonal_averaging(H_filtered, N)

    return output


def estimate_rank(signal, max_rank=10):
    """
    Automatic rank estimation using elbow detection on singular values.

    Computes SVD of Hankel matrix and detects the elbow point using
    the kneed library. Falls back to rank=5 if kneed is unavailable.

    Parameters
    ----------
    signal : np.ndarray
        Input signal of shape (N,).
    max_rank : int
        Maximum rank to consider.

    Returns
    -------
    int
        Estimated rank in [1, max_rank].
    """
    N = len(signal)
    L = min(500, N // 2)

    # Construct Hankel matrix
    first_col = signal[:L]
    last_row = signal[L - 1:]
    H = hankel(first_col, last_row)

    # Compute SVD
    _, s, _ = svd(H, full_matrices=False)

    # Try kneed library for elbow detection
    try:
        from kneed import KneeLocator
        max_rank = min(max_rank, len(s))
        x = list(range(1, max_rank + 1))
        y = s[:max_rank].tolist()
        kneedle = KneeLocator(x, y, curve='convex', direction='decreasing')
        if kneedle.knee is not None:
            return int(kneedle.knee)
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: use energy-based heuristic
    total_energy = np.sum(s ** 2)
    cumulative_energy = np.cumsum(s ** 2) / total_energy
    # Find rank that captures 90% of energy
    rank_90 = np.searchsorted(cumulative_energy, 0.9) + 1
    rank_90 = max(1, min(rank_90, max_rank))

    return int(rank_90)


def _diagonal_averaging(H, N):
    """
    Diagonal averaging (inverse Hankel transform) to reconstruct signal
    from a Hankel matrix.

    Parameters
    ----------
    H : np.ndarray
        Hankel matrix of shape (L, K).
    N : int
        Expected output signal length.

    Returns
    -------
    np.ndarray
        Reconstructed signal of shape (N,).
    """
    L, K = H.shape
    output = np.zeros(N)
    count = np.zeros(N)

    for i in range(L):
        for j in range(K):
            idx = i + j
            if idx < N:
                output[idx] += H[i, j]
                count[idx] += 1

    mask = count > 0
    output[mask] /= count[mask]

    return output


def _brand_rank1_update(U, s, Vt, new_col, rank):
    """
    Brand 2006 fast rank-1 SVD update.

    Given existing SVD of matrix M = U @ diag(s) @ Vt and a new column b,
    computes the updated SVD of [M | b] without recomputing from scratch.

    Reference: Brand (2006) "Fast Low-Rank Modifications of the Thin SVD"

    Parameters
    ----------
    U : np.ndarray
        Left singular vectors, shape (n, k).
    s : np.ndarray
        Singular values, shape (k,).
    Vt : np.ndarray
        Right singular vectors transposed, shape (k, m).
    new_col : np.ndarray
        New column to append, shape (n, 1).
    rank : int
        Target rank to maintain.

    Returns
    -------
    tuple
        Updated (U, s, Vt).
    """
    n = U.shape[0]
    m = Vt.shape[1]
    k = len(s)

    # Project new column onto existing left singular vectors
    p = U.T @ new_col  # (k, 1)
    r = new_col - U @ p  # (n, 1)

    # Normalize residual
    rho = np.linalg.norm(r)
    if rho > 1e-10:
        r_hat = r / rho
    else:
        r_hat = np.zeros_like(r)

    # Build augmented k+1 x k+1 matrix:
    # K = [[diag(s), p],
    #      [0,       rho]]
    K = np.zeros((k + 1, k + 1))
    K[:k, :k] = np.diag(s)
    K[:k, k] = p.flatten()
    K[k, k] = rho

    # SVD of the small augmented matrix
    Uk, sk, Vtk = svd(K, full_matrices=False)

    # Update left singular vectors: U_new = [U, r_hat] @ Uk
    U_aug = np.hstack([U, r_hat])  # (n, k+1)
    U_new = U_aug @ Uk  # (n, k+1)

    # Update right singular vectors:
    # Vt_new = Vtk @ [[Vt, 0], [0, 1]]
    Vt_aug = np.zeros((k + 1, m + 1))
    Vt_aug[:k, :m] = Vt
    Vt_aug[k, m] = 1.0
    Vt_new = Vtk @ Vt_aug  # (k+1, m+1)

    # Truncate to target rank
    rank = min(rank, len(sk))
    U_new = U_new[:, :rank]
    s_new = sk[:rank]
    Vt_new = Vt_new[:rank, :]

    return U_new, s_new, Vt_new
