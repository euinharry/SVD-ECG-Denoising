"""
Visualization module for ECG denoising comparison.

Three plotting functions:
1. plot_noise_comparison — 2×3 subplot grid comparing clean/noisy/denoised signals
2. plot_metrics_summary — grouped bar chart + heatmap of denoising metrics
3. plot_singular_values — scree plot of singular value decay from Hankel SVD
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for file saving
import matplotlib.pyplot as plt
from scipy.linalg import svd, hankel


# ── Design tokens (consistent across all plots) ──────────────────────────────
COLORS = {
    'clean': '#2563EB',       # blue-600
    'noisy': '#DC2626',       # red-600
    'denoised': '#16A34A',    # green-600
    'standard': '#8B5CF6',    # violet-500
    'hankel': '#0891B2',      # cyan-600
    'recursive': '#EA580C',   # orange-600
    'randomized': '#DB2777',  # pink-600
    'notch_svd_50hz': '#059669',  # emerald-600
    'grid': '#E5E7EB',        # gray-200
    'bg': '#FAFAFA',          # gray-50
    'text': '#1F2937',        # gray-800
    'text_secondary': '#6B7280',  # gray-500
    'accent': '#1D4ED8',      # blue-700
}
METHOD_COLORS = ['standard', 'hankel', 'recursive', 'randomized', 'notch_svd_50hz']
LINEWIDTH = 1.0
FONTSIZE_TITLE = 10
FONTSIZE_LABEL = 8
FONTSIZE_TICK = 7


def _apply_ax_style(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent styling to an axes object."""
    ax.set_facecolor(COLORS['bg'])
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linewidth=0.5)
    ax.tick_params(labelsize=FONTSIZE_TICK, colors=COLORS['text_secondary'])
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(0.5)
    if title:
        ax.set_title(title, fontsize=FONTSIZE_TITLE, color=COLORS['text'], pad=6, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONTSIZE_LABEL, color=COLORS['text_secondary'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONTSIZE_LABEL, color=COLORS['text_secondary'])


def _compute_metrics(clean, noisy, denoised):
    """Compute SNR improvement and correlation for a denoised signal."""
    snr_before = 10 * np.log10(np.var(clean) / np.var(clean - noisy))
    snr_after = 10 * np.log10(np.var(clean) / np.var(clean - denoised))
    snr_imp = snr_after - snr_before
    corr = np.corrcoef(clean, denoised)[0, 1]
    return snr_imp, corr


def plot_noise_comparison(clean, noisy, denoised_dict, noise_type, snr_db, save_path):
    """
    Create a 2×3 subplot grid comparing original, noisy, and denoised signals.

    Parameters
    ----------
    clean : np.ndarray
        Original clean signal of shape (N,).
    noisy : np.ndarray
        Noisy signal of shape (N,).
    denoised_dict : dict
        Mapping of method name → denoised signal array.
        Expected keys: 'standard', 'hankel', 'recursive', 'randomized'.
    noise_type : str
        Name of the noise type (e.g., 'white_noise', 'pink_noise').
    snr_db : float
        Signal-to-noise ratio in dB used to generate the noisy signal.
    save_path : str
        File path to save the PNG figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(16, 10), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    fig.suptitle(
        f'ECG Denoising Comparison — {noise_type.replace("_", " ").title()} (SNR={snr_db}dB)',
        fontsize=12, color=COLORS['text'], fontweight='bold', y=0.98
    )

    # ── Subplot 1: Original clean signal ──
    ax = axes[0, 0]
    ax.plot(clean, color=COLORS['clean'], linewidth=LINEWIDTH)
    _apply_ax_style(ax, title='Original Clean Signal', xlabel='Sample', ylabel='Amplitude')

    # ── Subplot 2: Noisy signal ──
    ax = axes[0, 1]
    ax.plot(noisy, color=COLORS['noisy'], linewidth=LINEWIDTH, alpha=0.85)
    _apply_ax_style(ax, title=f'Noisy Signal (SNR={snr_db}dB)', xlabel='Sample', ylabel='Amplitude')

    # ── Subplots 3-7: Denoised signals ──
    method_positions = [(0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]
    for idx, method in enumerate(METHOD_COLORS):
        row, col = method_positions[idx]
        ax = axes[row, col]
        if method in denoised_dict:
            denoised = denoised_dict[method]
            ax.plot(denoised, color=COLORS.get(method, COLORS['denoised']), linewidth=LINEWIDTH)
            snr_imp, corr = _compute_metrics(clean, noisy, denoised)
            title = f'{method.title()} SVD\nΔSNR={snr_imp:+.1f}dB  r={corr:.3f}'
        else:
            title = f'{method.title()} SVD\n(not available)'
        _apply_ax_style(ax, title=title, xlabel='Sample', ylabel='Amplitude')

    # ── Hide unused subplots ──
    used_positions = set(method_positions[:len(METHOD_COLORS)])
    for r in range(3):
        for c in range(3):
            if (r, c) not in used_positions and (r, c) != (0, 0) and (r, c) != (0, 1):
                axes[r, c].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close(fig)


def plot_metrics_summary(metrics_df, save_path):
    """
    Create a figure with grouped bar chart (SNR improvement) and heatmap (best method).

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with columns: noise_type, snr_db, method, snr_improvement, mse, correlation, runtime.
    save_path : str
        File path to save the PNG figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax_bar, ax_heat) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    fig.suptitle('Denoising Performance Summary', fontsize=12, color=COLORS['text'], fontweight='bold', y=0.98)

    # ── Left: Grouped bar chart — SNR improvement by noise type, grouped by method ──
    noise_types = metrics_df['noise_type'].unique()
    methods = metrics_df['method'].unique()
    n_noise = len(noise_types)
    n_methods = len(methods)

    # Compute mean SNR improvement per (noise_type, method)
    grouped = metrics_df.groupby(['noise_type', 'method'])['snr_improvement'].mean().unstack(fill_value=0)

    x = np.arange(n_noise)
    total_width = 0.8
    bar_width = total_width / n_methods

    for i, method in enumerate(methods):
        if method in grouped.columns:
            values = grouped[method].values
        else:
            values = np.zeros(n_noise)
        offset = (i - n_methods / 2 + 0.5) * bar_width
        color = COLORS.get(method, '#6B7280')
        ax_bar.bar(x + offset, values, bar_width * 0.9, label=method.title(), color=color, alpha=0.85)

    _apply_ax_style(ax_bar, title='SNR Improvement by Noise Type & Method',
                    xlabel='Noise Type', ylabel='SNR Improvement (dB)')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([nt.replace('_', '\n') for nt in noise_types], fontsize=FONTSIZE_TICK)
    ax_bar.legend(fontsize=7, loc='best', framealpha=0.8, edgecolor=COLORS['grid'])
    ax_bar.axhline(y=0, color=COLORS['text_secondary'], linewidth=0.5, linestyle='--')

    # ── Right: Heatmap — best method for each noise_type × snr_db ──
    snr_levels = sorted(metrics_df['snr_db'].unique())
    # Build matrix: rows=noise_types, cols=snr_levels, values=index of best method
    best_matrix = np.zeros((n_noise, len(snr_levels)))
    best_method_matrix = np.empty((n_noise, len(snr_levels)), dtype=object)

    for i, nt in enumerate(noise_types):
        for j, snr in enumerate(snr_levels):
            subset = metrics_df[(metrics_df['noise_type'] == nt) & (metrics_df['snr_db'] == snr)]
            if len(subset) > 0:
                best_idx = subset['snr_improvement'].idxmax()
                best_row = subset.loc[best_idx]
                best_matrix[i, j] = list(methods).index(best_row['method']) if best_row['method'] in methods else -1
                best_method_matrix[i, j] = best_row['method']
            else:
                best_matrix[i, j] = -1
                best_method_matrix[i, j] = 'N/A'

    # Draw heatmap as colored cells with method labels
    cmap = plt.cm.Set2
    ax_heat.imshow(best_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=max(n_methods - 1, 1))

    # Add text labels
    for i in range(n_noise):
        for j in range(len(snr_levels)):
            method_label = best_method_matrix[i, j][:3].upper() if best_method_matrix[i, j] != 'N/A' else 'N/A'
            ax_heat.text(j, i, method_label, ha='center', va='center',
                         fontsize=7, color=COLORS['text'], fontweight='bold')

    _apply_ax_style(ax_heat, title='Best Method per Noise Type × SNR', xlabel='SNR (dB)', ylabel='Noise Type')
    ax_heat.set_xticks(np.arange(len(snr_levels)))
    ax_heat.set_xticklabels([str(s) for s in snr_levels], fontsize=FONTSIZE_TICK)
    ax_heat.set_yticks(np.arange(n_noise))
    ax_heat.set_yticklabels([nt.replace('_', '\n') for nt in noise_types], fontsize=FONTSIZE_TICK)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close(fig)


def plot_singular_values(signal, L=500, save_path=None):
    """
    Plot singular value decay (scree plot) from Hankel matrix SVD.

    Constructs a Hankel matrix of shape (N-L+1, L) from the input signal,
    computes SVD, and plots the singular values on a log scale with the
    estimated rank (elbow point) marked.

    Parameters
    ----------
    signal : np.ndarray
        Input signal of shape (N,).
    L : int, optional
        Embedding dimension for the Hankel matrix. Default is 500.
    save_path : str or None, optional
        File path to save the PNG figure. If None, the figure is not saved.

    Returns
    -------
    np.ndarray
        Array of singular values.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    N = len(signal)
    K = N - L + 1
    H = hankel(signal[:L], signal[L - 1:])
    U, s, Vt = svd(H, full_matrices=False)

    # Estimate rank via elbow detection (90% cumulative energy)
    total_energy = np.sum(s ** 2)
    cumulative = np.cumsum(s ** 2) / total_energy
    estimated_rank = int(np.searchsorted(cumulative, 0.90)) + 1
    estimated_rank = max(1, min(estimated_rank, len(s) - 1))

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')

    indices = np.arange(1, len(s) + 1)
    ax.semilogy(indices, s, color=COLORS['accent'], linewidth=1.2, label='Singular values')
    ax.axvline(x=estimated_rank, color=COLORS['noisy'], linewidth=1.0, linestyle='--',
               label=f'Estimated rank = {estimated_rank}')
    ax.plot(estimated_rank, s[estimated_rank - 1], 'o', color=COLORS['noisy'], markersize=6, zorder=5)

    _apply_ax_style(ax, title='Singular Value Decay (Scree Plot)',
                    xlabel='Singular Value Index', ylabel='Singular Value (log scale)')
    ax.legend(fontsize=8, loc='upper right', framealpha=0.8, edgecolor=COLORS['grid'])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close(fig)

    return s
