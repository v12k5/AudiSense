import numpy as np
import matplotlib.pyplot as plt
import os

def load_stft_data(signal_name, base_folder='matrix'):
    """
    Load the STFT matrix and fs from the .npz file.
    
    Parameters:
    - signal_name (str): Name of the input signal for sub-folder.
    - base_folder (str): Base folder for matrices (default: 'matrix').
    
    Returns:
    - stft_matrix (np.ndarray): Complex STFT matrix (freq x time).
    - fs (int): Sampling rate.
    """
    sub_folder = os.path.join(base_folder, signal_name)
    input_path = os.path.join(sub_folder, 'stft_data.npz')
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"STFT data not found at {input_path}")
    data = np.load(input_path)
    stft_matrix = data['stft_matrix']
    fs = data['fs']
    data.close()
    return stft_matrix, fs

def adaptive_thresholding(stft_matrix, T1=0.3):
    """
    Apply adaptive hard thresholding on STFT coefficients per frequency bin.
    
    Parameters:
    - stft_matrix (np.ndarray): Complex STFT matrix of shape (freq_bins, time_frames).
    - T1 (float): Threshold parameter for normalized coefficients (default: 2.0).
    
    Returns:
    - thresholded_matrix (np.ndarray): Sparsified STFT matrix (same shape).
    """
    # Shape: (K_freq, M_time)
    K, M = stft_matrix.shape
    
    # Compute mean and std per frequency bin (across time frames, axis=1)
    mu = np.mean(stft_matrix, axis=1, keepdims=True)  # Shape (K, 1)
    # For std, use std of the complex values
    sigma = np.std(stft_matrix, axis=1, keepdims=True)
    sigma = np.maximum(sigma, 1e-10)  # Avoid division by zero
    
    # Normalize: X_n = (X - mu) / sigma
    X_n = (stft_matrix - mu) / sigma
    
    # Hard thresholding based on |X_n|
    thresholded_matrix = np.zeros_like(stft_matrix)
    mask = np.abs(X_n) >= T1
    thresholded_matrix[mask] = stft_matrix[mask]
    
    return thresholded_matrix

def plot_comparison(stft_matrix, thresholded_matrix, signal_name, fs, T1=0.3):
    """
    Plot side-by-side magnitude spectrograms of original and thresholded STFT in dB scale.
    
    Parameters:
    - stft_matrix (np.ndarray): Original STFT.
    - thresholded_matrix (np.ndarray): Thresholded STFT.
    - signal_name (str): Signal name for plot title/save.
    - fs (int): Sampling rate (for freq axis).
    - T1 (float): Threshold value for title.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original magnitude in dB
    mag_orig = np.abs(stft_matrix)
    im1 = ax1.imshow(20 * np.log10(mag_orig + 1e-10), aspect='auto', origin='lower',
                     extent=[0, mag_orig.shape[1] / fs, 0, fs / 2], cmap='viridis')
    ax1.set_title(f'Original STFT Magnitude - {signal_name}')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=ax1, label='Magnitude (dB)')
    
    # Thresholded magnitude in dB
    mag_thresh = np.abs(thresholded_matrix)
    im2 = ax2.imshow(20 * np.log10(mag_thresh + 1e-10), aspect='auto', origin='lower',
                     extent=[0, mag_thresh.shape[1] / fs, 0, fs / 2], cmap='viridis')
    ax2.set_title(f'Thresholded STFT Magnitude (T1={T1}) - {signal_name}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    plt.colorbar(im2, ax=ax2, label='Magnitude (dB)')
    
    plt.tight_layout()
    plot_path = os.path.join('matrix', signal_name, f'thresholded_spectrogram_{signal_name}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Spectrogram comparison saved to {plot_path}")
    plt.show()

def save_thresholded_data(thresholded_matrix, signal_name, base_folder='matrix'):
    """
    Save thresholded STFT matrix to a .npz file in the sub-folder.
    
    Parameters:
    - thresholded_matrix (np.ndarray): Thresholded STFT matrix.
    - signal_name (str): Name of the input signal for sub-folder.
    - base_folder (str): Base folder for matrices (default: 'matrix').
    """
    sub_folder = os.path.join(base_folder, signal_name)
    output_path = os.path.join(sub_folder, 'thresholded_data.npz')
    np.savez(output_path, thresholded_stft=thresholded_matrix)
    print(f"Thresholded data saved to {output_path}")

# Main execution
if __name__ == "__main__":
    signal_name = 'p226_001'  # Hardcoded for now
    T1 = 0.3  # Tunable threshold parameter
    
    stft_matrix, fs = load_stft_data(signal_name)
    print(f"Loaded STFT matrix with shape: {stft_matrix.shape}, fs: {fs}")
    
    thresholded_matrix = adaptive_thresholding(stft_matrix, T1=T1)
    # Sparsity: percentage of zero-magnitude coefficients
    sparsity = np.sum(np.abs(thresholded_matrix) < 1e-10) / thresholded_matrix.size * 100
    print(f"Thresholded matrix sparsity: {sparsity:.1f}% near-zero coefficients")
    
    plot_comparison(stft_matrix, thresholded_matrix, signal_name, fs, T1=T1)
    
    save_thresholded_data(thresholded_matrix, signal_name)