import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import matplotlib.pyplot as plt
import os

def load_audio(file_path):
    """
    Load the WAV file and extract sampling rate and signal.
    
    Parameters:
    - file_path (str): Path to the WAV file.
    
    Returns:
    - fs (int): Sampling rate.
    - signal (np.ndarray): Audio signal (mono, float64 normalized).
    """
    fs, signal = wavfile.read(file_path)
    if signal.ndim > 1:
        signal = signal[:, 0]  # Take first channel if stereo
    signal = signal.astype(np.float64) / np.max(np.abs(signal))  # Normalize to [-1, 1]
    return fs, signal

def next_power_of_2(n):
    """
    Find the smallest power of 2 greater than or equal to n.
    
    Parameters:
    - n (int): Input value.
    
    Returns:
    - int: Next power of 2.
    """
    return 2 ** int(np.ceil(np.log2(n)))

def compute_power2_stft(signal, fs, frame_length_ms=25, overlap_percent=50):
    """
    Compute STFT with nperseg as next power of 2 >= frame_length_ms * fs / 1000,
    using Hanning window and specified overlap.
    
    Parameters:
    - signal (np.ndarray): Input time-domain signal.
    - fs (int): Sampling rate.
    - frame_length_ms (int): Target frame length in milliseconds.
    - overlap_percent (int): Overlap percentage.
    
    Returns:
    - f (np.ndarray): Frequency bins.
    - t (np.ndarray): Time frames.
    - stft_matrix (np.ndarray): Complex STFT matrix (freq x time).
    """
    target_length = int(frame_length_ms * fs / 1000)
    frame_length = next_power_of_2(target_length)  # Ensure power of 2
    hop_length = int(frame_length * (1 - overlap_percent / 100))
    f, t, stft_matrix = stft(signal, fs=fs, window='hann', nperseg=frame_length,
                             noverlap=frame_length - hop_length, nfft=frame_length,
                             detrend=False, return_onesided=True, boundary='zeros',
                             padded=True)
    return f, t, stft_matrix, frame_length  # Return frame_length for reference

def adaptive_thresholding(stft_matrix, T1=0.5):
    """
    Apply adaptive thresholding per frequency bin across time frames.
    - Compute mu and sigma per freq bin.
    - Normalize to Xni, threshold if |Xni| < T1 (hard, omega=1).
    
    Parameters:
    - stft_matrix (np.ndarray): Complex STFT (freq x time).
    - T1 (float): Normalized threshold (0 < T1 <= 1).
    
    Returns:
    - thresholded_stft (np.ndarray): Thresholded STFT matrix.
    """
    num_freq, num_time = stft_matrix.shape
    thresholded_stft = np.zeros_like(stft_matrix, dtype=complex)
    
    for freq in range(num_freq):
        X_k = stft_matrix[freq, :]  # Complex coefficients for this freq across time
        mu = np.mean(X_k)  # Complex mean (Eq 12)
        var = np.mean(np.abs(X_k)**2) - np.abs(mu)**2  # Variance (Eq 13 base)
        sigma = np.sqrt(np.maximum(var, 1e-10))  # Avoid division by zero
        
        Xni = (X_k - mu) / sigma  # Normalized (Eq 11)
        
        mask = np.abs(Xni) >= T1  # Threshold condition
        thresholded_stft[freq, :] = np.where(mask, X_k, 0)  # Hard threshold (omega=1)
    
    return thresholded_stft

def plot_spectrograms(f, t, mag_original, mag_thresholded):
    """
    Plot magnitude spectrograms for original and thresholded STFT.
    
    Parameters:
    - f (np.ndarray): Frequency bins.
    - t (np.ndarray): Time frames.
    - mag_original (np.ndarray): Original magnitude (freq x time).
    - mag_thresholded (np.ndarray): Thresholded magnitude (freq x time).
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original
    im0 = axs[0].imshow(20 * np.log10(mag_original + 1e-10), aspect='auto', origin='lower',
                        extent=[t[0], t[-1], f[0], f[-1]], cmap='viridis')
    axs[0].set_title('Original Noisy Spectrogram')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Frequency (Hz)')
    plt.colorbar(im0, ax=axs[0], label='Magnitude (dB)')
    
    # Thresholded
    im1 = axs[1].imshow(20 * np.log10(mag_thresholded + 1e-10), aspect='auto', origin='lower',
                        extent=[t[0], t[-1], f[0], f[-1]], cmap='viridis')
    axs[1].set_title('Thresholded Spectrogram (T1 = 0.5)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=axs[1], label='Magnitude (dB)')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    data_folder = 'data'
    file_name = 'sp01_train_sn5.wav'
    file_path = os.path.join(data_folder, file_name)
    
    # Re-compute STFT with power-of-2 adjustment for paper compliance
    fs, signal = load_audio(file_path)
    print(f"Loaded audio with sampling rate: {fs} Hz, length: {len(signal)} samples")
    
    f, t, stft_matrix, actual_frame_length = compute_power2_stft(signal, fs)
    print(f"Re-computed STFT with nperseg={actual_frame_length} (power of 2), shape: {stft_matrix.shape}")
    
    # Apply thresholding
    thresholded_stft = adaptive_thresholding(stft_matrix)
    
    # Compute magnitudes for plotting
    mag_original = np.abs(stft_matrix)
    mag_thresholded = np.abs(thresholded_stft)
    
    # Plot comparison
    plot_spectrograms(f, t, mag_original, mag_thresholded)
    print("Thresholding applied and spectrograms plotted.")