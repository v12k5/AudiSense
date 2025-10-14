import numpy as np
import matplotlib.pyplot as plt
import os

def load_stft_data(signal_name, base_folder='matrix', file_name='stft_data.npz'):
    """
    Load STFT matrix and sampling rate from .npz file in structured folder.
    
    Parameters:
    - signal_name (str): Name of the input signal (without extension) for sub-folder.
    - base_folder (str): Base folder for matrices (default: 'matrix').
    - file_name (str): Name of the .npz file (default: 'stft_data.npz').
    
    Returns:
    - stft_matrix (np.ndarray): Complex STFT matrix.
    - fs (int): Sampling rate.
    """
    sub_folder = os.path.join(base_folder, signal_name)
    input_path = os.path.join(sub_folder, file_name)
    data = np.load(input_path)
    return data['stft_matrix'], data['fs']

def estimate_noise_psd(magnitude, vad_threshold_factor=1.5, min_frames_for_noise=10):
    """
    Estimate noise PSD using energy-based VAD.
    - Compute frame energies.
    - Adaptive threshold: mean energy * factor.
    - Non-speech frames (energy < threshold) used to average noise PSD.
    
    Parameters:
    - magnitude (np.ndarray): Magnitude spectrogram (|STFT|).
    - vad_threshold_factor (float): Multiplier for adaptive energy threshold.
    - min_frames_for_noise (int): Minimum non-speech frames required.
    
    Returns:
    - noise_psd (np.ndarray): Estimated noise PSD (freq x 1).
    """
    energies = np.mean(magnitude**2, axis=0)  # Frame energies
    threshold = np.mean(energies) * vad_threshold_factor
    non_speech_mask = energies < threshold
    if np.sum(non_speech_mask) < min_frames_for_noise:
        raise ValueError("Insufficient non-speech frames for noise estimation.")
    noise_frames = magnitude[:, non_speech_mask]
    noise_psd = np.mean(noise_frames**2, axis=1, keepdims=True)  # Average PSD
    return noise_psd

def compute_snrs(magnitude_squared, noise_psd, beta=0.98, snr_min_db=-18):
    """
    Compute a posteriori and a priori SNRs.
    
    Parameters:
    - magnitude_squared (np.ndarray): |X|^2 (freq x time).
    - noise_psd (np.ndarray): Noise PSD (freq x 1).
    - beta (float): Smoothing factor for a priori SNR.
    - snr_min_db (float): Minimum SNR floor in dB.
    
    Returns:
    - snr_pri (np.ndarray): A priori SNR (freq x time).
    """
    snr_post = magnitude_squared / noise_psd  # Eq 5
    snr_post = np.maximum(snr_post, 10**(snr_min_db / 10))  # Floor
    
    num_frames = magnitude_squared.shape[1]
    snr_pri = np.zeros_like(snr_post)
    s_est_prev = np.zeros(magnitude_squared.shape[0])  # Initial previous estimate
    
    for m in range(num_frames):
        snr_pri_temp = np.maximum(snr_post[:, m] - 1, 0)  # P[snr_post - 1]
        snr_pri[:, m] = beta * (s_est_prev**2 / noise_psd[:, 0]) + (1 - beta) * snr_pri_temp
        s_est_prev = np.sqrt(magnitude_squared[:, m]) * (snr_pri[:, m] / (1 + snr_pri[:, m]))  # Update for next
    
    return snr_pri

def apply_gain(stft_matrix, snr_pri):
    """
    Compute and apply Wiener gain to get enhanced STFT.
    
    Parameters:
    - stft_matrix (np.ndarray): Original complex STFT.
    - snr_pri (np.ndarray): A priori SNR.
    
    Returns:
    - enhanced_stft (np.ndarray): Enhanced complex STFT (S_est).
    """
    gain = snr_pri / (1 + snr_pri)  # Eq 7
    enhanced_stft = stft_matrix * gain  # Eq 8 (element-wise)
    return enhanced_stft

def save_enhanced_stft_data(enhanced_stft, fs, signal_name, base_folder='matrix'):
    """
    Save enhanced STFT matrix and sampling rate to a .npz file in structured folder.
    
    Parameters:
    - enhanced_stft (np.ndarray): Enhanced STFT matrix.
    - fs (int): Sampling rate.
    - signal_name (str): Name of the input signal (without extension) for sub-folder.
    - base_folder (str): Base folder for matrices (default: 'matrix').
    """
    sub_folder = os.path.join(base_folder, signal_name)
    output_path = os.path.join(sub_folder, 'enhanced_stft_data.npz')
    np.savez(output_path, enhanced_stft=enhanced_stft, fs=fs)

def plot_spectrograms(mag_noisy, mag_enhanced, fs):
    """
    Plot magnitude spectrograms for original and enhanced.
    
    Parameters:
    - mag_noisy (np.ndarray): |X| (freq x time).
    - mag_enhanced (np.ndarray): |S_est| (freq x time).
    - fs (int): Sampling rate for axis labeling.
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original noisy
    im0 = axs[0].imshow(20 * np.log10(mag_noisy + 1e-10), aspect='auto', origin='lower',
                        extent=[0, mag_noisy.shape[1] / fs, 0, fs / 2], cmap='viridis')
    axs[0].set_title('Original Noisy Spectrogram')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Frequency (Hz)')
    plt.colorbar(im0, ax=axs[0], label='dB')
    
    # Enhanced
    im1 = axs[1].imshow(20 * np.log10(mag_enhanced + 1e-10), aspect='auto', origin='lower',
                        extent=[0, mag_enhanced.shape[1] / fs, 0, fs / 2], cmap='viridis')
    axs[1].set_title('Enhanced Spectrogram')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im1, ax=axs[1], label='dB')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    signal_name = 'sp01_train_sn5'  # Name without extension, matching Code 1
    stft_matrix, fs = load_stft_data(signal_name)
    print(f"Loaded STFT with shape: {stft_matrix.shape}, fs: {fs} Hz")
    
    magnitude = np.abs(stft_matrix)
    magnitude_squared = magnitude**2
    
    noise_psd = estimate_noise_psd(magnitude)
    print("Noise PSD estimated")
    
    snr_pri = compute_snrs(magnitude_squared, noise_psd)
    print("SNRs computed")
    
    enhanced_stft = apply_gain(stft_matrix, snr_pri)
    enhanced_magnitude = np.abs(enhanced_stft)
    print("Gain applied")
    
    save_enhanced_stft_data(enhanced_stft, fs, signal_name)
    print(f"Enhanced STFT data saved to 'matrix/{signal_name}/adaptiveGain.npz'")
    
    plot_spectrograms(magnitude, enhanced_magnitude, fs)