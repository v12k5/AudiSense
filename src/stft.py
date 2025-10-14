import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
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

def compute_stft(signal, fs, frame_length_ms=25, overlap_percent=50):
    """
    Compute STFT with Hanning window and specified parameters.
    
    Parameters:
    - signal (np.ndarray): Input time-domain signal.
    - fs (int): Sampling rate.
    - frame_length_ms (int): Frame length in milliseconds.
    - overlap_percent (int): Overlap percentage.
    
    Returns:
    - f (np.ndarray): Frequency bins.
    - t (np.ndarray): Time frames.
    - stft_matrix (np.ndarray): Complex STFT matrix (freq x time).
    """
    frame_length = int(frame_length_ms * fs / 1000)  # Samples per frame
    hop_length = int(frame_length * (1 - overlap_percent / 100))  # Hop size
    f, t, stft_matrix = stft(signal, fs=fs, window='hann', nperseg=frame_length,
                             noverlap=frame_length - hop_length, nfft=frame_length,
                             detrend=False, return_onesided=True, boundary='zeros',
                             padded=True)
    return f, t, stft_matrix

def save_stft_data(stft_matrix, fs, signal_name, base_folder='matrix'):
    """
    Save STFT matrix and sampling rate to a .npz file in a structured folder.
    
    Parameters:
    - stft_matrix (np.ndarray): STFT matrix.
    - fs (int): Sampling rate.
    - signal_name (str): Name of the input signal (without extension) for sub-folder.
    - base_folder (str): Base folder for matrices (default: 'matrix').
    """
    sub_folder = os.path.join(base_folder, signal_name)
    os.makedirs(sub_folder, exist_ok=True)
    output_path = os.path.join(sub_folder, 'stft_data.npz')
    np.savez(output_path, stft_matrix=stft_matrix, fs=fs)

# Main execution
if __name__ == "__main__":
    data_folder = 'data'
    file_name = 'sp01_train_sn5.wav'
    signal_name = os.path.splitext(file_name)[0]  # Extract name without extension
    file_path = os.path.join(data_folder, file_name)
    
    fs, signal = load_audio(file_path)
    print(f"Loaded audio with sampling rate: {fs} Hz, length: {len(signal)} samples")
    
    f, t, stft_matrix = compute_stft(signal, fs)
    print(f"STFT shape: {stft_matrix.shape} (freq bins: {len(f)}, time frames: {len(t)})")
    
    save_stft_data(stft_matrix, fs, signal_name)
    print(f"STFT data saved to 'matrix/{signal_name}/stft_data.npz'")