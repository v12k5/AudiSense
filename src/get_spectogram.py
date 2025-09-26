import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import get_window

# ==============================
# Step 1: Load the audio signal
# ==============================
# Replace with your audio file path
fs, signal = wavfile.read("data\\sp01_train_sn5.wav")  

# If stereo, convert to mono by averaging channels
if signal.ndim == 2:
    signal = signal.mean(axis=1)

# Normalize to [-1, 1] if needed
signal = signal / np.max(np.abs(signal))

# ==============================
# Step 2: Define STFT parameters
# ==============================
frame_size = int(0.025 * fs)   # 25 ms frame
hop_size   = int(0.010 * fs)   # 10 ms hop (overlap)
window     = get_window("hamming", frame_size)

# Number of FFT points (can be power of 2 >= frame_size)
nfft = 512

# ==============================
# Step 3: Frame the signal
# ==============================
def frame_signal(sig, frame_size, hop_size):
    """Split signal into overlapping frames"""
    num_frames = 1 + int((len(sig) - frame_size) / hop_size)
    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        start = i * hop_size
        frames[i] = sig[start:start + frame_size]
    return frames

frames = frame_signal(signal, frame_size, hop_size)

# ==============================
# Step 4: Apply window + FFT
# ==============================
def compute_stft(frames, window, nfft):
    """Apply window and compute FFT for each frame"""
    stft_matrix = np.zeros((len(frames), nfft), dtype=np.complex64)
    for i, frame in enumerate(frames):
        windowed = frame * window               # (Step 2: windowing)
        spectrum = np.fft.fft(windowed, nfft)   # (Step 3: FFT)
        stft_matrix[i] = spectrum
    return stft_matrix

stft_matrix = compute_stft(frames, window, nfft)

# ==============================
# Step 5: Get magnitude & spectrogram
# ==============================
magnitude_spectrogram = np.abs(stft_matrix.T)   # shape: (freq_bins, time_frames)

# Convert to dB scale for visualization
spectrogram_db = 20 * np.log10(magnitude_spectrogram + 1e-10)

# ==============================
# Visualization
# ==============================
plt.figure(figsize=(10, 6))
plt.imshow(spectrogram_db[:nfft//2], 
           origin="lower", 
           aspect="auto", 
           cmap="inferno", 
           extent=[0, len(frames)*hop_size/fs, 0, fs/2])
plt.colorbar(label="Magnitude (dB)")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.title("Spectrogram using STFT (Step-by-step)")
plt.show()