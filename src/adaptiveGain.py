import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import get_window

# =========================================================
# USER PARAMETERS
# =========================================================
INPUT_WAV = "data\\sp01_train_sn5.wav"   # <-- change path to your noisy wav
OUTPUT_WAV = "output\\enhanced.wav"              # output file for enhanced speech

frame_ms = 25        # frame length in ms
hop_ms = 10          # hop length in ms (overlap = frame_ms - hop_ms)
nfft = 512           # FFT size
window_type = "hamming"

alpha = 0.98         # smoothing factor for decision-directed method
G_min = 0.01         # minimum gain floor (to avoid musical noise)
eps = 1e-10          # small constant to prevent divide-by-zero

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def next_pow2(x):
    """Return the next power of 2 >= x (used for FFT)."""
    return 1 << (x - 1).bit_length()

def frame_signal(sig, frame_size, hop_size):
    """Slice signal into overlapping frames."""
    num_frames = 1 + int(np.floor((len(sig) - frame_size) / hop_size))
    frames = np.zeros((num_frames, frame_size), dtype=sig.dtype)
    for i in range(num_frames):
        start = i * hop_size
        frames[i] = sig[start:start + frame_size]
    return frames

def compute_stft(frames, window, nfft):
    """Apply windowing and compute STFT using rFFT (only positive freqs)."""
    windowed = frames * window[None, :]
    stft = np.fft.rfft(windowed, n=nfft, axis=1)
    return stft

def reconstruct_istft(stft_matrix, frame_len, hop, window, nfft, out_len=None):
    """
    Inverse STFT with overlap-add method.
    Uses window-squared normalization to avoid amplitude distortion.
    """
    num_frames = stft_matrix.shape[0]
    # Convert each spectrum back to time-domain frames
    time_frames = np.fft.irfft(stft_matrix, n=nfft, axis=1)[:, :frame_len]

    if out_len is None:
        out_len = (num_frames - 1) * hop + frame_len

    out = np.zeros(out_len, dtype=np.float64)
    win_sum = np.zeros(out_len, dtype=np.float64)

    for m in range(num_frames):
        start = m * hop
        out[start:start + frame_len] += time_frames[m] * window
        win_sum[start:start + frame_len] += window**2

    # Normalize by window energy (wherever > 0)
    nz = win_sum > eps
    out[nz] = out[nz] / win_sum[nz]

    return out

def plot_spectrogram(stft_matrix, fs, nfft, hop, title="Spectrogram (dB)"):
    """Visualize spectrogram in dB scale."""
    mag = np.abs(stft_matrix).T
    mag_db = 20 * np.log10(mag + 1e-12)
    time_len = stft_matrix.shape[0] * hop / fs
    plt.imshow(mag_db, origin='lower', aspect='auto',
               extent=[0, time_len, 0, fs/2], vmin=-80, vmax=0, cmap='inferno')
    plt.colorbar(label="Magnitude (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)

# =========================================================
# NOISE PSD ESTIMATION (VAD based)
# =========================================================
def energy_vad(frames, window, threshold_scale=0.5):
    """
    Simple VAD: compute frame energy, compare with threshold.
    Returns: boolean mask (speech=True, noise=False).
    """
    win_frames = frames * window[None, :]
    energy = np.sum(win_frames**2, axis=1) / frames.shape[1]
    median = np.median(energy)
    std = np.std(energy)
    thresh = median + threshold_scale * std
    is_speech = energy > thresh
    return is_speech, energy, thresh

def estimate_noise_psd(stft_matrix, is_speech):
    """Average noise power spectrum over frames classified as noise by VAD."""
    power = np.abs(stft_matrix)**2
    noise_frames = ~is_speech
    if np.sum(noise_frames) >= 1:
        noise_psd = np.mean(power[noise_frames, :], axis=0)
    else:
        noise_psd = np.min(power, axis=0)
    noise_psd = np.maximum(noise_psd, eps)
    return noise_psd

# =========================================================
# WIENER FILTER (Decision-Directed)
# =========================================================
def wiener_filter(stft_matrix, noise_psd, alpha=0.98, G_min=0.01):
    """
    Apply Wiener filter with decision-directed a priori SNR.
    """
    num_frames, nbin = stft_matrix.shape
    enhanced = np.zeros_like(stft_matrix, dtype=np.complex64)

    # Buffers for previous frame
    S_prev = np.zeros(nbin, dtype=np.complex64)

    for m in range(num_frames):
        X = stft_matrix[m]
        power_X = np.abs(X)**2

        # A posteriori SNR
        gamma = power_X / (noise_psd + eps)

        # A priori SNR (decision-directed)
        xi = alpha * (np.abs(S_prev)**2) / (noise_psd + eps) + (1 - alpha) * np.maximum(gamma - 1.0, 0.0)

        # Wiener gain
        G = xi / (1.0 + xi)
        G = np.maximum(G, G_min)

        # Apply gain
        S = G * X
        enhanced[m] = S

        # Save for next frame
        S_prev = S

    return enhanced

# =========================================================
# MAIN PROCESSING
# =========================================================
if not os.path.exists(INPUT_WAV):
    raise FileNotFoundError(f"File not found: {INPUT_WAV}")

# 1) Load audio
fs, signal = wavfile.read(INPUT_WAV)
if signal.ndim == 2:  # stereo -> mono
    signal = signal.mean(axis=1)
signal = signal.astype(np.float64)
signal = signal / (np.max(np.abs(signal)) + eps)

# 2) Frame the signal
frame_len = int(round(frame_ms/1000.0 * fs))
hop = int(round(hop_ms/1000.0 * fs))
if nfft < frame_len:
    nfft = next_pow2(frame_len)

frames = frame_signal(signal, frame_len, hop)
window = get_window(window_type, frame_len)

# 3) STFT
stft_matrix = compute_stft(frames, window, nfft)

# 4) VAD + noise PSD
is_speech, energies, vad_thresh = energy_vad(frames, window, threshold_scale=0.5)
noise_psd = estimate_noise_psd(stft_matrix, is_speech)

# 5) Wiener filtering
enhanced_stft = wiener_filter(stft_matrix, noise_psd, alpha=alpha, G_min=G_min)

# 6) Reconstruct enhanced signal
enhanced_signal = reconstruct_istft(enhanced_stft, frame_len, hop, window, nfft,
                                    out_len=(frames.shape[0]-1)*hop + frame_len)

# Align lengths of signals for plotting
min_len = min(len(signal), len(enhanced_signal))
signal = signal[:min_len]
enhanced_signal = enhanced_signal[:min_len]

# Normalize enhanced for saving
enhanced_signal = enhanced_signal / (np.max(np.abs(enhanced_signal)) + eps)
scaled_int16 = (enhanced_signal * 32767).astype(np.int16)
wavfile.write(OUTPUT_WAV, fs, scaled_int16)

# =========================================================
# VISUALIZATION
# =========================================================
plt.figure(figsize=(14,9))

# Waveforms
t = np.arange(min_len) / fs
plt.subplot(3,1,1)
plt.plot(t, signal, label="Noisy (normalized)", alpha=0.7)
plt.plot(t, enhanced_signal, label="Enhanced (normalized)", alpha=0.7)
plt.legend()
plt.title("Waveforms")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Spectrograms
plt.subplot(3,2,3)
plot_spectrogram(stft_matrix, fs, nfft, hop, title="Noisy Spectrogram (dB)")
plt.subplot(3,2,4)
plot_spectrogram(enhanced_stft, fs, nfft, hop, title="Enhanced Spectrogram (dB)")

# VAD energies
plt.subplot(3,1,3)
time_frames = np.arange(frames.shape[0]) * hop / fs
plt.plot(time_frames, energies, label="Frame energy")
plt.hlines(vad_thresh, time_frames[0], time_frames[-1], colors='r', linestyles='--', label=f"VAD threshold")
plt.scatter(time_frames[~is_speech], energies[~is_speech], color='green', s=10, label="Noise frames")
plt.scatter(time_frames[is_speech], energies[is_speech], color='black', s=10, label="Speech frames")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Energy")
plt.title("VAD Decision")

plt.tight_layout()
plt.show()

print(f"✅ Enhanced audio saved to: {OUTPUT_WAV}")
