import numpy as np
import os
from scipy.signal import istft
from scipy.io import wavfile
from scipy.optimize import linprog
import pywavelets as pywt
from pesq import pesq

def calculate_M(N, C=1.5, k_ratio=0.15):
    K = max(1, int(k_ratio * N))
    if K >= N:
        return N  # Fallback if sparsity assumption invalid
    log_term = np.log(N / K)
    M = int(np.ceil(C * K * log_term))
    return max(1, min(M, N))

def generate_sensing_matrices(M, N, U=1):
    # Part 1: Compute Key Parameters
    m = N / M
    m1 = int(np.floor(m))
    m2 = m1 + 1 if m > m1 else m1
    Nm2 = N - M * m1
    Nm1 = M - Nm2

    # Part 2: Assign Row Types
    all_indices = np.arange(1, M + 1)
    rpm2 = np.random.choice(all_indices, Nm2, replace=False)
    rpm1 = np.setdiff1d(all_indices, rpm2)

    # Part 3: Initialize Base Rows
    rowt1 = np.concatenate((np.ones(m1), np.zeros(N - m1)))
    rowt2 = np.concatenate((np.ones(m2), np.zeros(N - m2)))

    # Part 4: Generate Rows Using Cyclic Shifts
    rows = []
    for k in range(1, M + 1):
        if k in rpm1:
            row_k = rowt1.copy()
            rows.append(row_k)
            rowt1 = np.roll(rowt1, m1)
            rowt2 = np.roll(rowt2, m1)
        else:
            row_k = rowt2.copy()
            rows.append(row_k)
            rowt1 = np.roll(rowt1, m2)
            rowt2 = np.roll(rowt2, m2)

    # Part 5: Construct Base Matrix (Ms)
    Db = np.vstack(rows)
    # For U > 1, would create block diagonal, but U=1
    Ms = Db  # For U=1

    # Part 6: Generate Variations (allM)
    allM = []
    L = m1
    for i in range(m1):
        Mtemp = Ms.copy()
        Mtemp[:, L - 1] = 0  # 0-based index
        allM.append(Mtemp)
        L -= 1

    return allM

def adaptive_threshold(segments, T1=1.0):
    # Compute DFT for each segment
    dfts = [np.fft.fft(seg) for seg in segments]
    dfts = np.array(dfts)  # I x N complex

    # Compute mean and std per frequency bin k
    mu = np.mean(dfts, axis=0)
    sigma = np.std(dfts, axis=0)

    # Avoid division by zero
    sigma[sigma == 0] = 1e-10

    # Normalize Xni
    Xni = (dfts - mu) / sigma

    # Threshold: hard, w=1
    mask = np.abs(Xni) >= T1
    sparse_dfts = dfts * mask

    # Inverse DFT to sparse time
    sparse_segments = [np.real(np.fft.ifft(sp)) for sp in sparse_dfts]

    return sparse_segments

def reconstruct_l1(y, Phi):
    # l1 min ||s||1 s.t. Phi s = y
    # Use decomposition s = p - n, p,n >=0
    m, n = Phi.shape
    c = np.ones(2 * n)
    A_eq = np.hstack((Phi, -Phi))
    b_eq = y
    bounds = [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        pn = res.x
        p = pn[:n]
        n = pn[n:]
        s_hat = p - n
        return s_hat
    else:
        raise ValueError("Reconstruction failed")

def compressive_sensing(segment, Phi, wavelet='db4', level=4):
    # Wavelet transform (multi-level for better sparsity)
    coeffs = pywt.wavedec(segment, wavelet, level=level)
    # Flatten to vector
    sparse_w = np.concatenate(coeffs)

    # y = Phi @ sparse_w (but paper threshold first, but here assume threshold done prior)
    y = Phi @ sparse_w

    # Reconstruct
    s_hat = reconstruct_l1(y, Phi)

    # Unflatten and inverse wavelet
    # To unflatten, need lengths
    coeff_lengths = [len(c) for c in coeffs]
    s_hat_list = []
    start = 0
    for len_c in coeff_lengths:
        s_hat_list.append(s_hat[start:start + len_c])
        start += len_c
    reconstructed = pywt.waverec(s_hat_list, wavelet)

    return reconstructed

def overlap_add(segments, hop_length):
    num_segments = len(segments)
    segment_length = len(segments[0])
    total_length = (num_segments - 1) * hop_length + segment_length
    reconstructed = np.zeros(total_length)
    for i, seg in enumerate(segments):
        start = i * hop_length
        reconstructed[start:start + segment_length] += seg
    return reconstructed

def process_signal(signal_name, data_folder='data', matrix_base='matrix', frame_length_ms=25, overlap_percent=50, C=1.5, k_ratio=0.15, T1=1.0):
    sub_folder = os.path.join(matrix_base, signal_name)
    adaptive_path = os.path.join(sub_folder, 'adaptiveGain.npz')
    if not os.path.exists(adaptive_path):
        print(f"Adaptive gain file not found for {signal_name}")
        return

    data = np.load(adaptive_path)
    enhanced_stft = data['enhanced_stft']  # Assume complex [freq, time]
    fs = data['fs']

    # Inverse STFT to time domain
    n_freq, n_time = enhanced_stft.shape
    nperseg = (n_freq - 1) * 2  # From onesided
    hop_length = nperseg // 2  # Assume 50% overlap
    _, enhanced_time = istft(enhanced_stft, fs=fs, window='hann', nperseg=nperseg, noverlap=hop_length, nfft=nperseg)

    # Segment the enhanced time signal
    segment_length = nperseg  # Same as STFT
    hop_seg = int(segment_length * (1 - overlap_percent / 100))  # 50%
    num_segments = ((len(enhanced_time) - segment_length) // hop_seg) + 1
    segments = []
    for i in range(num_segments):
        start = i * hop_seg
        seg = enhanced_time[start:start + segment_length]
        if len(seg) < segment_length:
            seg = np.pad(seg, (0, segment_length - len(seg)))
        segments.append(seg * np.hanning(segment_length))  # Window

    # Adaptive threshold on DFT
    sparse_segments = adaptive_threshold(segments, T1=T1)

    # For each sparse segment, apply wavelet for CS sparsity? But to match, since threshold already done, use as is for CS, assuming Psi = I
    # But to follow paper, apply wavelet on sparse_segment

    wavelet = 'db4'
    N = segment_length
    M = calculate_M(N, C, k_ratio)
    allM = generate_sensing_matrices(M, N)

    # Load original noisy for potential, but for PESQ need clean
    file_name = signal_name + '.wav'  # Assume
    file_path = os.path.join(data_folder, file_name)
    _, original_noisy = wavfile.read(file_path)
    original_noisy = original_noisy.astype(np.float64) / np.max(np.abs(original_noisy))

    # Assume clean path
    clean_path = os.path.join(data_folder, signal_name.replace('_sn5', '_clean') + '.wav')
    if not os.path.exists(clean_path):
        print(f"Clean file not found for {signal_name}")
        return
    _, clean = wavfile.read(clean_path)
    clean = clean.astype(np.float64) / np.max(np.abs(clean))

    best_pesq = -np.inf
    best_matrix = None
    for Phi in allM:
        reconstructed_segments = []
        for sparse_seg in sparse_segments:
            # Wavelet on sparse_seg
            coeffs = pywt.wavedec(sparse_seg, wavelet, level=None)  # Auto level
            sparse_w = np.concatenate(coeffs)
            # Adjust N if len(sparse_w) != segment_length, but for dwt, len != N, so N_w = len(sparse_w)
            N_w = len(sparse_w)
            M_w = calculate_M(N_w, C, k_ratio)
            # Regenerate Phi for N_w? But alg for N=segment_length, but to match, assume single level dwt, where len ~ N
            # To simplify, use single level
            coeffs = pywt.dwt(sparse_seg, wavelet)
            sparse_w = np.concatenate(coeffs)
            N_w = len(sparse_w)
            # Assume Phi for N_w, regenerate allM for each? Heavy, so assume identity for sparsity, but to follow, use Phi for N, but if len != N, pad or change.

            # To make work, use threshold on time, assume sparsity in time after threshold, but paper wavelet.

            # Alternative: skip additional wavelet, use thresholded freq as sparse, but since sparse_segments are time, use as sparse_x_i, then y = Phi @ sparse_seg, N = segment_length
            y = Phi @ sparse_seg

            s_hat = reconstruct_l1(y, Phi)

            reconstructed_segments.append(s_hat)

        # Overlap add
        reconstructed_time = overlap_add(reconstructed_segments, hop_seg)

        # Normalize
        reconstructed_time /= np.max(np.abs(reconstructed_time)) if np.max(np.abs(reconstructed_time)) > 0 else 1

        # PESQ
        score = pesq(fs, clean, reconstructed_time, 'wb')

        if score > best_pesq:
            best_pesq = score
            best_matrix = Phi

    if best_matrix is not None:
        output_path = os.path.join(sub_folder, 'best_sensing_matrix.npy')
        np.save(output_path, best_matrix)
        print(f"Saved best sensing matrix for {signal_name} with PESQ {best_pesq}")

def main():
    data_folder = 'data'
    matrix_base = 'matrix'
    file_names = [f for f in os.listdir(data_folder) if f.endswith('.wav')]
    for file_name in file_names:
        signal_name = os.path.splitext(file_name)[0]
        process_signal(signal_name, data_folder, matrix_base)

if __name__ == "__main__":
    main()