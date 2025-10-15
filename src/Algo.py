import numpy as np
import os
from scipy.signal import istft
from scipy.io import wavfile
from scipy.optimize import linprog
from pesq import pesq

def calculate_M(N, C=2.5, k_ratio=0.15):
    """
    Calculate number of measurements M using the formula M >= C * K * log(N / K).
    
    Parameters:
    - N (int): Signal length (e.g., number of frequency bins).
    - C (float): Constant factor (default: 1.5).
    - k_ratio (float): Sparsity ratio K/N (default: 0.15).
    
    Returns:
    - M (int): Number of measurements.
    """
    K = max(1, int(k_ratio * N))
    if K >= N:
        return N  # Fallback if sparsity assumption invalid
    log_term = np.log(N / K)
    M = int(np.ceil(C * K * log_term))
    return max(1, min(M, N))

def generate_sensing_matrices(M, N, U=1):
    """
    Generate array of sparse binary sensing matrices using Algorithm 1.
    
    Parameters:
    - M (int): Number of measurements (rows).
    - N (int): Signal length (columns).
    - U (int): Number of diagonal blocks (default: 1 for small N).
    
    Returns:
    - allM (list): List of M x N sensing matrices (variations).
    """
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
    # For U > 1, create block-diagonal; here U=1
    Ms = Db

    # Part 6: Generate Variations (allM)
    allM = []
    L = m1
    for i in range(m1):
        Mtemp = Ms.copy()
        Mtemp[:, L - 1] = 0  # 0-based index
        allM.append(Mtemp)
        L -= 1

    return allM

def reconstruct_l1(y, Phi):
    """
    Reconstruct sparse signal using ℓ1-norm minimization: min ||s||_1 s.t. Phi s = y.
    
    Parameters:
    - y (np.ndarray): Compressed measurements (length M).
    - Phi (np.ndarray): Sensing matrix (M x N).
    
    Returns:
    - s_hat (np.ndarray): Reconstructed sparse signal (length N).
    """
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

def process_signal(signal_name, data_folder='data', matrix_base='matrix', clean_folder='clean_train', C=1.5, k_ratio=0.15, bypass_cs=False):
    """
    ... (keep existing docstring)
    - bypass_cs (bool): If True, use identity matrix to bypass CS (for debugging).
    """
    sub_folder = os.path.join(matrix_base, signal_name)
    thresholded_path = os.path.join(sub_folder, 'thresholded_data.npz')
    stft_path = os.path.join(sub_folder, 'stft_data.npz')
    if not os.path.exists(thresholded_path):
        print(f"Thresholded data not found for {signal_name}")
        return

    # Load thresholded STFT (sparse, complex [freq_bins, time_frames])
    thresh_data = np.load(thresholded_path)
    thresholded_stft = thresh_data['thresholded_stft']
    thresh_data.close()

    # Load fs from stft_data.npz
    stft_data = np.load(stft_path)
    fs = stft_data['fs']
    stft_data.close()

    K, M_time = thresholded_stft.shape  # N = K (freq bins per frame)
    N = K
    if bypass_cs:
        print(f"Bypassing CS for {signal_name} - using identity matrix.")
        allM = [np.eye(N, dtype=np.float64)]  # Single identity matrix (no compression)
    else:
        M = calculate_M(N, C, k_ratio)
        allM = generate_sensing_matrices(M, N)

    # Load clean signal from clean_train folder
    clean_file_path = os.path.join(clean_folder, signal_name + '.wav')
    if not os.path.exists(clean_file_path):
        print(f"Clean file not found for {signal_name} at {clean_file_path}")
        return
    _, clean = wavfile.read(clean_file_path)
    clean = clean.astype(np.float64) / np.max(np.abs(clean))

    # Determine PESQ mode
    if fs == 8000:
        pesq_mode = 'nb'
    elif fs == 16000:
        pesq_mode = 'wb'
    else:
        raise ValueError(f"Unsupported sampling rate {fs} Hz for PESQ. Must be 8000 ('nb') or 16000 ('wb').")

    best_pesq = -np.inf
    best_matrix = None
    for idx, Phi in enumerate(allM):
        # Reconstruct STFT per time frame (complex handling: separate real/imag)
        reconstructed_stft = np.zeros_like(thresholded_stft, dtype=complex)
        for t in range(M_time):
            x_real = np.real(thresholded_stft[:, t])
            x_imag = np.imag(thresholded_stft[:, t])
            if bypass_cs:
                # No compression: y = x, s_hat = x
                s_hat_real = x_real
                s_hat_imag = x_imag
            else:
                y_real = Phi @ x_real
                y_imag = Phi @ x_imag
                s_hat_real = reconstruct_l1(y_real, Phi)
                s_hat_imag = reconstruct_l1(y_imag, Phi)
            reconstructed_stft[:, t] = s_hat_real + 1j * s_hat_imag

        # Inverse STFT to time domain
        nperseg = 2 * (K - 1) if K > 1 else 256  # Approximate
        hop_length = nperseg // 2  # 50% overlap
        _, reconstructed_time = istft(reconstructed_stft, fs=fs, window='hann',
                                      nperseg=nperseg, noverlap=hop_length, nfft=nperseg)

        # Normalize
        if np.max(np.abs(reconstructed_time)) > 0:
            reconstructed_time /= np.max(np.abs(reconstructed_time))

        # Truncate to match clean length
        min_length = min(len(clean), len(reconstructed_time))
        clean_trunc = clean[:min_length]
        reconstructed_trunc = reconstructed_time[:min_length]

        # PESQ score
        try:
            score = pesq(fs, clean_trunc, reconstructed_trunc, pesq_mode)
        except Exception as e:
            print(f"PESQ failed for matrix {idx}: {e}")
            score = -np.inf

        if score > best_pesq:
            best_pesq = score
            best_matrix = Phi

    if best_matrix is not None:
        output_path = os.path.join(sub_folder, 'best_sensing_matrix.npy')
        np.save(output_path, best_matrix)
        print(f"Saved best sensing matrix for {signal_name} with PESQ {best_pesq:.3f}")

    # Print bypass result
    if bypass_cs:
        print(f"Bypass PESQ for {signal_name}: {best_pesq:.3f}")

def main():
    """
    Main execution: Process all signals in data folder.
    """
    data_folder = 'noisy_train'
    matrix_base = 'matrix'
    clean_folder = 'clean_train'
    if not os.path.exists(data_folder):
        print(f"Data folder '{data_folder}' not found. Please ensure noisy .wav files are there.")
        return
    file_names = [f for f in os.listdir(data_folder) if f.endswith('.wav')]
    print(f"Found {len(file_names)} signals to process.")
    for file_name in file_names:
        signal_name = os.path.splitext(file_name)[0]
        print(f"Processing {signal_name}...")
        process_signal(signal_name, data_folder, matrix_base, clean_folder, bypass_cs=False)

if __name__ == "__main__":
    main()