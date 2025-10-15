# Speech Enhancement using Compressive Sensing

This project provides a speech enhancement pipeline using Compressive Sensing (CS) to denoise audio signals.

## How to Use

### Prerequisites

*   Python 3.x
*   Windows Operating System (instructions are for PowerShell)
*   The project uses `.wav` audio files.

### Installation

1.  **Clone the repository:**
    ```powershell
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required packages:**
    ```powershell
    pip install -r requirements.txt
    ```

3.  **Troubleshooting `pesq` Installation on Windows:**

    The `pesq` Python package requires C++ build tools to be compiled during installation. If you encounter errors during the installation of `pesq`, you will need to install the Microsoft C++ Build Tools.

    *   **Install Microsoft C++ Build Tools:**
        1.  Go to the [Visual Studio Build Tools download page](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
        2.  Download and run the installer for "Build Tools for Visual Studio".
        3.  During the installation, make sure to select the **"Desktop development with C++"** workload.

    *   **Restart and Re-install:**
        1.  After the installation is complete, restart your PowerShell or VS Code terminal to ensure the changes are applied.
        2.  Try installing `pesq` again:
            ```powershell
            pip install pesq
            ```

### Directory Structure

Before running the project, ensure your directory structure is set up as follows:

```
.
├── noisy_train/
│   └── p226_001.wav
├── clean_train/
│   └── p226_001.wav
├── src/
│   ├── stft.py
│   ├── threshold.py
│   ├── adaptiveGain.py
│   └── Algo.py
└── requirements.txt
```

*   `noisy_train/`: Place your noisy audio files (`.wav`) in this directory.
*   `clean_train/`: Place the corresponding clean (ground truth) audio files in this directory. The file names should match the noisy files.

### Running the Project

The speech enhancement process is divided into several steps. You need to run the scripts from the `src` directory in the following order.

**Note:** The scripts currently have a hardcoded `signal_name`. You will need to modify the `signal_name` variable in each script to match the name of the file you want to process (without the `.wav` extension).

1.  **Step 1: Compute STFT**

    This script computes the Short-Time Fourier Transform (STFT) of the noisy audio and saves it as a `.npz` file.

    ```powershell
    python src/stft.py
    ```
    *Before running, open `src/stft.py` and change `signal_name = 'p226_001'` to your desired file name.*

2.  **Step 2: Apply Thresholding**

    This script loads the STFT data, applies adaptive thresholding to make it sparse, and saves the result.

    ```powershell
    python src/threshold.py
    ```
    *Before running, open `src/threshold.py` and change `signal_name = 'p226_001'` to your desired file name.*

3.  **Step 3: Run the Compressive Sensing Algorithm**

    This is the main script that performs the compressive sensing, reconstruction, and evaluation. It will process all the `.wav` files present in the `noisy_train` directory.

    ```powershell
    python src/Algo.py
    ```
    This script will find the best sensing matrix for each audio file and save it in the `matrix/<signal_name>/` directory.

4.  **Step 4 (Optional): Adaptive Gain for Comparison**

    This script applies a more traditional Wiener filter for speech enhancement. It can be used for comparison.

    ```powershell
    python src/adaptiveGain.py
    ```
    *Before running, open `src/adaptiveGain.py` and change `signal_name = 'p226_001'` to your desired file name.*