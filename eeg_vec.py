"""
EEG Feature Extraction Tool
CSV (Synthetic EEG) → Epoching → Feature Extraction

IMPORTANT:
- Metadata columns are EXCLUDED from feature extraction
- Only EEG band columns (_delta/_theta/_alpha/_beta/_gamma) are used

Features:
- RMS
- Band Power
- Spectral Entropy (robust, no complex issues)
- Cross-channel Coherence

Output:
- ML-ready feature CSV
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch, coherence

# =====================================================
# SAFE FEATURE FUNCTIONS
# =====================================================

def rms(sig: np.ndarray) -> float:
    sig = np.asarray(sig, dtype=np.float64)
    return float(np.sqrt(np.mean(sig * sig)))


def band_power(sig: np.ndarray, fs: int) -> float:
    sig = np.asarray(sig, dtype=np.float64)
    freqs, psd = welch(sig, fs=fs, nperseg=len(sig))
    psd = np.real(psd)
    psd = np.clip(psd, 0, None)
    return float(np.trapz(psd, freqs))


def spectral_entropy(sig: np.ndarray, fs: int) -> float:
    """
    Robust spectral entropy:
    - forces real PSD
    - avoids log(0)
    - returns pure float
    """
    sig = np.asarray(sig, dtype=np.float64)
    freqs, psd = welch(sig, fs=fs, nperseg=len(sig))

    psd = np.real(psd)
    psd = np.clip(psd, 0, None)

    total_power = np.sum(psd)
    if total_power <= 0:
        return 0.0

    psd_norm = psd / total_power
    psd_norm = np.clip(psd_norm, 1e-12, 1.0)

    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    return float(entropy)


def coherence_feature(sig1: np.ndarray, sig2: np.ndarray, fs: int) -> float:
    sig1 = np.asarray(sig1, dtype=np.float64)
    sig2 = np.asarray(sig2, dtype=np.float64)

    _, coh = coherence(sig1, sig2, fs=fs, nperseg=len(sig1))
    coh = np.real(coh)
    coh = np.clip(coh, 0, 1)
    return float(np.mean(coh))


# =====================================================
# EPOCHING (EEG SIGNALS ONLY)
# =====================================================

def epoch_signal(df, srate, epoch_sec, overlap):
    """
    Uses ONLY EEG band columns.
    Metadata columns are ignored completely.
    """

    BANDS = ("_delta", "_theta", "_alpha", "_beta", "_gamma")

    signal_cols = [
        c for c in df.columns
        if c.endswith(BANDS)
    ]

    if not signal_cols:
        raise ValueError("No EEG band columns found (_delta/_theta/_alpha/_beta/_gamma).")

    epoch_len = int(epoch_sec * srate)
    step = int(epoch_len * (1 - overlap))

    epochs = []
    labels = []

    for start in range(0, len(df) - epoch_len + 1, step):
        end = start + epoch_len
        segment = df.iloc[start:end][signal_cols].values
        epochs.append(segment)

        # label from metadata (ONLY for labeling, not features)
        labels.append(df["mental_state"].iloc[0] if "mental_state" in df.columns else "unknown")

    return np.array(epochs), labels, signal_cols


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(epochs, signal_cols, fs):
    rows = []

    # Selected coherence pairs (safe & meaningful)
    coherence_pairs = [
        ("Fp1_alpha", "Fp2_alpha"),
        ("F3_beta", "F4_beta"),
        ("C3_alpha", "C4_alpha"),
    ]

    col_index = {c: i for i, c in enumerate(signal_cols)}

    for epoch in epochs:
        features = {}

        # Per-channel-band features
        for idx, col in enumerate(signal_cols):
            sig = epoch[:, idx]

            features[f"{col}_RMS"] = rms(sig)
            features[f"{col}_POWER"] = band_power(sig, fs)
            features[f"{col}_ENTROPY"] = spectral_entropy(sig, fs)

        # Cross-channel coherence features
        for c1, c2 in coherence_pairs:
            if c1 in col_index and c2 in col_index:
                features[f"COH_{c1}_{c2}"] = coherence_feature(
                    epoch[:, col_index[c1]],
                    epoch[:, col_index[c2]],
                    fs
                )

        rows.append(features)

    return pd.DataFrame(rows)


# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(page_title="EEG Feature Extractor", layout="wide")
st.title("EEG Feature Extraction Tool")
st.markdown(
    "**Pipeline:** CSV (Synthetic EEG) → Epoching → Feature Extraction  \n"
    "*Metadata columns are ignored during feature extraction.*"
)

st.sidebar.header("Input Parameters")

uploaded_file = st.sidebar.file_uploader(
    "Upload Synthetic EEG CSV",
    type=["csv"]
)

sample_rate = st.sidebar.selectbox("Sampling Rate (Hz)", [128, 256, 512], index=1)
epoch_sec = st.sidebar.number_input("Epoch Length (seconds)", 1.0, 10.0, 2.0)
overlap = st.sidebar.slider("Epoch Overlap", 0.0, 0.9, 0.5)

run_btn = st.sidebar.button("Run Feature Extraction")

# =====================================================
# MAIN EXECUTION
# =====================================================

if uploaded_file and run_btn:
    with st.spinner("Processing EEG data..."):
        df = pd.read_csv(uploaded_file)

        st.subheader("Input CSV Preview")
        st.dataframe(df.head())

        epochs, labels, signal_cols = epoch_signal(
            df,
            srate=sample_rate,
            epoch_sec=epoch_sec,
            overlap=overlap
        )

        st.success(f"Epoching completed: {len(epochs)} epochs created")
        st.write("EEG signal columns used:", signal_cols[:10], "..." if len(signal_cols) > 10 else "")

        feature_df = extract_features(epochs, signal_cols, sample_rate)

        # add label column (metadata only for labeling)
        feature_df["label"] = labels

        st.subheader("Extracted Features (Preview)")
        st.dataframe(feature_df.head())

        st.write("Feature matrix shape:", feature_df.shape)

        csv_bytes = feature_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Feature CSV",
            csv_bytes,
            file_name="eeg_features.csv",
            mime="text/csv"
        )

else:
    st.info("Upload a synthetic EEG CSV and click **Run Feature Extraction**.")
