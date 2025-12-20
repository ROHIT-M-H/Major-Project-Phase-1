"""
EEG Feature Extraction Tool
CSV (Synthetic EEG) → Epoching → Feature Extraction

IMPROVED COHERENCE REALISM

IMPORTANT:
- Metadata columns are EXCLUDED from feature extraction
- Only EEG band columns (_delta/_theta/_alpha/_beta/_gamma) are used
- Label is taken from `final_emotion`

Features:
- RMS
- Band Power (Welch)
- Spectral Entropy
- Band-limited Cross-channel Coherence

Output:
- ML-ready feature CSV
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch, coherence

# =====================================================
# BAND DEFINITIONS (for realistic coherence)
# =====================================================
BAND_LIMITS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 80),
}

# =====================================================
# SAFE FEATURE FUNCTIONS
# =====================================================

def rms(sig: np.ndarray) -> float:
    sig = np.asarray(sig, dtype=np.float64)
    return float(np.sqrt(np.mean(sig * sig)))


def band_power(sig: np.ndarray, fs: int) -> float:
    sig = np.asarray(sig, dtype=np.float64)
    freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), fs * 2))
    psd = np.real(psd)
    psd = np.clip(psd, 0, None)
    return float(np.trapz(psd, freqs))


def spectral_entropy(sig: np.ndarray, fs: int) -> float:
    sig = np.asarray(sig, dtype=np.float64)
    freqs, psd = welch(sig, fs=fs, nperseg=min(len(sig), fs * 2))

    psd = np.real(psd)
    psd = np.clip(psd, 0, None)

    total_power = np.sum(psd)
    if total_power <= 0:
        return 0.0

    psd_norm = psd / total_power
    psd_norm = np.clip(psd_norm, 1e-12, 1.0)

    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    return float(entropy)


def band_limited_coherence(sig1, sig2, fs, band):
    """
    Realistic coherence:
    - Computes coherence
    - Restricts to target frequency band
    - Avoids artificial inflation
    """
    sig1 = np.asarray(sig1, dtype=np.float64)
    sig2 = np.asarray(sig2, dtype=np.float64)

    freqs, coh = coherence(
        sig1,
        sig2,
        fs=fs,
        nperseg=min(len(sig1), fs * 2)
    )

    coh = np.real(coh)
    coh = np.clip(coh, 0, 1)

    f_low, f_high = BAND_LIMITS[band]
    mask = (freqs >= f_low) & (freqs <= f_high)

    if not np.any(mask):
        return 0.0

    return float(np.mean(coh[mask]))


# =====================================================
# EPOCHING
# =====================================================

def epoch_signal(df, srate, epoch_sec, overlap):
    BANDS = ("_delta", "_theta", "_alpha", "_beta", "_gamma")
    signal_cols = [c for c in df.columns if c.endswith(BANDS)]

    if not signal_cols:
        raise ValueError("No EEG band columns found.")

    epoch_len = int(epoch_sec * srate)
    step = max(1, int(epoch_len * (1 - overlap)))

    epochs, labels = [], []

    for start in range(0, len(df) - epoch_len + 1, step):
        end = start + epoch_len
        epochs.append(df.iloc[start:end][signal_cols].values)
        labels.append(df["final_emotion"].iloc[0] if "final_emotion" in df.columns else "unknown")

    return np.array(epochs), labels, signal_cols


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_features(epochs, signal_cols, fs):
    rows = []

    # (channel1, channel2, band)
    coherence_pairs = [
        ("Fp1_alpha", "Fp2_alpha", "alpha"),
        ("F3_beta", "F4_beta", "beta"),
        ("C3_alpha", "C4_alpha", "alpha"),
        ("P3_alpha", "P4_alpha", "alpha"),
    ]

    col_index = {c: i for i, c in enumerate(signal_cols)}

    for epoch in epochs:
        features = {}

        for idx, col in enumerate(signal_cols):
            sig = epoch[:, idx]
            features[f"{col}_RMS"] = rms(sig)
            features[f"{col}_POWER"] = band_power(sig, fs)
            features[f"{col}_ENTROPY"] = spectral_entropy(sig, fs)

        for c1, c2, band in coherence_pairs:
            if c1 in col_index and c2 in col_index:
                features[f"COH_{c1}_{c2}"] = band_limited_coherence(
                    epoch[:, col_index[c1]],
                    epoch[:, col_index[c2]],
                    fs,
                    band
                )

        rows.append(features)

    return pd.DataFrame(rows)


# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(page_title="EEG Feature Extractor", layout="wide")
st.title("EEG Feature Extraction Tool")

st.markdown(
    "**Pipeline:** Synthetic EEG CSV → Epoching → Feature Extraction  \n"
    "*Metadata is used ONLY for labeling, never as features.*"
)

st.sidebar.header("Input Parameters")

uploaded_file = st.sidebar.file_uploader("Upload Synthetic EEG CSV", type=["csv"])
sample_rate = st.sidebar.selectbox("Sampling Rate (Hz)", [128, 256, 512], index=1)
epoch_sec = st.sidebar.number_input("Epoch Length (seconds)", 1.0, 10.0, 2.0)
overlap = st.sidebar.slider("Epoch Overlap", 0.0, 0.9, 0.5)

run_btn = st.sidebar.button("Run Feature Extraction")

# =====================================================
# MAIN
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

        st.success(f"Epoching completed: {len(epochs)} epochs")
        st.write("EEG columns used:", signal_cols[:10], "..." if len(signal_cols) > 10 else "")

        feature_df = extract_features(epochs, signal_cols, sample_rate)
        feature_df["label"] = labels

        st.subheader("Extracted Features (Preview)")
        st.dataframe(feature_df.head())

        st.write("Feature matrix shape:", feature_df.shape)

        st.download_button(
            "Download Feature CSV",
            feature_df.to_csv(index=False).encode("utf-8"),
            file_name="eeg_features.csv",
            mime="text/csv"
        )
else:
    st.info("Upload a synthetic EEG CSV and click **Run Feature Extraction**.")
