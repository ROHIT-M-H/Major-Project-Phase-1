import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.utils import to_categorical

from docx import Document

# ================= CONFIG =================
TRAIN_DIR = "Features/Train"
MODEL_PATH = "eeg_cnn_lstm_model.h5"
ENCODER_PATH = "label_encoder.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_PATH = "feature_names.pkl"
GENRE_MAP_PATH = "genre_emotion_map.pkl"

# ================= VALENCE MAP =================
VALENCE_AROUSAL = {
    "happy": (-0.8, 0.7),
    "calm": (0.6, 0.2),
    "sad": (-0.6, 0.3),
    "angry": (-0.7, 0.8),
    "neutral": (0.0, 0.0)
}

# ================= DATA LOADER =================
def load_training_data():
    X, y = [], []
    feature_names = None
    genre_map = {}

    for root, _, files in os.walk(TRAIN_DIR):
        for file in files:
            if file.endswith(".csv"):
                genre = root.split(os.sep)[-2]
                df = pd.read_csv(os.path.join(root, file))

                if "label" not in df.columns:
                    continue

                if feature_names is None:
                    feature_names = df.drop(columns=["label"]).columns.tolist()

                for emotion in df["label"]:
                    genre_map.setdefault(genre, []).append(emotion)

                X.append(df.drop(columns=["label"]).values)
                y.extend(df["label"].values)

    X = np.vstack(X)
    y = np.array(y)

    joblib.dump(feature_names, FEATURE_PATH)
    joblib.dump(genre_map, GENRE_MAP_PATH)

    return X, y, feature_names

# ================= MODEL =================
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, 3, activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),

        Conv1D(128, 3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(2),

        Bidirectional(LSTM(64)),
        Dropout(0.4),

        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    return model

# ================= STREAMLIT UI =================
st.set_page_config("EEG Emotion System", layout="wide")
st.title("🧠 EEG-Based Music Emotion Analysis System")

menu = st.sidebar.radio("Navigation", ["Train Model", "Test & Dashboard"])

# ================= TRAIN =================
if menu == "Train Model":
    if st.button("🚀 Start Training"):
        X, y, feature_names = load_training_data()

        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y)
        y_cat = to_categorical(y_enc)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_cat, test_size=0.2, random_state=42
        )

        model = build_model((X_train.shape[1], 1), y_cat.shape[1])
        model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test))

        model.save(MODEL_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        joblib.dump(scaler, SCALER_PATH)

        preds = np.argmax(model.predict(X_test), axis=1)
        true = np.argmax(y_test, axis=1)

        cm = confusion_matrix(true, preds)

        st.success("✅ Training Completed")

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
        st.pyplot(fig)

        st.text(classification_report(true, preds, target_names=encoder.classes_))

# ================= TEST =================
if menu == "Test & Dashboard":
    uploaded = st.file_uploader("Upload EEG CSV (Unlabeled)", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        model = load_model(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURE_PATH)
        genre_map = joblib.load(GENRE_MAP_PATH)

        for col in feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[feature_names]

        X = scaler.transform(df.values)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        probs = model.predict(X)
        emotions = encoder.inverse_transform(np.argmax(probs, axis=1))
        confidence = np.max(probs, axis=1) * 100

        df["Predicted Emotion"] = emotions
        df["Confidence (%)"] = confidence.round(2)

        final_emotion = df["Predicted Emotion"].mode()[0]
        final_conf = df["Confidence (%)"].mean()

        valence, arousal = VALENCE_AROUSAL.get(final_emotion, (0, 0))
        impact = "Positive" if valence > 0 else "Negative"

        st.subheader("Dashboard")
        st.metric("Final Emotion", final_emotion.upper(), f"{final_conf:.2f}%")
        st.metric("Valence", valence)
        st.metric("Arousal", arousal)
        st.metric("Music Impact", impact)

        if impact == "Negative":
            good_genres = [
                g for g, e in genre_map.items()
                if e.count("happy") + e.count("calm") > len(e) * 0.5
            ]
            st.success(f"🎧 Recommended Genres: {', '.join(set(good_genres))}")

        st.dataframe(df)

        # ========== AUTO REPORT ==========
        if st.button("📄 Generate Report"):
            doc = Document()
            doc.add_heading("EEG Emotion Analysis Report", 0)
            doc.add_paragraph(f"Final Emotion: {final_emotion}")
            doc.add_paragraph(f"Confidence: {final_conf:.2f}%")
            doc.add_paragraph(f"Valence: {valence}")
            doc.add_paragraph(f"Arousal: {arousal}")
            doc.add_paragraph(f"Music Impact: {impact}")

            if impact == "Negative":
                doc.add_paragraph("Recommended Genres:")
                for g in set(good_genres):
                    doc.add_paragraph(f"- {g}")

            doc_path = "EEG_Emotion_Report.docx"
            doc.save(doc_path)

            st.success("📄 Report Generated")
            with open(doc_path, "rb") as f:
                st.download_button("Download Report", f, file_name=doc_path)
