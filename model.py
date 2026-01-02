import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.utils import to_categorical

from docx import Document

# ================= CONFIG =================
TRAIN_DIR = "Features/Train"
TEST_DIR = "Features/Test"

MODEL_PATH = "eeg_cnn_lstm_model.h5"
ENCODER_PATH = "label_encoder.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_PATH = "feature_names.pkl"
META_PATH = "metadata.pkl"

# ================= VALENCE MAP =================
VALENCE_AROUSAL = {
    "happy": (0.7, 0.6),
    "calm": (0.5, 0.2),
    "sad": (-0.6, 0.3),
    "angry": (-0.7, 0.8),
    "neutral": (0.0, 0.0)
}

# ================= DATA LOADER =================
def load_dataset(base_dir):
    X, y, meta = [], [], []
    feature_names = None

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".csv"):
                parts = root.split(os.sep)
                gender = parts[-4]
                age = parts[-3]
                genre = parts[-2]

                df = pd.read_csv(os.path.join(root, file))
                if "label" not in df.columns:
                    continue

                if feature_names is None:
                    feature_names = df.drop(columns=["label"]).columns.tolist()

                X.append(df[feature_names].values)
                y.extend(df["label"].values)

                for _ in range(len(df)):
                    meta.append({
                        "gender": gender,
                        "age": age,
                        "genre": genre
                    })

    return np.vstack(X), np.array(y), feature_names, pd.DataFrame(meta)

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
st.title("🧠 EEG-Based Music Emotion Analysis")

menu = st.sidebar.radio("Navigation", [
    "Train Model",
    "Test & Advanced Analytics",
    "Predict Emotion"
])

# ================= TRAIN =================
if menu == "Train Model":
    if st.button("🚀 Train Model"):
        X, y, features, meta = load_dataset(TRAIN_DIR)

        encoder = LabelEncoder()
        y_enc = encoder.fit_transform(y)
        y_cat = to_categorical(y_enc)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).reshape(len(X), X.shape[1], 1)

        model = build_model((X.shape[1], 1), y_cat.shape[1])
        model.fit(X_scaled, y_cat, epochs=25, batch_size=32, validation_split=0.2)

        model.save(MODEL_PATH)
        joblib.dump(encoder, ENCODER_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(features, FEATURE_PATH)
        joblib.dump(meta, META_PATH)

        st.success("✅ Model trained and saved successfully")

# ================= TEST + ANALYTICS =================
if menu == "Test & Advanced Analytics":
    if st.button("🧪 Evaluate Test Data"):
        model = load_model(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)

        X, y, _, meta = load_dataset(TEST_DIR)
        X = scaler.transform(X).reshape(len(X), X.shape[1], 1)

        probs = model.predict(X)
        preds = np.argmax(probs, axis=1)
        true = encoder.transform(y)

        # ---- Metrics ----
        st.metric("Accuracy", f"{accuracy_score(true, preds)*100:.2f}%")
        st.metric("Precision", f"{precision_score(true, preds, average='weighted')*100:.2f}%")
        st.metric("Recall", f"{recall_score(true, preds, average='weighted')*100:.2f}%")

        # ---- Confusion Matrix ----
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(true, preds), annot=True, fmt="d",
                    xticklabels=encoder.classes_,
                    yticklabels=encoder.classes_, cmap="Blues")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.text(classification_report(true, preds, target_names=encoder.classes_))

        # ================= DEMOGRAPHIC GRAPHS =================
        meta["Emotion"] = encoder.inverse_transform(preds)
        meta["Confidence"] = np.max(probs, axis=1)

        st.subheader("📈 Demographic Influence")

        # Gender vs Emotion
        fig, ax = plt.subplots()
        pd.crosstab(meta["gender"], meta["Emotion"], normalize="index").plot(kind="bar", ax=ax)
        ax.set_title("Gender vs Emotion Distribution")
        ax.set_ylabel("Proportion")
        st.pyplot(fig)

        # Age vs Emotion
        fig, ax = plt.subplots()
        pd.crosstab(meta["age"], meta["Emotion"], normalize="index").plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Age Group vs Emotion Distribution")
        ax.set_ylabel("Proportion")
        st.pyplot(fig)

        # ================= GENRE-WISE EMOTION STRENGTH =================
        st.subheader("🧪 Genre-wise Emotion Strength")

        genre_emotion_strength = meta.groupby(["genre", "Emotion"])["Confidence"].mean().unstack().fillna(0)

        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(genre_emotion_strength, annot=True, cmap="YlGnBu")
        ax.set_title("Genre vs Emotion Strength (Mean Confidence)")
        st.pyplot(fig)

        st.dataframe(genre_emotion_strength)

# ================= PREDICT =================
if menu == "Predict Emotion":
    uploaded = st.file_uploader("Upload EEG CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        model = load_model(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
        features = joblib.load(FEATURE_PATH)

        for col in features:
            if col not in df.columns:
                df[col] = 0
        df = df[features]

        X = scaler.transform(df).reshape(len(df), df.shape[1], 1)
        probs = model.predict(X)

        emotion = encoder.inverse_transform([np.argmax(probs.mean(axis=0))])[0]
        valence, arousal = VALENCE_AROUSAL.get(emotion, (0, 0))

        st.metric("Predicted Emotion", emotion.upper())
        # st.metric("Valence", valence)
        # st.metric("Arousal", arousal)
        # st.metric("Impact", "Positive" if valence >= 0 else "Negative")
