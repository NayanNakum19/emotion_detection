import streamlit as st
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from utils.feature_utils import extract_mfcc_cnn
from utils.plot_utils import plot_waveform, plot_spectrogram, update_session_log, plot_emotion_trend

st.set_page_config(page_title="🎙️ Emotion Detection App", layout="centered")
st.markdown("""
<style>
body { background-color: #0a0a0a; color: white; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Voice-Based Emotion Recognition")
duration = st.slider("🎛️ Select Recording Duration (s)", 1, 10, 3)

if st.button("🎧 Start Recording"):
    fs = 22050
    st.info("Recording in progress...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        wav.write(tmpfile.name, fs, recording)
        path = tmpfile.name

    st.success("✅ Recording complete!")
    st.pyplot(plot_waveform(path))
    st.pyplot(plot_spectrogram(path))

    model = tf.keras.models.load_model("models/cnn_emotion_model.h5")
    emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

    mfcc = extract_mfcc_cnn(path)
    if mfcc is not None:
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        prediction = model.predict(mfcc)
        emotion = emotion_labels[np.argmax(prediction)]

        st.markdown(f"### 🧠 Predicted Emotion: *{emotion.upper()}*")
        confidences = {label: float(score) for label, score in zip(emotion_labels, prediction[0])}
        bar = go.Figure([go.Bar(x=list(confidences.keys()), y=list(confidences.values()), marker_color='cyan')])
        bar.update_layout(title="🔍 Emotion Prediction Confidence", xaxis_title="Emotion", yaxis_title="Probability")
        st.plotly_chart(bar, use_container_width=True)

        update_session_log(emotion)

        fig = plot_emotion_trend()
        if fig:
            st.subheader("📈 Emotion Trend Log")
            st.pyplot(fig)

        gif_url = {
            "happy": "https://media.giphy.com/media/1BdIPYgNlGyyQ/giphy.gif",
            "sad": "https://media.giphy.com/media/d2lcHJTG5Tscg/giphy.gif",
            "angry": "https://media.giphy.com/media/l3q2K5jinAlChoCLS/giphy.gif",
            "calm": "https://media.giphy.com/media/l0ExdMHUDKteztyfe/giphy.gif",
            "neutral": "https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif",
            "fearful": "https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif",
            "disgust": "https://media.giphy.com/media/Zau0yrl17uzdK/giphy.gif",
            "surprised": "https://media.giphy.com/media/l0IylOPCNkiqOgMyA/giphy.gif"
        }.get(emotion, "")

        if gif_url:
            st.image(gif_url, width=300, caption=f"Emotion Avatar: {emotion.capitalize()}")
    else:
        st.error("❌ Could not extract features from audio.")