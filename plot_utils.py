import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import numpy as np

def plot_waveform(file_path):
    y, sr = librosa.load(file_path)
    plt.figure(figsize=(10, 3))
    plt.plot(np.linspace(0, len(y)/sr, len(y)), y, color='cyan')
    plt.title("🎵 Voice Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return plt

def plot_spectrogram(file_path):
    y, sr = librosa.load(file_path)
    stft = np.abs(librosa.stft(y))
    db = librosa.amplitude_to_db(stft, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(db, sr=sr, x_axis='time', y_axis='hz', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title("🔊 Spectrogram")
    plt.tight_layout()
    return plt

def update_session_log(emotion, file="logs/session_emotions.csv"):
    import os, csv
    from datetime import datetime
    os.makedirs("logs", exist_ok=True)
    with open(file, mode='a', newline='') as f:
        csv.writer(f).writerow([datetime.now().strftime("%H:%M:%S"), emotion])

def plot_emotion_trend(file="logs/session_emotions.csv"):
    try:
        df = pd.read_csv(file, header=None, names=["Time", "Emotion"])
        plt.figure(figsize=(10, 3))
        plt.plot(df["Time"], df["Emotion"], marker='o', linestyle='-', color='magenta')
        plt.xticks(rotation=45)
        plt.title("📈 Emotion Trend")
        plt.xlabel("Time")
        plt.ylabel("Emotion")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        return plt
    except Exception as e:
        print(f"Trend Plot Error: {e}")
        return None