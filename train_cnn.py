import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils.feature_utils import extract_mfcc_cnn

emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
}

data_path = "data/Audio_Speech_Actors_01-24"
X, y = [], []
for actor in os.listdir(data_path):
    for file in os.listdir(os.path.join(data_path, actor)):
        emotion = emotion_map[file.split("-")[2]]
        path = os.path.join(data_path, actor, file)
        feat = extract_mfcc_cnn(path)
        if feat is not None:
            X.append(feat)
            y.append(emotion)

X = np.array(X)[..., np.newaxis]
y_encoded = LabelEncoder().fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_encoded)
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(40,174,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(8, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
model.save("models/cnn_emotion_model.h5")