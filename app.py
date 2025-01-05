from flask import Flask, request, jsonify, render_template, g
import librosa
import soundfile as sf
import numpy as np
import pyaudio
import wave
import pickle
import sqlite3
import os

app = Flask(__name__)

# Load the trained model and scaler from the specified path
def load_model_and_scaler(model_path="audio_recorder/models/trained_model.pkl", scaler_path="audio_recorder/models/scaler.pkl"):
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)

        with open(scaler_path, "rb") as file:
            scaler = pickle.load(file)

        return model, scaler
    except Exception as e:
        print(f"Error loading model and scaler: {e}")
        return None, None

# Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def connect_db(db_name='audio_emotion_detection.db'):
    if 'db' not in g:
        g.db = sqlite3.connect(db_name)
        cursor = g.db.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS AudioData (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            audio BLOB,
            emotion TEXT,
            neutral_prob REAL,
            calm_prob REAL,
            happy_prob REAL,
            sad_prob REAL,
            angry_prob REAL,
            fearful_prob REAL,
            disgust_prob REAL,
            surprised_prob REAL
        )''')
        g.db.commit()
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Recording user audio
def record_audio(filename="Predict-Record-Audio.wav", seconds=5, fs=44100):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1

    p = pyaudio.PyAudio()  # Create an interface to PortAudio
    print('Recording...')

    stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
    frames = []  # Initialize array to store frames

    for _ in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()  # Terminate the PortAudio interface

    print('Finished recording')

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

    return filename

# Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        
        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        
        if mel:
            mel_spectrogram = librosa.feature.melspectrogram(y=X, sr=sample_rate)
            mel = np.mean(mel_spectrogram.T, axis=0)
            result = np.hstack((result, mel))

    return result

# Predict with emotion probabilities
def predict_emotion_with_probabilities(model, scaler, features):
    features_scaled = scaler.transform([features])  # Scale the features using the pre-trained scaler
    probabilities = model.predict_proba(features_scaled)  # Get probabilities for each emotion
    probabilities_dict = {}

    for emotion, probability in zip(model.classes_, probabilities[0]):
        probabilities_dict[emotion] = probability
        print(f"{emotion}: {probability * 100:.2f}%")

    predicted_emotion = model.predict(features_scaled)
    print(f"Predicted Emotion: {predicted_emotion[0]}")
    
    return predicted_emotion[0], probabilities_dict

# Insert audio file into SQLite database
def insert_audio_to_db(cursor, filename, emotion="unknown", probabilities=None):
    if probabilities is None:
        probabilities = {emotion: 0.0}  # Default to zero if no probabilities are provided

    with open(filename, 'rb') as file:
        audio_data = file.read()

    try:
        cursor.execute("INSERT INTO AudioData (filename, audio, emotion, neutral_prob, calm_prob, happy_prob, sad_prob, angry_prob, fearful_prob, disgust_prob, surprised_prob) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                       (filename, sqlite3.Binary(audio_data), emotion,
                        probabilities.get('neutral', 0), probabilities.get('calm', 0),
                        probabilities.get('happy', 0), probabilities.get('sad', 0),
                        probabilities.get('angry', 0), probabilities.get('fearful', 0),
                        probabilities.get('disgust', 0), probabilities.get('surprised', 0)))
        print(f"Audio {filename} inserted into database.")
    except Exception as e:
        print(f"Error inserting {filename} into database: {e}")

# Initialize global model and scaler
model, scaler = load_model_and_scaler()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record_and_predict():
    conn = connect_db()  # Use the connection from g
    cursor = conn.cursor()
    
    filename = record_audio()  # Record audio to predict
    feature_predict_audio = extract_feature(filename, mfcc=True, chroma=True, mel=True)  # Extract features
    predicted_emotion, probabilities = predict_emotion_with_probabilities(model, scaler, feature_predict_audio)  # Predict emotion
    insert_audio_to_db(cursor, filename, predicted_emotion, probabilities)  # Insert audio and prediction into SQLite
    conn.commit()  # Commit after recording and predicting
    return jsonify({'emotion': predicted_emotion, 'probabilities': probabilities})

@app.route('/retrieve', methods=['GET'])
def retrieve_and_predict():
    conn = connect_db()  # Ensure a database connection is established
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, audio FROM AudioData")
    rows = cursor.fetchall()
    results = []
    
    for row in rows:
        audio_id, filename, audio_blob = row
        audio_path = f"retrieved_{filename}"
        
        # Write blob to a temporary file
        with open(audio_path, "wb") as f:
            f.write(audio_blob)
        
        print(f"Audio {audio_id} retrieved from database.")
        
        # Predict emotion on retrieved audio
        feature_predict_audio = extract_feature(audio_path, mfcc=True, chroma=True, mel=True)
        predicted_emotion, probabilities = predict_emotion_with_probabilities(model, scaler, feature_predict_audio)
        
        # Store result
        results.append({'id': audio_id, 'emotion': predicted_emotion, 'probabilities': probabilities})
        
        # Remove the temporary file
        os.remove(audio_path)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
