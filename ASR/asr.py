import tensorflow as tf
import tensorflow_hub as hub
import librosa
import numpy as np

# Load the Wav2Vec2 model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/Wav2Vec2/1")

def transcribe_audio(audio, sr):
    if sr != 16000:
        audio = librosa.resample(audio, sr, 16000)
    audio = np.expand_dims(audio, axis=0)
    #Use the model to transcribe the audio
    logits = model.signatures["serving_default"](tf.constant(audio, dtype=tf.float32))["logits"]
    predicted_ids = tf.argmax(logits, axis=-1).numpy().flatten()
    #Convert the transcription to text
    transcription = ''.join([chr(i) for i in predicted_ids])
    return transcription
