import librosa
import noisereduce as nr

def preprocess_audio(audio, sr, target_sr=16000):
    if sr != target_sr:
        audio = librosa.resample(audio, sr, target_sr)
        sr = target_sr
    audio = librosa.util.normalize(audio)
    noise_part = audio[:int(0.5 * sr)]
    audio = nr.reduce_noise(audio, noise_part, sr=sr)
    return audio, sr