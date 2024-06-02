import os
import glob
import soundfile as sf

from load_audio import load_audio
from preprocessing import preprocess_audio
from enhancement.enhancement import compute_spectrogram, plot_spectrogram
from ASR.asr import transcribe_audio

#Directory paths
audio_dir = "C:/Users/dacia/Desktop/facultate - 2nd half/Bachelors/Audios/forProcessing"
processed_audio_dir = "C:/Users/dacia/Desktop/facultate - 2nd half/Bachelors/Audios/processed"
transcripts_dir = "C:/Users/dacia/Desktop/facultate - 2nd half/Bachelors/Audios/transcripts"

os.makedirs(processed_audio_dir, exist_ok=True)
os.makedirs(transcripts_dir, exist_ok=True)

#Find all files with .wav extension in the directory
audio_paths = glob.glob(os.path.join(audio_dir, "*.wav"))

#Process each audio file
for file_path in audio_paths:
    #Load and preprocess audio
    audio, sr = load_audio(file_path)
    audio, sr = preprocess_audio(audio, sr)
    
    #Save processed audio to new file
    processed_audio_path = os.path.join(processed_audio_dir, os.path.basename(file_path))
    sf.write(processed_audio_path, audio, sr)
    
    #Compute and plot spectrogram
    spectrogram_db = compute_spectrogram(audio, sr)
    plot_spectrogram(spectrogram_db, sr, title=os.path.basename(file_path))
    
    #Transcribe audio
    transcript = transcribe_audio(audio, sr)
    
    #Save transcript to text file
    transcript_file_path = os.path.join(transcripts_dir, os.path.basename(file_path) + ".txt")
    with open(transcript_file_path, 'w') as f:
        f.write(transcript)
