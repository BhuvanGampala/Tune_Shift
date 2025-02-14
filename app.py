from flask import Flask, request, render_template, send_from_directory, flash, redirect
import os
import librosa
import soundfile as sf
from pydub import AudioSegment
import numpy as np
from scipy.signal import butter, lfilter

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # This is needed for sessions and flash messages

# Create output directory if it doesn't exist
os.makedirs('static/output', exist_ok=True)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.45 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

def add_reverb(sound, decay=0.5):
    decay_db = 10 * np.log10(decay)
    return sound.apply_gain(decay_db)

@app.route('/', methods=['GET', 'POST'])
def index():
    output_file = None

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            # Save the uploaded MP3 file
            mp3_path = os.path.join('static', 'input', uploaded_file.filename)
            uploaded_file.save(mp3_path)

            # Show a flash message confirming file upload
            flash('File uploaded successfully!', 'success')

            # Load the audio file using librosa
            y, sr = librosa.load(mp3_path, sr=None)

            # Pitch shifting
            y_pitch_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=4)

            # Harmonic and percussive separation
            harmonic, percussive = librosa.effects.hpss(y)

            # Save harmonic and percussive to separate WAV files
            harmonic_path = os.path.join('static', 'output', 'harmonic.wav')
            percussive_path = os.path.join('static', 'output', 'percussive.wav')
            sf.write(harmonic_path, harmonic, sr)
            sf.write(percussive_path, percussive, sr)

            # Time stretching
            y_time_stretched = librosa.effects.time_stretch(y, rate=0.15)

            # Combine pitch-shifted and time-stretched audio
            min_length = min(len(y_pitch_shifted), len(y_time_stretched))
            combined_audio = y_pitch_shifted[:min_length] + y_time_stretched[:min_length]

            # Apply bandpass filter
            y_filtered = bandpass_filter(combined_audio, lowcut=300, highcut=3000, fs=sr)

            # Save filtered audio
            filtered_audio_path = os.path.join('static', 'output', 'filtered_audio.wav')
            sf.write(filtered_audio_path, y_filtered, sr)

            # Add reverb
            audio_segment = AudioSegment.from_wav(filtered_audio_path)
            y_reverb = add_reverb(audio_segment)
            reverb_audio_path = os.path.join('static', 'output', 'reverb_audio.wav')
            y_reverb.export(reverb_audio_path, format="wav")

            # Adjust volume
            volume_change_db = 5
            y_adjusted_volume = y_reverb + volume_change_db
            adjusted_volume_audio_path = os.path.join('static', 'output', 'adjusted_volume_audio.wav')
            y_adjusted_volume.export(adjusted_volume_audio_path, format="wav")

            # Convert to MP3
            final_audio_path = os.path.join('static', 'output', 'transformed_audio.mp3')
            y_final_audio = AudioSegment.from_wav(adjusted_volume_audio_path)
            y_final_audio.export(final_audio_path, format="mp3")

            output_file = 'output/transformed_audio.mp3'  # Path for the frontend to use

    return render_template('index.html', output_file=output_file)

if __name__ == '__main__':
    app.run(debug=True)
