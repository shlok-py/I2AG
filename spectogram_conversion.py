import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def audio_to_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path)
    S = librosa.stft(y)                           # Short Time Fourier Transformation
    D = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    
    plt.savefig(save_path)
    plt.close()

def process_animal_folder(animal_folder):
    audio_folder = os.path.join(animal_folder, 'audio')
    output_folder = os.path.join(animal_folder, 'spectrograms')

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for audio_file in os.listdir(audio_folder):
        if audio_file.endswith(('.wav', '.mp3')):
            audio_path = os.path.join(audio_folder, audio_file)
            output_path = os.path.join(output_folder, audio_file.replace('.wav', '_spectrogram.jpg').replace('.mp3', '_spectrogram.jpg'))
            
            audio_to_spectrogram(audio_path, output_path)

def process_all_animals(main_folder):
    for animal_folder in os.listdir(main_folder):
        animal_path = os.path.join(main_folder, animal_folder)
        if os.path.isdir(animal_path):
            process_animal_folder(animal_path)

# Example usage
main_folder = "D:\\30animalsprocessed_audio_folders"
process_all_animals(main_folder)
