import os
import torch
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm

def process_audio_folder(animal_folder, target_duration=5.0, sample_rate=44100, output_dir="processed_audio"):
    os.makedirs(output_dir, exist_ok=True)

    resample_transform = Resample(orig_freq=sample_rate, new_freq=sample_rate)

    animal_class = os.path.basename(animal_folder)
    output_subfolder = os.path.join(output_dir, animal_class)
    os.makedirs(output_subfolder, exist_ok=True)

    audio_folder = os.path.join(animal_folder, 'audio')
    for audio_file in tqdm(os.listdir(audio_folder), desc=f"Processing {animal_class}"):
        input_path = os.path.join(audio_folder, audio_file)
        output_path = os.path.join(output_subfolder, audio_file)

        try:
            # Check the file size before attempting to load it
            file_size = os.path.getsize(input_path)

            if file_size < 1024:  # Set a threshold for minimum file size (adjust as needed)
                print(f"Skipped: {audio_file} - Irregular file size ({file_size} bytes)")
                continue

            waveform, _ = torchaudio.load(input_path)
            waveform = resample_transform(waveform)

            current_duration = waveform.size(1) / sample_rate

            if current_duration < target_duration:
                padding_size = int((target_duration - current_duration) * sample_rate)
                waveform = torch.nn.functional.pad(waveform, (0, padding_size))
            elif current_duration > target_duration:
                target_size = int(target_duration * sample_rate)
                waveform = waveform[:, :target_size]

            torchaudio.save(output_path, waveform, sample_rate)
            print(f"Processed: {audio_file}")
        except UnicodeEncodeError:
            # Handle UnicodeEncodeError by printing a fallback message
            print("Error processing file: Unicode character not supported in output")
        except Exception as e:
            # Print the error message while handling Unicode characters
            print(f"Error processing {audio_file}: {e}".encode(errors='ignore').decode())

            # If there's an error, delete the problematic file and move to the next one
            os.remove(input_path)
            print(f"Deleted: {audio_file}")

# Specify your input and output directories
input_root = "D:\\30animals"
output_directory = "D:\\30animalsprocessed_audio_folders"

# Specify the animals you want to process
animals_to_process = ['Zebra']

# Filter animal_folders based on animals_to_process
animal_folders = [os.path.join(input_root, animal) for animal in animals_to_process]

# Process each animal folder
for animal_folder in animal_folders:
    process_audio_folder(animal_folder, output_dir=output_directory)
