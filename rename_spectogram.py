import os

def rename_spectrograms(animal_name, main_folder):
    spectrograms_folder = os.path.join(main_folder, animal_name, 'spectrograms')

    if os.path.exists(spectrograms_folder):
        spectrogram_files = [file for file in os.listdir(spectrograms_folder) if os.path.isfile(os.path.join(spectrograms_folder, file))]

        for index, old_spectrogram_name in enumerate(spectrogram_files, start=1):
            old_spectrogram_path = os.path.join(spectrograms_folder, old_spectrogram_name)
            new_spectrogram_name = f"{animal_name}{index}.jpg"  # You can change the file extension if needed
            new_spectrogram_path = os.path.join(spectrograms_folder, new_spectrogram_name)

            # Rename the spectrogram file
            os.rename(old_spectrogram_path, new_spectrogram_path)
            print(f"Renamed spectrogram: {old_spectrogram_name} to {new_spectrogram_name}")

def process_all_animals(main_folder):
    animal_names =['Alligator','Frog','Gecko','Gorilla','Hippopotamus',
                  'Jaguar','Leopard','Lion','Lynx','mongoose','Monkey','moose','Orangutan','Panda','Panther','Peacock',
                    'Raccon','Rhino','Snake','Wild_Boar','Wolf','Zebra']
    for animal_name in animal_names:
        rename_spectrograms(animal_name, main_folder)

# Example usage
animals_folder = r"D:\\30animalsprocessed_audio_folders"  
process_all_animals(animals_folder)
