import os

def rename_images(animal_name, main_folder):
    images_folder = os.path.join(main_folder, animal_name, 'audio')

    if os.path.exists(images_folder):
        image_files = [file for file in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, file))]

        for index, old_image_name in enumerate(image_files, start=1):
            old_image_path = os.path.join(images_folder, old_image_name)
            new_image_name = f"{animal_name}{index}.wav"  # You can change the file extension if needed
            new_image_path = os.path.join(images_folder, new_image_name)

            # Rename the image file
            os.rename(old_image_path, new_image_path)
            print(f"Renamed: {old_image_name} to {new_image_name}")

def process_all_animals(main_folder):
    animal_names = ['Zebra']
    for animal_name in animal_names:
        rename_images(animal_name, main_folder)

# Example usage
animals_folder = r"D:\30animalsprocessed_audio_folders"
process_all_animals(animals_folder)
