import os

def sanitize_and_print(string):
    try:
        # Attempt to print the string
        print(string)
    except UnicodeEncodeError:
        # If a UnicodeEncodeError occurs, sanitize the string before printing
        sanitized_string = string.encode('ascii', 'ignore').decode('ascii')
        print(f"Sanitized: {sanitized_string}")

def rename_images(animal_name, main_folder):
    images_folder = os.path.join(main_folder, animal_name, 'images')

    if os.path.exists(images_folder):
        image_files = [file for file in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, file))]

        # Sort the image files to ensure consistent order
        image_files.sort()

        for index, old_image_name in enumerate(image_files, start=1):
            old_image_path = os.path.join(images_folder, old_image_name)
            new_image_name = f"{animal_name}{index}.jpg"  # Change the file extension if needed
            new_image_path = os.path.join(images_folder, new_image_name)

            try:
                # Rename the image file
                os.rename(old_image_path, new_image_path)
                sanitize_and_print(f"Renamed: {old_image_name} to {new_image_name}")
            except UnicodeEncodeError as e:
                # Handle UnicodeEncodeError by printing an error message
                sanitize_and_print(f"Error renaming {old_image_name}: {e}")
                # Continue to the next file
                continue
            except Exception as e:
                # Handle other exceptions by printing an error message
                sanitize_and_print(f"Error renaming {old_image_name}: {e}")

def process_specific_animals(main_folder):
    specific_animals = ['Bear','Cheetah','Deer','Elephant','Fox']

    for animal_name in specific_animals:
        rename_images(animal_name, main_folder)

# Example usage
animals_folder = r"D:\30animalsprocessed_audio_folders"
process_specific_animals(animals_folder)
