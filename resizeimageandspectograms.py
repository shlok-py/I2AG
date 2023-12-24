from PIL import Image
import os

def resize_images(images_folder, output_folder, target_size=(1000, 400)):
    os.makedirs(output_folder, exist_ok=True)

    for image_file in os.listdir(images_folder):
        if image_file.endswith('.jpg'):
            # Process and resize the image
            image_path = os.path.join(images_folder, image_file)
            resized_image_path = os.path.join(output_folder, image_file.replace('.jpg', '_resized.jpg'))
            resize_image(image_path, resized_image_path, target_size=target_size)

def resize_image(image_path, output_path, target_size=(1000, 400)):
    image = Image.open(image_path)

    # Convert the image to RGB if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    resized_image = image.resize(target_size)
    resized_image.save(output_path)

def process_animal_folder(animal_folder, target_size=(1000, 400)):
    images_folder = os.path.join(animal_folder, 'images')
    output_images_folder = os.path.join(animal_folder, 'resized_images')

    resize_images(images_folder, output_images_folder, target_size=target_size)

def process_selected_animals(main_folder, selected_animals, target_size=(1000, 400)):
    for animal_name in selected_animals:
        animal_path = os.path.join(main_folder, animal_name)
        if os.path.isdir(animal_path):
            process_animal_folder(animal_path, target_size=target_size)

# Example usage for 'Wolf' and 'Zebra' folders
main_folder = "D:\\30animalsprocessed_audio_folders"
selected_animals = ['Bear','Elephant','Fox','Deer','Cheetah']
process_selected_animals(main_folder, selected_animals)
