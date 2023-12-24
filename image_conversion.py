import os

def check_and_delete_non_jpg_images(folder_path):
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        # Check if the file is an image
        if os.path.isfile(image_path) and os.path.splitext(filename)[1].lower() != '.jpg':
            # Delete non-JPG images
            os.remove(image_path)
            print(f"Deleted: {filename}")

if __name__ == "__main__":
    # Replace 'path_to_alligator_folder' with the actual path
    alligator_folder = r"D:\\30animals\\Wild_Boar\\images"

    check_and_delete_non_jpg_images(alligator_folder)
