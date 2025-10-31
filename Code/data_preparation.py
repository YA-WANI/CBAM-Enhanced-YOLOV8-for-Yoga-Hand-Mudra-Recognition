import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

def prepare_data(source_dir, dataset_name='hand_gestures'):
    """
    Prepares and splits the dataset into train, validation, and test sets,
    and generates the YOLOv8 data.yaml configuration file.

    Args:
        source_dir (str): The path to the directory containing the images and labels.
        dataset_name (str): The name for the new dataset directory.
    """
    print("Starting data preparation...")
    source_path = Path(source_dir)
    images_path = source_path / 'images'
    labels_path = source_path / 'labels'

    # Get a list of all image paths
    image_list = sorted([str(p) for p in images_path.rglob('*.jpg')])
    if not image_list:
        print(f"Error: No images found in {images_path}")
        return

    # Split the dataset into training, validation, and test sets
    train_images, temp_images = train_test_split(image_list, test_size=0.3, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

    # Create the new directory structure
    base_dir = Path(dataset_name)
    (base_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (base_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (base_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
    (base_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
    (base_dir / 'test' / 'images').mkdir(parents=True, exist_ok=True)
    (base_dir / 'test' / 'labels').mkdir(parents=True, exist_ok=True)

    # Function to copy files to their new location
    def copy_files(file_list, target_path):
        for img_path in file_list:
            img_path_obj = Path(img_path)
            label_path_obj = labels_path / img_path_obj.with_suffix('.txt').name

            # Copy image and label
            shutil.copy(img_path_obj, target_path / 'images')
            shutil.copy(label_path_obj, target_path / 'labels')

    print(f"Copying {len(train_images)} training images and labels...")
    copy_files(train_images, base_dir / 'train')
    print(f"Copying {len(val_images)} validation images and labels...")
    copy_files(val_images, base_dir / 'val')
    print(f"Copying {len(test_images)} test images and labels...")
    copy_files(test_images, base_dir / 'test')

    # Read class names from a provided file or default to a list
    class_names = ['Adi', 'Anjali', 'Apana, 'Gyan', 'Hakini','Adi', 'Prana', 'Prithvi, 'Surya', 'Varun']

    # Create the data.yaml file
    data_yaml = {
        'path': str(base_dir.resolve()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(class_names)}
    }

    with open(base_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print("\nData preparation complete!")
    print(f"Dataset structure and data.yaml created in '{dataset_name}' directory.")

if __name__ == '__main__':
    # Define the source directory where your original dataset is located
    SOURCE_DIRECTORY = 'path/to/your/dataset'  # e.g., 'C:/Users/yolov_hand_mudra/dataset'
    prepare_data(source_dir=SOURCE_DIRECTORY, dataset_name='hand_mudra_dataset')
