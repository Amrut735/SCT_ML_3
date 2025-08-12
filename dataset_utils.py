import os
import shutil
import random
from pathlib import Path

def create_dataset_structure(base_path="dataset"):
    """
    Create the required dataset folder structure
    
    Args:
        base_path (str): Base path for the dataset
    """
    folders = [
        os.path.join(base_path, "train", "cats"),
        os.path.join(base_path, "train", "dogs"),
        os.path.join(base_path, "test", "cats"),
        os.path.join(base_path, "test", "dogs")
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Created folder: {folder}")

def validate_dataset_structure(dataset_path="dataset"):
    """
    Validate that the dataset has the correct structure
    
    Args:
        dataset_path (str): Path to the dataset
        
    Returns:
        bool: True if structure is valid, False otherwise
    """
    required_folders = [
        os.path.join(dataset_path, "train", "cats"),
        os.path.join(dataset_path, "train", "dogs"),
        os.path.join(dataset_path, "test", "cats"),
        os.path.join(dataset_path, "test", "dogs")
    ]
    
    for folder in required_folders:
        if not os.path.exists(folder):
            print(f"Missing required folder: {folder}")
            return False
    
    print("Dataset structure is valid!")
    return True

def count_images_in_dataset(dataset_path="dataset"):
    """
    Count the number of images in each folder
    
    Args:
        dataset_path (str): Path to the dataset
    """
    folders = {
        "train_cats": os.path.join(dataset_path, "train", "cats"),
        "train_dogs": os.path.join(dataset_path, "train", "dogs"),
        "test_cats": os.path.join(dataset_path, "test", "cats"),
        "test_dogs": os.path.join(dataset_path, "test", "dogs")
    }
    
    print("Image counts:")
    print("-" * 30)
    
    total_images = 0
    for name, folder in folders.items():
        if os.path.exists(folder):
            image_count = len([f for f in os.listdir(folder) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            print(f"{name}: {image_count} images")
            total_images += image_count
        else:
            print(f"{name}: Folder not found")
    
    print("-" * 30)
    print(f"Total images: {total_images}")

def create_sample_dataset(source_path, dataset_path="dataset", train_ratio=0.8, max_images_per_class=1000):
    """
    Create a sample dataset from a source folder containing cat and dog images
    
    Args:
        source_path (str): Path to source folder containing cat and dog images
        dataset_path (str): Path to create the dataset
        train_ratio (float): Ratio of images to use for training (0.0 to 1.0)
        max_images_per_class (int): Maximum number of images per class to use
    """
    # Create dataset structure
    create_dataset_structure(dataset_path)
    
    # Find cat and dog images in source folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    cat_images = []
    dog_images = []
    
    for file in os.listdir(source_path):
        if file.lower().endswith(image_extensions):
            file_lower = file.lower()
            if 'cat' in file_lower:
                cat_images.append(file)
            elif 'dog' in file_lower:
                dog_images.append(file)
    
    print(f"Found {len(cat_images)} cat images and {len(dog_images)} dog images")
    
    # Limit the number of images per class
    cat_images = cat_images[:max_images_per_class]
    dog_images = dog_images[:max_images_per_class]
    
    # Split into train and test
    random.shuffle(cat_images)
    random.shuffle(dog_images)
    
    cat_train_count = int(len(cat_images) * train_ratio)
    dog_train_count = int(len(dog_images) * train_ratio)
    
    cat_train = cat_images[:cat_train_count]
    cat_test = cat_images[cat_train_count:]
    dog_train = dog_images[:dog_train_count]
    dog_test = dog_images[dog_train_count:]
    
    # Copy images to appropriate folders
    def copy_images(image_list, source_folder, target_folder):
        for img in image_list:
            src = os.path.join(source_folder, img)
            dst = os.path.join(target_folder, img)
            shutil.copy2(src, dst)
    
    copy_images(cat_train, source_path, os.path.join(dataset_path, "train", "cats"))
    copy_images(cat_test, source_path, os.path.join(dataset_path, "test", "cats"))
    copy_images(dog_train, source_path, os.path.join(dataset_path, "train", "dogs"))
    copy_images(dog_test, source_path, os.path.join(dataset_path, "test", "dogs"))
    
    print(f"Created sample dataset:")
    print(f"  Train cats: {len(cat_train)}")
    print(f"  Train dogs: {len(dog_train)}")
    print(f"  Test cats: {len(cat_test)}")
    print(f"  Test dogs: {len(dog_test)}")

def download_sample_images():
    """
    Provide instructions for downloading sample images
    """
    print("To get sample images for testing:")
    print("1. Download the Kaggle Dogs vs Cats dataset:")
    print("   https://www.kaggle.com/c/dogs-vs-cats")
    print("2. Extract the train.zip file")
    print("3. Use create_sample_dataset() function to organize the images")
    print("\nAlternative: Use any folder containing cat and dog images")
    print("and use create_sample_dataset() to organize them.")

def main():
    """Main function for dataset utilities"""
    print("Dataset Utilities for Cat-Dog Classification")
    print("=" * 50)
    
    # Check if dataset exists
    if os.path.exists("dataset"):
        print("Dataset folder found!")
        validate_dataset_structure()
        count_images_in_dataset()
    else:
        print("Dataset folder not found.")
        print("Use one of the following options:")
        print("1. create_dataset_structure() - Create empty folder structure")
        print("2. create_sample_dataset(source_path) - Create from existing images")
        print("3. download_sample_images() - Get instructions for downloading")

if __name__ == "__main__":
    main() 