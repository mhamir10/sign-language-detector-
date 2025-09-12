import os
import shutil
import random

def split_data(source_image_dir, source_label_dir, train_image_dir, train_label_dir, val_image_dir, val_label_dir, split_ratio=0.85):
    """
    Splits the training data into new training and validation sets.
    source_image_dir: Original training images folder.
    source_label_dir: Original training label files folder.
    train_image_dir: Destination folder for new training images.
    train_label_dir: Destination folder for new training label files.
    val_image_dir: Destination folder for validation images.
    val_label_dir: Destination folder for validation label files.
    split_ratio: Ratio of data for the training set (e.g., 0.85 means 85% training, 15% validation).
    """

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # List all image files
    image_files = [f for f in os.listdir(source_image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files) # Shuffle the files

    # Split into training and validation sets
    num_train = int(len(image_files) * split_ratio)
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]

    print(f"total image: {len(image_files)}")
    print(f"image for new training: {len(train_files)}")
    print(f"image for validation: {len(val_files)}")
    for f in train_files:
        shutil.copy(os.path.join(source_image_dir, f), os.path.join(train_image_dir, f))
        label_file = os.path.splitext(f)[0] + '.txt'
        if os.path.exists(os.path.join(source_label_dir, label_file)):
            shutil.copy(os.path.join(source_label_dir, label_file), os.path.join(train_label_dir, label_file))
        else:
            print(f"Warning: Label file {label_file} not found for {f}.")

    for f in val_files:
        shutil.copy(os.path.join(source_image_dir, f), os.path.join(val_image_dir, f))
        label_file = os.path.splitext(f)[0] + '.txt'
        if os.path.exists(os.path.join(source_label_dir, label_file)):
            shutil.copy(os.path.join(source_label_dir, label_file), os.path.join(val_label_dir, label_file))
        else:
            print(f"Warning: Label file {label_file} not found for {f}.")

print("Data splitting process initiated...")

dataset_root = "D:/university/8th semmester/machine learning/bangla sign language detector"

initial_train_images = os.path.join(dataset_root, 'images', 'train')
initial_train_labels = os.path.join(dataset_root, 'labels', 'train')

new_train_images_dir = os.path.join(dataset_root, 'train_split', 'images')
new_train_labels_dir = os.path.join(dataset_root, 'train_split', 'labels')
new_val_images_dir = os.path.join(dataset_root, 'val_split', 'images')
new_val_labels_dir = os.path.join(dataset_root, 'val_split', 'labels')

split_data(initial_train_images, initial_train_labels,
           new_train_images_dir, new_train_labels_dir,
           new_val_images_dir, new_val_labels_dir,
           split_ratio=0.85)

print("Data splitting complete!")
