import os
import cv2
import random
import shutil
import numpy as np
from sklearn.model_selection import train_test_split


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

raw_data_dir = os.path.join(root_dir, 'data', 'raw', 'maps')
processed_data_dir = os.path.join(root_dir, 'data', 'processed')

# os.path.makedirs(os.path.join(processed_data_dir, 'train'),  exist_ok=True)
# os.path.makedirs(os.path.join(processed_data_dir, 'train'),  exist_ok=True)
# os.path.makedirs(os.path.join(processed_data_dir, 'train'),  exist_ok=True)


for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(processed_data_dir, split, 'satellite'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, split, 'map'), exist_ok=True)

image_files_train = [os.path.join(raw_data_dir, 'train', f) for f in os.listdir(os.path.join(raw_data_dir, 'train')) if f.endswith('.jpg')]
image_files_val = [os.path.join(raw_data_dir, 'val', f) for f in os.listdir(os.path.join(raw_data_dir, 'val')) if f.endswith('.jpg')]

image_files = image_files_train + image_files_val

train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

import cv2
import numpy as np
import random

def augment_pair(sat_img, map_img, num_augments=3):
    """Augment a single satellite-map pair and return list of augmented pairs."""
    augmented_pairs = []

    for _ in range(num_augments):
        sat_aug = sat_img.copy()
        map_aug = map_img.copy()

        # 1. Flip
        if random.random() < 0.5:
            sat_aug = cv2.flip(sat_aug, 1)  # Horizontal
            map_aug = cv2.flip(map_aug, 1)
        if random.random() < 0.3:
            sat_aug = cv2.flip(sat_aug, 0)  # Vertical
            map_aug = cv2.flip(map_aug, 0)

        # 2. Rotation ±15 degrees
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((sat_aug.shape[1]//2, sat_aug.shape[0]//2), angle, 1)
        sat_aug = cv2.warpAffine(sat_aug, M, (sat_aug.shape[1], sat_aug.shape[0]), borderMode=cv2.BORDER_REFLECT)
        map_aug = cv2.warpAffine(map_aug, M, (map_aug.shape[1], map_aug.shape[0]), borderMode=cv2.BORDER_REFLECT)

        # 3. Brightness/contrast (satellite only)
        if random.random() < 0.5:
            alpha = random.uniform(0.9, 1.1)  # contrast
            beta = random.randint(-10, 10)    # brightness
            sat_aug = cv2.convertScaleAbs(sat_aug, alpha=alpha, beta=beta)

        # 4. Optional: add small Gaussian noise
        if random.random() < 0.1:
            noise = np.random.normal(0, 5, sat_aug.shape).astype(np.uint8)
            sat_aug = cv2.add(sat_aug, noise)

        augmented_pairs.append((sat_aug, map_aug))

    return augmented_pairs


def process_and_augment_image(image_path, split, file_index, num_augments=3):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to load image {image_path}. Skipping.")
        return file_index

    satellite = image[:, :600, :]
    map_img = image[:, 600:, :]
    satellite_resized = cv2.resize(satellite, (64, 64))
    map_resized = cv2.resize(map_img, (64, 64))

    # Save original
    sat_save_path = os.path.join(processed_data_dir, split, 'satellite', f'{file_index}.jpg')
    map_save_path = os.path.join(processed_data_dir, split, 'map', f'{file_index}.jpg')
    cv2.imwrite(sat_save_path, satellite_resized)
    cv2.imwrite(map_save_path, map_resized)
    file_index += 1

    # Save augmented images
    augmented_pairs = augment_pair(satellite_resized, map_resized, num_augments=num_augments)
    for sat_aug, map_aug in augmented_pairs:
        sat_save_path = os.path.join(processed_data_dir, split, 'satellite', f'{file_index}.jpg')
        map_save_path = os.path.join(processed_data_dir, split, 'map', f'{file_index}.jpg')
        cv2.imwrite(sat_save_path, sat_aug)
        cv2.imwrite(map_save_path, map_aug)
        file_index += 1

    return file_index

def process_image(image_path, split, file_index):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Failed to load image {image_path}. Skipping.")
        return
    satellite = image[:, :600, :]
    map_img = image[:, 600:, :]
    satellite_resized = cv2.resize(satellite, (64, 64))
    map_resized = cv2.resize(map_img, (64, 64))
    satellite_save_path = os.path.join(processed_data_dir, split, 'satellite', f'{file_index}.jpg')
    map_save_path = os.path.join(processed_data_dir, split, 'map', f'{file_index}.jpg')
    cv2.imwrite(satellite_save_path, satellite_resized)
    cv2.imwrite(map_save_path, map_resized)
    print(f"Processed: {image_path} -> {satellite_save_path}, {map_save_path}\n")


file_index = 1
num_augments = 3  # each original image generates 3 more

for split, files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
    for image_path in files:
        file_index = process_and_augment_image(image_path, split, file_index, num_augments=num_augments)
        # process_image(image_path, split, file_index)
        # file_index += 1


print("\nPreprocessing complete!")
