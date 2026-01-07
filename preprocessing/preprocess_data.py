import os
import cv2
import random
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

root_dir = '.'

raw_data_dir = os.path.join(root_dir, 'data', 'raw', 'maps')
processed_data_dir = os.path.join(root_dir, 'data', 'processed')

for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(processed_data_dir, split, 'satellite'), exist_ok=True)
    os.makedirs(os.path.join(processed_data_dir, split, 'map'), exist_ok=True)

image_files_train = [os.path.join(raw_data_dir, 'train', f) for f in os.listdir(os.path.join(raw_data_dir, 'train')) if f.endswith('.jpg')]
image_files_val = [os.path.join(raw_data_dir, 'val', f) for f in os.listdir(os.path.join(raw_data_dir, 'val')) if f.endswith('.jpg')]

image_files = image_files_train + image_files_val

train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

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
    print(f"Processed: {image_path} -> {satellite_save_path}, {map_save_path}")

file_index = 1

for split, files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
    for image_path in files:
        process_image(image_path, split, file_index)
        file_index += 1

print("Preprocessing complete!")
