import os
import random
import shutil

train_images_folder = "D-Fire_dataset/train/images"
train_labels_folder = "D-Fire_dataset/train/labels"
val_images_folder = "D-Fire_dataset/val/images"
val_labels_folder = "D-Fire_dataset/val/labels"

# Create the validation folders if they don't exist
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

# Get the list of image files in the train set
image_files = os.listdir(train_images_folder)

# Calculate the number of images to move to the validation set
num_val_images = int(0.1 * len(image_files))

# Randomly select the images to move
val_image_files = random.sample(image_files, num_val_images)

# Move the selected images and their corresponding labels to the validation set
for image_file in val_image_files:
   # Move image file
   image_src = os.path.join(train_images_folder, image_file)
   image_dst = os.path.join(val_images_folder, image_file)
   shutil.move(image_src, image_dst)

   # Move label file
   label_file = image_file.replace(".jpg", ".txt")
   label_src = os.path.join(train_labels_folder, label_file)
   label_dst = os.path.join(val_labels_folder, label_file)
   shutil.move(label_src, label_dst)