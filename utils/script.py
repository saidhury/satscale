import os
import cv2

# Define absolute paths
input_train_folder = r"D:\Projects\archive\Dataset\train\train"
input_test_folder = r"D:\Projects\archive\Dataset\test\test"

# Output folders for HR (high-res) and LR (low-res) images
output_hr_folder = "RESISC45_HR"
output_lr_folder = "RESISC45_LR"

# Ensure output folders exist
os.makedirs(output_hr_folder, exist_ok=True)
os.makedirs(output_lr_folder, exist_ok=True)


# Function to process images
def process_images(input_folder, output_hr_folder, output_lr_folder):
    for category in os.listdir(input_folder):
        category_path = os.path.join(input_folder, category)

        if os.path.isdir(category_path):  # Ensure it's a folder
            hr_category_path = os.path.join(output_hr_folder, category)
            lr_category_path = os.path.join(output_lr_folder, category)

            os.makedirs(hr_category_path, exist_ok=True)
            os.makedirs(lr_category_path, exist_ok=True)

            # Get first 5 images only
            images = sorted(os.listdir(category_path))[:5]

            for img_name in images:
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path)

                if img is not None:
                    # Save HR image
                    cv2.imwrite(os.path.join(hr_category_path, img_name), img)

                    # Create and save LR image (50% of original size)
                    img_lr = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
                    cv2.imwrite(os.path.join(lr_category_path, img_name), img_lr)


# Process both train and test folders
process_images(input_train_folder, os.path.join(output_hr_folder, "train"), os.path.join(output_lr_folder, "train"))
process_images(input_test_folder, os.path.join(output_hr_folder, "test"), os.path.join(output_lr_folder, "test"))

print("Paired LR-HR dataset created successfully.")
