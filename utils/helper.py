import os
import shutil

# Define the base directories
folders = ["RESISC45_HR", "RESISC45_LR"]

for folder in folders:
    test_path = os.path.join(folder, "train")

    # Traverse the test subfolders
    for subfolder in os.listdir(test_path):
        subfolder_path = os.path.join(test_path, subfolder)

        if os.path.isdir(subfolder_path):  # Ensure it's a directory
            for file in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file)
                dest_path = os.path.join(folder, file)

                # Move the file
                shutil.move(file_path, dest_path)

            # Remove the now empty subfolder
            os.rmdir(subfolder_path)

    # Remove the empty test directory
    os.rmdir(test_path)
