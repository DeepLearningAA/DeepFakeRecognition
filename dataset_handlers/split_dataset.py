import os
from sklearn.model_selection import train_test_split
import shutil

def split_data(input_folder, output_folder, test_size=0.2):
    os.makedirs(output_folder, exist_ok=True)
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    classes = os.listdir(input_folder)

    for class_name in classes:
        class_folder = os.path.join(input_folder, class_name)
        if os.path.isdir(class_folder):
            files = os.listdir(class_folder)

            train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)

            for file_name in train_files:
                src_path = os.path.join(class_folder, file_name)
                dest_path = os.path.join(train_folder, class_name, file_name)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(src_path, dest_path)

            for file_name in test_files:
                src_path = os.path.join(class_folder, file_name)
                dest_path = os.path.join(test_folder, class_name, file_name)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(src_path, dest_path)

    print("División completada.")

# Ejemplo de uso
input_folder = 'data/real_and_fake_face'
output_folder = 'data/splitted'

split_data(input_folder, output_folder, test_size=0.2)
