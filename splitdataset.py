import os
from math import ceil
from random import sample
from shutil import move


folder="/home/gabe/jetson-inference/python/training/classification/data/fibrosis" #replace this with the dataset folder


def get_model_class_names(data_directory: str) -> list:
    os.chdir(data_directory)
    class_names = list(filter(os.path.isdir, os.listdir()))
    os.chdir('..')
    return class_names


def list_class_images(data_directory: str, class_name: str, extension_filters = ('jpg', 'png', 'jpeg', 'webp')) -> dict:
    class_images = {}
    for category in ['train', 'test', 'val']:
        class_images[category] = []
        image_directory = os.path.join(data_directory, category, class_name)
        os.makedirs(image_directory, exist_ok=True)
        if not os.listdir(image_directory) and category=="train":
            os.rmdir(image_directory)
            move(os.path.join(data_directory, class_name), os.path.join(data_directory, category))
        for file_name in os.listdir(image_directory):
            name, extension = os.path.splitext(file_name)
            extension = extension.lower().lstrip(".")
            if extension in extension_filters:
                class_images[category].append(file_name)
    return class_images
