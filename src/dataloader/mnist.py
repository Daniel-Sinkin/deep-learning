"""Dataloader for the MNIST dataset"""

import struct
from array import array
from pathlib import Path

import kagglehub
import numpy as np


class MnistDataloader:
    """
    Dataloader dealing with the MNIST dataset, data source is kagglehub.
    """

    def __init__(self):
        self.training_images_filepath = None
        self.training_labels_filepath = None
        self.test_images_filepath = None
        self.test_labels_filepath = None

        self.is_initialized = False

    def read_images_labels_flattened(
        self, use_training_data: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reads either training or test data."""
        if not self.is_initialized:
            raise RuntimeError("Haven't initialized the data!")

        if use_training_data:
            labels_filepath = self.training_labels_filepath
            images_filepath = self.training_images_filepath
        else:
            labels_filepath = self.test_labels_filepath
            images_filepath = self.test_images_filepath

        labels = []
        with open(labels_filepath, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")
            labels = array("B", file.read())

        labels = np.eye(10, dtype=np.float32)[labels]

        with open(images_filepath, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols : (i + 1) * rows * cols])
            images.append(img)
        image_arr = np.vstack(images)

        return image_arr, labels

    def download_data(self) -> None:
        """Downloads the data from kagglehub, or re-uses the cached data if availiable."""
        data_folderpath = Path(kagglehub.dataset_download("hojjatk/mnist-dataset"))
        self.training_images_filepath = data_folderpath.joinpath(
            "train-images-idx3-ubyte/train-images-idx3-ubyte"
        )
        self.training_labels_filepath = data_folderpath.joinpath(
            "train-labels-idx1-ubyte/train-labels-idx1-ubyte"
        )
        self.test_images_filepath = data_folderpath.joinpath(
            "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte"
        )
        self.test_labels_filepath = data_folderpath.joinpath(
            "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte"
        )

        self.is_initialized = True

    def load_data(self):
        """Loads the data from cache and returns two tuples, training data then test data."""
        if not self.is_initialized:
            raise RuntimeError("Haven't initialized the data!")
        x_train, y_train = self.read_images_labels_flattened(use_training_data=True)
        x_test, y_test = self.read_images_labels_flattened(use_training_data=False)
        return (x_train, y_train), (x_test, y_test)
