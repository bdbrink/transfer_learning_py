import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Specify the classes to be filtered out
split_classes = [8, 9]

# Create boolean masks to filter out specific classes from the training set
mask_train_8 = np.isin(y_train, split_classes, invert=True).flatten()

# Create boolean masks to filter out specific classes from the test set
mask_test_8 = np.isin(y_test, split_classes, invert=True).flatten()

# Print the first 20 elements of the training set mask
print(mask_train_8[:20])
