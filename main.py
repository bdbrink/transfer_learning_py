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

# Create subsets of the training and test data by filtering out specific classes
x_train_8, y_train_8 = x_train[mask_train_8], y_train[mask_train_8]
x_test_8, y_test_8 = x_test[mask_test_8], y_test[mask_test_8]

# Create boolean masks to filter out specific classes from the training set
mask_train_2 = np.isin(y_train, split_classes).flatten()
# Create boolean masks to filter out specific classes from the test set
mask_test_2 = np.isin(y_test, split_classes).flatten()

# Create subsets of the training and test data by filtering out specific classes
x_train_2, y_train_2 = x_train[mask_train_2], y_train[mask_train_2]
x_test_2, y_test_2 = x_test[mask_test_2], y_test[mask_test_2]

print(len(x_train_2))  # Print the number of instances in the x_train_2 dataset
print(len(x_train_8))  # Print the number of instances in the x_train_8 dataset

y_train_2 = np.isin(y_train_2, split_classes[0]).astype(int)  # Create a binary mask where y_train_2 elements belonging to split_classes[0] are set to 1, and others are set to 0
y_train_2 = np.isin(y_test_2, split_classes[0]).astype(int)  # Create a binary mask where y_test_2 elements belonging to split_classes[0] are set to 1, and others are set to 0
