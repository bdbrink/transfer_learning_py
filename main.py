import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
split_classes = [8,9]

mask_train_8 = np.isin(y_train, split_classes, invert=True).flatten()
mask_test_8 = np.isin(y_test, split_classes, invert=True).flatten()

print(mask_train_8[:20])