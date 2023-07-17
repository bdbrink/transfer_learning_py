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

model = Sequential(
    [
        # Convolutional layer 1
        Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
        # Max pooling layer 1
        MaxPooling2D((2, 2)),
        # Dropout layer 1
        Dropout(0.25),

        # Convolutional layer 2
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        # Max pooling layer 2
        MaxPooling2D((2, 2)),
        # Dropout layer 2
        Dropout(0.25),

        # Flatten the feature maps
        Flatten(),

        # Fully connected layer 1
        Dense(512, activation="relu"),
        # Dropout layer 3
        Dropout(0.5),
        # Fully connected layer 2 (output layer)
        Dense(8, activation="softmax")
    ]
)

# Compile the model by specifying the optimizer, loss function, and evaluation metric(s)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model on the filtered training dataset (x_train_8, y_train_8) for 10 epochs,
# using the filtered test dataset (x_test_8, y_test_8) for validation.
model.fit(x_train_8, y_train_8, epochs=10, validation_data=(x_test_8, y_test_8))

