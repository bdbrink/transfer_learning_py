# CIFAR-10 Dataset Filtering Script

This script demonstrates how to filter specific classes from the CIFAR-10 dataset using boolean masks. It uses the TensorFlow library to load the CIFAR-10 dataset, applies filters to exclude specific classes, and prints the resulting boolean masks.

## Prerequisites
- Python 3.x
- TensorFlow
- NumPy


## Script Overview
1. The script starts by importing the necessary libraries: `numpy` for array operations and `tensorflow.keras` for dataset loading and model creation.

2. The CIFAR-10 dataset is loaded using the `cifar10.load_data()` function from TensorFlow's Keras API. The dataset consists of 50,000 training images and 10,000 test images, each labeled with one of ten classes.

3. The `split_classes` variable is defined to specify the classes that need to be filtered out from the dataset. In this script, classes 8 (ship) and 9 (truck) are chosen.

4. Boolean masks are created using NumPy's `np.isin()` function to identify the instances belonging to the classes specified in `split_classes`. The masks are created separately for the training set (`mask_train_8`) and the test set (`mask_test_8`).

5. Finally, the script prints the first 20 elements of the training set mask (`mask_train_8`) to demonstrate the filtering process.

## Customization
- To filter different classes from the CIFAR-10 dataset, modify the `split_classes` variable accordingly. You can specify any combination of class numbers from 0 to 9.
- To perform additional operations or analysis on the filtered dataset, you can extend the script as needed.

## Note
- This script only demonstrates the creation of boolean masks to filter out specific classes from the CIFAR-10 dataset. Further steps, such as creating a new dataset or training a model with the filtered data, are not included in this script.

