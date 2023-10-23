# Neural Network Implementation

This repository contains a simple implementation of a neural network using Python and NumPy. The neural network supports various activation functions, including sigmoid, linear, and softmax. It includes functionality for training the network using mean squared error (MSE) loss and making predictions.

## Prerequisites

Make sure you have the following libraries installed before running the code:
```bash
pip install numpy matplotlib tqdm

## Usage

### Data Preparation

The code generates synthetic data for two classes and splits it into training and testing sets.

```python
import numpy as np

def get_ds(class1_data, class2_data):
    # Implementation of get_ds function
    pass

mean_class1, std_class1 = 10, 0.1
mean_class2, std_class2 = 11, 0.2
num_samples = 100
class1_data = np.random.normal(mean_class1, std_class1, num_samples)
class2_data = np.random.normal(mean_class2, std_class2, num_samples)

X_train, Y_train, X_test, Y_test = get_ds(class1_data, class2_data)

