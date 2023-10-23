# Neural Network Implementation

This repository contains a Python implementation of a neural network using NumPy. The network supports various activation functions, including sigmoid, linear, and softmax. It includes functionality for training the network using mean squared error (MSE) loss and making predictions.

## Prerequisites

Make sure you have the following libraries installed before running the code:

```bash
pip install numpy matplotlib tqdm
```
# Usage
## Data Preparation
The code generates synthetic data for two classes and splits it into training and testing sets. you can change it based on your data.
```
mean_class1, std_class1 = 10, 0.1
mean_class2, std_class2 = 11, 0.2
num_samples = 100
class1_data = np.random.normal(mean_class1, std_class1, num_samples)
class2_data = np.random.normal(mean_class2, std_class2, num_samples)

X_train, Y_train, X_test, Y_test = get_ds(class1_data, class2_data)
```

## Network Configuration

Define the architecture of your neural network by specifying the number of nodes and activation functions for each layer in the `layers` dictionary. The name of each layer should be in the format `Danse*i*`. `n` represents the number of neurons in each layer, and `'activation'` specifies the activation function for each layer.


```
layers = {
    'Dense1': {'n': 5, 'activation': 'sigmoid'},
    'Dense2': {'n': 4, 'activation': 'sigmoid'},
    'Dense3': {'n': 1, 'activation': 'sigmoid'}
}

```
