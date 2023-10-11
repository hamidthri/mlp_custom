# import numpy as np
# import matplotlib.pyplot as plt
#
# def plot_confusion_matrix(true_labels, predicted_labels):
#     # Find unique classes in true labels and predicted labels
#     classes = np.unique(np.concatenate((true_labels, predicted_labels)))
#
#     # Initialize confusion matrix with zeros
#     confusion_matrix = np.zeros((len(classes), len(classes)))
#
#     # Fill the confusion matrix
#     for i in range(len(true_labels)):
#         true_class_index = np.where(classes == true_labels[i])[0][0]
#         predicted_class_index = np.where(classes == predicted_labels[i])[0][0]
#         confusion_matrix[true_class_index][predicted_class_index] += 1
#
#     # Plot confusion matrix as a heatmap
#     plt.figure(figsize=(len(classes), len(classes)))
#     plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title('Confusion Matrix')
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#
#     # Display the confusion matrix values on the heatmap
#     for i in range(len(classes)):
#         for j in range(len(classes)):
#             plt.text(j, i, str(int(confusion_matrix[i, j])), horizontalalignment='center', color='black')
#
#     plt.show()
#
# # Example usage
# true_labels = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1])
# predicted_labels = np.array([1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0,  1, 0, 0, 0,  1, 0])
#
# plot_confusion_matrix(true_labels, predicted_labels)
#

import numpy as np
#
# a = np.array([1, 2])
# print(a.shape)

import numpy as np

class NN:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, predicted_probs, true_labels):
        num_samples = true_labels.shape[0]
        log_probs = -np.log(predicted_probs[range(num_samples), true_labels])
        loss = np.sum(log_probs) / num_samples
        return loss

    def train(self, X_train, y_train, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            logits = np.dot(X_train, self.weights) + self.biases
            predicted_probs = self.softmax(logits)

            # Compute loss
            loss = self.cross_entropy_loss(predicted_probs, y_train)

            # Backpropagation - Compute gradients
            error = predicted_probs
            error[range(y_train.shape[0]), y_train] -= 1
            gradients = np.dot(X_train.T, error) / y_train.shape[0]

            # Update weights and biases using gradient descent
            self.weights -= self.learning_rate * gradients
            self.biases -= self.learning_rate * np.mean(error, axis=0)

            if epoch % 10 == 0:
                print(f'Epoch: {epoch}, Loss: {loss:.4f}')

# Example usage
input_size = 784  # Example input size for MNIST data
output_size = 10   # Example output size for 10 classes (digits 0 to 9)
X_train = np.random.rand(100, input_size)  # Example input data
y_train = np.random.randint(0, output_size, size=(100,))  # Example true labels

# Initialize and train the neural network
model = NN(input_size, output_size)
model.train(X_train, y_train)
