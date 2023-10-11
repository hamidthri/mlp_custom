
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class Activations:
	def __init__(self):
		pass
	def initialize_weight(self):
		self.weights = {}
		for i in range(1, len(self.layer) + 1):
			variance = 2.0 / (self.X_train.shape[1] if i == 1 else self.layer["Dense{}".format(i - 1)]["n"] + self.layer["Dense{}".format(i)]["n"])
			self.weights["W{}".format(i)] = np.random.normal(loc=0, scale=np.sqrt(variance), size=(self.layer["Dense{}".format(i)]["n"], self.X_train.shape[1] if i == 1 else self.layer["Dense{}".format(i - 1)]["n"]))
			self.weights["b{}".format(i)] = np.random.normal(loc=0, scale=np.sqrt(variance), size=(self.layer["Dense{}".format(i)]["n"], 1))

	def my_softmax(self, z):
		a = np.exp(z) / np.sum(np.exp(z))
		return a
	def sig(self, z):
		return 1 / (1 + np.exp(-z))

	def linear(self, z):
		return z

	def relu(self, z):
		return np.max(0, z)

	def d_sig(self, z):
		return np.diagflat(self.sig(z) * (1 - self.sig(z)))

	def MSE(self, label, out):
		return 0.5 * (label - out)**2

	def activations(self, i, z):
		activation = self.layer["Dense{}".format(i)]['activation']
		if activation == 'sigmoid':
			return self.sig(z)
		elif activation == 'linear':
			return self.linear(z)

	def d_activations(self, i, z):
		activation = self.layer["Dense{}".format(i)]['activation']
		if activation == 'sigmoid':
			return self.d_sig(z)
		elif activation == 'linear':
			return np.eye(z.shape[0])

class NN(Activations):
	def __init__(self, X_train, Y_train, X_test, Y_test, layer, lr: float, Epochs: int):
		self.Epochs = Epochs
		self.lr = lr
		self.Y_train = Y_train
		self.X_train = X_train
		self.X_test = X_test
		self.Y_test = Y_test
		self.layer = layer
		self.initialize_weight()
		self.outs = {}
		self.dw = {}
		self.df = {}
		self.da = {}



	def FF(self):
		for i in range(1, len(self.layer) + 1):
			z = np.matmul(self.weights['W{}'.format(i)], self.outs['a{}'.format(i - 1)]) + self.weights['b{}'.format(i)]
			self.outs["a{}".format(i)] = self.activations(i, z)
			self.df['df{}'.format(i)] = self.d_activations(i, z)
		return self.outs

	def chain_rule(self):
		self.da['da{}'.format(len(self.layer))] = self.error
		for i in range(len(self.layer), 1, -1):
			self.da['da{}'.format(i - 1)] = (self.df['df{}'.format(i)] @ self.weights['W{}'.format(i)]).T @  self.da['da{}'.format(i)]
	def back_prob(self):
		self.chain_rule()
		for i in range(1, len(self.layer) + 1):
			self.dw['dW{}'.format(i)] = - (self.df['df{}'.format(i)] @ self.da['da{}'.format(i)]) @ (self.outs['a{}'.format(i - 1)]).T
			self.dw['db{}'.format(i)] = - (self.df['df{}'.format(i)] @ self.da['da{}'.format(i)])
		return self.dw
	def update(self):
		self.dw = self.back_prob()
		for j in range(1, len(self.layer) + 1):
			self.weights['W{}'.format(j)] = self.weights['W{}'.format(j)] - self.lr * self.dw['dW{}'.format(j)]
			self.weights['b{}'.format(j)] = self.weights['b{}'.format(j)] - self.lr * self.dw['db{}'.format(j)]
	def train(self):
		for j in range(self.Epochs):
			for i in (range(self.X_train.shape[0])):
				self.outs['a0'] = self.X_train[i, :].reshape(-1, 1)
				self.output = self.FF()
				self.error = self.Y_train[i, :] - self.outs["a{}".format(len(self.layer))]
				self.update()
			loss = self.MSE(self.Y_train[i, :], self.output['a{}'.format(len(self.layer))])
			print(f"loss: {loss}")
		print(f"W: {self.weights}")
	def predict(self, X, Y):
		a = []
		for j in (range(X.shape[0])):
			self.outs['a0'] = X[j, :].reshape(-1, 1)
			for i in range(1, len(self.layer) + 1):
				z = np.matmul(self.weights['W{}'.format(i)], self.outs['a{}'.format(i - 1)]) + self.weights[
					'b{}'.format(i)]
				self.outs["a{}".format(i)] = np.round(self.activations(i, z)) if self.layer["Dense{}".format(i)]['activation'] == 'sigmoid' else self.activations(i, z)
				a.append(self.outs['a{}'.format(len(self.layer))][0][0])
			print(f"predicting for {X[j, :]}: {self.outs['a{}'.format(len(self.layer))][-1]}, actual value is : {Y[j, :]}")
		return np.asarray(a)
	# def plot(self):
	# 	x = np.linspace(-2, 2, 400)
	# 	y = self.weights['W1'] * x + self.weights['b1']
	# 	# y =
	# 	plt.figure(figsize=(8, 6))
	# 	plt.scatter(self.X_train, self.Y_train, color='blue', label='Features (X_train)')
	# 	plt.scatter(self.X_test, self.Y_test, color='red', label='Features (X_train)')
	# 	plt.plot(x, np.asarray(y).reshape(-1, 1).squeeze(), color='blue', label='y = wx + b')  # Modify the label accordingly
	# 	plt.xlabel('x')
	# 	plt.ylabel('y')
	# 	plt.title('Plot of the line y = 2x + 3')
	# 	plt.legend()
	# 	plt.grid(True)
	# 	plt.show()

	def plot_confusion_matrix(self, true_labels: np.ndarray, predicted_labels: np.ndarray):
		classes = np.unique(np.concatenate((true_labels.squeeze(), predicted_labels)))
		confusion_matrix = np.zeros((len(classes), len(classes)))
		for i in range(len(true_labels)):
			true_class_index = np.where(classes == true_labels[i])[0][0]
			predicted_class_index = np.where(classes == predicted_labels[i])[0][0]
			confusion_matrix[true_class_index][predicted_class_index] += 1

		# Plot confusion matrix as a heatmap
		plt.figure(figsize=(len(classes), len(classes)))
		plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
		plt.title('Confusion Matrix')
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)
		plt.xlabel('Predicted Label')
		plt.ylabel('True Label')

		# Display the confusion matrix values on the heatmap
		for i in range(len(classes)):
			for j in range(len(classes)):
				plt.text(j, i, str(int(confusion_matrix[i, j])), horizontalalignment='center', color='black')

		plt.show()

layers = {
			# 'Dense1': {'n': 1, 'activation': 'sigmoid'},
			'Dense1': {'n': 1, 'activation': 'sigmoid'},
			# 'Dense2': {'n': 1, 'activation': 'sigmoid'}
		}

def get_ds(class1_data, class2_data):
	data = np.concatenate((class1_data.reshape(-1, 1), class2_data.reshape(-1, 1)), axis=0)
	labels = np.concatenate((np.zeros((class1_data.shape[0], 1)), np.ones((class1_data.shape[0], 1))), axis=0)
	data_with_labels = np.concatenate((data, labels), axis=1)
	data_with_labels[:, 0] = (data_with_labels[:, 0] - np.mean(data_with_labels[:, 0])) / (
		np.std(data_with_labels[:, 0]))
	np.random.shuffle(data_with_labels)
	X = data_with_labels[:, :-1]
	y = data_with_labels[:, -1]
	train_split = 0.8
	num_samples = len(data_with_labels)
	num_train_samples = int(train_split * num_samples)
	X_train, X_test = X[:num_train_samples], X[num_train_samples:]
	Y_train, Y_test = y[:num_train_samples], y[num_train_samples:]
	return X_train, Y_train.reshape(-1, 1), X_test, Y_test.reshape(-1, 1)



mean_class1, std_class1 = 10, 0.3
mean_class2, std_class2 = 11, 0.2
num_samples = 100
class1_data = np.random.normal(mean_class1, std_class1, num_samples)
class2_data = np.random.normal(mean_class2, std_class2, num_samples)



X_train, Y_train, X_test, Y_test = get_ds(class1_data, class2_data)
model = NN(X_train, Y_train, X_test, Y_test, layers, lr=0.01, Epochs=150)


model.train()
predicted = model.predict(X_test, Y_test)
model.plot_confusion_matrix(Y_test, predicted)
predicted = model.predict(X_train, Y_train)
model.plot_confusion_matrix(Y_train, predicted)
# model.plot()
