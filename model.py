
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
class NN():
	def __init__(self, X_train, Y_train, X_test, Y_test, layer, lr, Epochs):
		self.Epochs = Epochs
		self.lr = lr
		self.Y_train = Y_train
		self.X_train = X_train
		self.X_test = X_test
		self.Y_test = Y_test


		self.layer = layer

		self.weights = {}
		self.outs = {}
		self.dw = {}
		self.df = {}
		self.da = {}

	def weight(self):
		for i in range(1, len(self.layer) + 1):
			self.weights["W{}".format(i)] = np.random.uniform(-1, 1, (self.layer["Dense{}".format(i)]["n"], X.shape[1] if i == 1 else self.layer["Dense{}".format(i - 1)]["n"]))
			self.weights["b{}".format(i)] = np.random.uniform(-1, 1, (self.layer["Dense{}".format(i)]["n"], 1))


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

	def FF(self):
		for i in range(1, len(self.layer) + 1):
			z = np.matmul(self.weights['W{}'.format(i)], self.outs['a{}'.format(i - 1)]) + self.weights['b{}'.format(i)]
			self.outs["a{}".format(i)] = self.activations(i, z)
			self.df['df{}'.format(i)] = self.d_activations(i, z)
		return self.outs

	def chain_rule(self, error):
		self.da['da{}'.format(len(self.layer))] = error
		for i in range(len(self.layer), 1, -1):
			self.da['da{}'.format(i - 1)] = (self.df['df{}'.format(i)] @ self.weights['W{}'.format(i)]).T @  self.da['da{}'.format(i)]
	def back_prob(self, i):
		error = self.Y_train[i, :] - self.outs["a{}".format(len(self.layer))]
		self.chain_rule(error)
		for i in range(1, len(self.layer) + 1):
			self.dw['dW{}'.format(i)] = - (self.df['df{}'.format(i)] @ self.da['da{}'.format(i)]) @ (self.outs['a{}'.format(i - 1)]).T
			self.dw['db{}'.format(i)] = - (self.df['df{}'.format(i)] @ self.da['da{}'.format(i)])
		return self.dw
	def update(self):
		for j in range(1, len(self.layer) + 1):
			self.weights['W{}'.format(j)] = self.weights['W{}'.format(j)] - self.lr * self.dw['dW{}'.format(j)]
			self.weights['b{}'.format(j)] = self.weights['b{}'.format(j)] - self.lr * self.dw['db{}'.format(j)]
	def train(self):
		self.weight()
		print(f"W: {self.weights}")
		for j in range(self.Epochs):
			for i in (range(self.X_train.shape[0])):
				inp = self.X_train[i, :].reshape(-1, 1)
				self.outs['a0'] = inp
				self.output = self.FF()
				self.dw = self.back_prob(i)
				self.update()
				# print(f"outs: {self.out}")
			loss = self.MSE(self.Y_train[i, :], self.output['a{}'.format(len(self.layer))])
			print(f"loss: {loss}")
		print(f"W: {self.weights}")
	def predict(self):
		a = []
		for j in (range(self.X_test.shape[0])):
			inp = self.X_test[j, :].reshape(-1, 1)
			self.outs['a0'] = inp
			for i in range(1, len(self.layer) + 1):
				z = np.matmul(self.weights['W{}'.format(i)], self.outs['a{}'.format(i - 1)]) + self.weights[
					'b{}'.format(i)]
				self.outs["a{}".format(i)] = np.round(self.activations(i, z)) if self.layer["Dense{}".format(i)]['activation'] == 'sigmoid' else self.activations(i, z)
				a.append(self.outs['a{}'.format(len(self.layer))])
			print(f"predicting for {self.X_test[j, :]}: {self.outs['a{}'.format(len(self.layer))][-1]}, actual value is : {self.Y_test[j, :]}")
		return a
	def plot(self):
		x = np.linspace(-2, 2, 400)
		# y = self.predict()
		# y =
		plt.figure(figsize=(8, 6))
		plt.scatter(self.X_train, self.Y_train, color='blue', label='Features (X_train)')
		plt.scatter(self.X_test, self.Y_test, color='red', label='Features (X_train)')
		plt.plot(x, np.asarray(y).reshape(-1, 1).squeeze(), color='blue', label='y = wx + b')  # Modify the label accordingly
		plt.xlabel('x')
		plt.ylabel('y')
		plt.title('Plot of the line y = 2x + 3')
		plt.legend()
		plt.grid(True)
		plt.show()


X = np.random.random((1000, 1))

# Y = 1 * X + 1

layers = {
			# 'Dense1': {'n': 1, 'activation': 'sigmoid'},
			'Dense1': {'n': 8, 'activation': 'sigmoid'},
			'Dense2': {'n': 1, 'activation': 'sigmoid'}
		}

def get_ds():
	mean_class1, std_class1 = 10, 0.5
	mean_class2, std_class2 = 13, 0.5

	# Generate random data points for each class
	num_samples = 100
	class1_data = np.random.normal(mean_class1, std_class1, num_samples)
	class2_data = np.random.normal(mean_class2, std_class2, num_samples)

	data = np.concatenate((class1_data.reshape(-1, 1), class2_data.reshape(-1, 1)), axis=0)
	labels = np.concatenate((np.zeros((num_samples, 1)), np.ones((num_samples, 1))), axis=0)
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


X_train, Y_train, X_test, Y_test = get_ds()
model = NN(X_train, Y_train, X_test, Y_test, layers, lr=0.01, Epochs=150)


model.train()
model.predict()
model.plot()






# def get_ds(m: int):
#     """Returns a dataset of m numbers ranging from 0 to 20"""
#     x = np.random.choice(m * 2, (m, 1), replace=False)
#     y = (x * 2) + 1
#     return x, y


#     def train(self):
#         for i in range(self.epoch):
#             for j in range(self.num_train):
#                 input_train = self.data_train[j, :self.n0]
#                 target_train = self.data_train[j, self.n0]
#                 net1 = input_train @ self.w1
#                 o1 = self.sig(net1)
#                 net2 = o1 @ self.w2
#                 o2 = net2
#                 error = target_train - o2
#                 self.w1 = self.w1 - o2.reshape(self.n0, 1) @ (
#                     self.lr * -1 * 1 * error * self.w2.T @ np.diagflat(o1 * (1 - o1)))
#                 self.w2 = self.w2 - o1.reshape(self.n1, 1) * self.lr * -1 * 1 * error

#             error_train_epoch = 0
#             self.output_train = []
#             target_trainn = []
#             for j in range(self.num_train):
#                 input_train = self.data_train[j, :self.n0]
#                 target_train = self.data_train[j, self.n0]
#                 net1 = input_train @ self.w1
#                 o1 = self.sig(net1)
#                 net2 = o1 @ self.w2
#                 o2 = net2
#                 error = target_train - o2
#                 self.error_train.append(error)
#                 self.output_train.append(o2[0])
#                 target_trainn.append(target_train)
#                 error_train_epoch += error ** 2
#             mse_train_epoch = 0.5 * error_train_epoch / self.num_train
#             self.mse_train.append(mse_train_epoch)

#     def plot_results(self):
#         fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#         axs[0].plot(self.mse_test, 'b-')  # Use 1-dimensional index here
#         axs[0].set_title('Test MSE')

#         axs[1].plot((np.array(self.mse_train).reshape(-1, 1)), 'r-')
#         axs[1].set_title('Train MSE')

#         plt.tight_layout()
#         plt.show()

#     def sig_(self, x):
#         return 1 / (1 + np.exp(-x))

#     def predict(self, x: np.ndarray):
#         z1 = (x @ self.w1)  # Correct the matrix multiplication here
#         a1 = self.sig_(z1)
#         z2 = (a1 @ self.w2)  # Correct the matrix multiplication here
#         a2 = z2
#         return a2
# if name == "main":
#     nn = NeuralNetwork()
#     nn.train()
#     nn.plot_results()
#     x_input = np.array([[5]])  # You need to pass the input as a numpy array
#     predicted_output = nn.predict(x_input)
#     print(predicted_output)
    #
# if name == "main":
#  x, y = get_ds(200)
#  ds = np.hstack((x, y))
#  print(ds)
#
#  model = Model(ds=ds, epochs=200, lr=0.001)
#  model.train()
#  print(model.predict(5))
#  model.show_history()
