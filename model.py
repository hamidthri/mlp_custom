
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
class NN():
	def __init__(self, X, Y, layer, lr, Epochs):
		self.Epochs = Epochs
		self.lr = lr
		self.Y = Y
		self.X = X
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
	
	def predict(self):
		for j in (range(self.X.shape[0])):
			inp = self.X[j, :].reshape(-1, 1)
			self.outs['a0'] = inp
			for i in range(1, len(self.layer) + 1):
				z = np.matmul(self.weights['W{}'.format(i)], self.outs['a{}'.format(i - 1)]) + self.weights['b{}'.format(i)]
				self.outs["a{}".format(i)] = self.activations(i, z)
			print(f"predicting for {X[j, :]}: {self.outs['a{}'.format(len(self.layer))][-1]}")
		# return self.outs[-1]
	def FF(self):
		for i in range(1, len(self.layer) + 1):
			z = np.matmul(self.weights['W{}'.format(i)], self.outs['a{}'.format(i - 1)]) + self.weights['b{}'.format(i)]
			self.outs["a{}".format(i)] = self.activations(i, z)
			self.df['df{}'.format(i)] = self.d_activations(i, z)
		return [self.outs["a{}".format(i)] for i in range(len(self.outs))]

	def chain_rule(self, error):
		self.da['da{}'.format(len(self.layer))] = error
		for i in range(len(self.layer), 1, -1):
			self.da['da{}'.format(i - 1)] = (self.df['df{}'.format(i)] @ self.weights['W{}'.format(i)]).T @  self.da['da{}'.format(i)]
	def back_prob(self, i, inp):
		error = self.Y[i, :] - self.outs["a{}".format(len(self.layer))]
		self.chain_rule(error)
		for i in range(1, len(self.layer) + 1):
			self.dw['dW{}'.format(i)] = - (self.lr * self.df['df{}'.format(i)] @ self.da['da{}'.format(i)]) @ (self.outs['a{}'.format(i - 1)]).T
			self.dw['db{}'.format(i)] = - (self.lr * self.df['df{}'.format(i)] @ self.da['da{}'.format(i)])
		return self.dw
	def update(self):
		for j in range(1, len(self.layer) + 1):
			self.weights['W{}'.format(j)] = self.weights['W{}'.format(j)] - self.dw['dW{}'.format(j)]
			self.weights['b{}'.format(j)] = self.weights['b{}'.format(j)] - self.dw['db{}'.format(j)]
	def train(self):
		self.weight()
		print(f"W: {self.weights}")
		for j in range(self.Epochs):
			for i in (range(self.X.shape[0])):
				inp = self.X[i, :].reshape(-1, 1)
				self.outs['a0'] = inp
				self.output = self.FF()
				self.dw = self.back_prob(i, inp)
				self.update()
				# print(f"outs: {self.out}")
				
				
			loss = self.MSE(self.Y[i, :], self.output[-1])
			print(f"loss: {loss}")
		print(f"W: {self.weights}")

X = np.random.random((1000, 1))

Y = 1 * X + 1

layers = {
			# 'Dense1': {'n': 1, 'activation': 'sigmoid'},
			# 'Dense2': {'n': 6, 'activation': 'sigmoid'},
			'Dense1': {'n': 1, 'activation': 'linear'}
		}

model = NN(X, Y, layers, lr=0.1, Epochs=15)


result = model.train()
# model.predict()






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
