#!/usr/local/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
class NN():
	def __init__(self, X, W):
		self.X = X

	def layer(self, W):

	def sig(self, z):
		return (1 / (1 + np.exp(-z)))

	def relu(self, z):
		return np.max(0, z)

	def d_sig(self, z):
		return (self.sig(z) * (1 - self.sig (z)))

	def MSE(self, label, out):
		return 0.5 * (label - out)**2

	def feed_forward(self):
		a = self.sig(self.X)
		return a
	def my_dense(A_in, W, b, f):
		z = np.matmul(A_in, W) + b
		A_out = f(z)
		return A_out


	def train(self, X, Y, Epochs):
		for j in range(Epochs):
			for i in tqdm(X.shape[0]):
				inp = X[:, i]
				out = self.feed_forward(inp)
				error = Y - out
				# loss = self.MSE(Y, out)

	# def back_prop(self, )

model = NN(0)

result = model.feed_forward()
print(f"out: {result}")