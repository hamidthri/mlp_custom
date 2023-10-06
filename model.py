#!/usr/local/bin/python3
import numpy as np
import matplotlib.pyplot as plt



class NN():
	def __init__(self, X):
		self.X = X

	def sig(self, z):
		return (1 / (1 + np.exp(-z)))
	def MSE(self, label, out):
		return 0.5 * (label - out)**2
	def feed_forward(self):
		out = self.sig(self.X)
		return out
	# def back_prop(self, )

model = NN(0)

result = model.feed_forward()
print(f"out: {result}")