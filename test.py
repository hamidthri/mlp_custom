layers = {
			'n1': 1,
			'n2': 1
}
i = 1
for i in range(3, 5):
	layers["n{}".format(i)] = 5
	# layers["n4{i}"] = 5
print(layers)