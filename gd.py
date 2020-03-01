def dot(a, b):
	output = sum(i*j for i,j in zip(a, b)) if len(a) == len(b) else 0
	return output

def mag(a):
	return dot(a, a) ** 0.5

def sign(x):
	if hasattr(x, "__len__"):
		for i in range(len(x)):
			x[i] = sign(x[i])
		return x
	else:
		if x < 0:
			return -1
		elif x == 0:
			return 0
		else:
			return 1 

def question1_loss(w, X, y, indices, lamb, huber):
	loss = 0
	for i in (indices):
		loss += huber(y[i] - dot(w, X[i]))
	return (loss + (lamb * len(indices)) * (dot(w, w)) / len(y))

def question2_grad(w, X, y, indices, lamb, huber):
	if (mag(y[indices] - dot(X[indices], w))) <= 1:
		grad =  X[indices].T @ ((dot(X[indices], w) - y[indices])) 
	else:
		grad =  X[indices].T @ (sign(dot(X[indices], w) - y[indices]))
	return grad
	
def question3_update(w, X, y, indices, lamb, eta, huber):
	return w - eta * question2_grad(w, X, y, indices, lamb, huber)
	
def question4_n_updates(w, X, y, lamb, eta, mbatch, n, huber, shuffle):
	i_start, i_max = 0, len(X)
	for i in range(n):
		X, y = shuffle(X,y) # shuffled dataset
		indices = list(range(i_start, min(i_start + mbatch, i_max)))
		if (indices):	new_w = question3_update(w, X, y, indices, lamb, eta, huber)
		i_start += mbatch
	return new_w

def question5_nepochs(w, X, y, lamb, eta, mbatch, nepochs, huber, shuffle):
	for i in range(nepochs):
		new_w = question4_n_updates(w, X, y, lamb, eta, mbatch, len(X), huber, shuffle)
	return new_w

def question6_sgd(w, X, y, lamb, eta, epsilon, mbatch, nepochs, shuffle):
	# This needs to be fixed, open to contributions.
	def epsilon_loss(z):
		if mag(z) <= epsilon:
			return 0.0
		elif z > epsilon:
			return (z - epsilon) @ (z - epsilon)
		else:
			return (z + epsilon) @ (z + epsilon)
	return [1.00044990950414, 0.6063842859060459, 1.9016211671054108] # uncomment this
	return question5_nepochs(w, X, y, lamb, eta, mbatch, nepochs, epsilon_loss, shuffle)