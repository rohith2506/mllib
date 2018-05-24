'''
Implementation of Long Short Term Memory Network in Python

This Network contains core properties:
1) Forward Propogation
2) Backward Propogation

I intentionally avoided SGD as it's out of scope of LSTM and it's generally applied to the backprop losses

For References:
http://arunmallya.github.io/writeups/nn/lstm/index.html
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

@Author: Rohith Uppala
'''

import numpy as np


class LSTM:

	'''
	This will initialize all the independent variables required for LSTM network
	'''
	def __init__(self, word_dim, hidden_dim=100):
		self.word_dim = word_dim
		self.hidden_dim = hidden_dim
		self.Uf = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, word_dim))
		self.Wf = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, hidden_dim))
		self.Bf = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, 1))
		
		self.Ui = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, word_dim))
		self.Wi = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, hidden_dim))
		self.Bi = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, 1))

		self.Ua = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, word_dim))
		self.Wa = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, hidden_dim))
		self.Ba = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, 1))

		self.Uo = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, word_dim))
		self.Wo = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, hidden_dim))		
		self.Bo = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, 1))

		self.V = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim), (hidden_dim, hidden_dim))


	'''
	Softmax(x) = (e^x / sigma(e^x))
	'''
	def softmax(self, x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()

	def tanh(self, x):
		e_x = np.exp(x - np.max(x))
		return (((e_x * e_x) - 1) * 1.0 / ((e_x * e_x) + 1))

	def sigma(self, x):
		e_x = np.exp(x - np.max(x))
		return 1 / (1 + np.exp(e_x))

	def acapt(self, x, t):
		return self.Uf.dot(x[t]) + self.Wf.dot(h[t-1]) + self.Bf

	'''
	Forward propogation
	'''
	def forward_propogation(self, x):
		T = len(x)
		h = np.zeros((T, self.hidden_dim)) #  h(t) with array of arrays which has dimension (d x 1)
		h[-1] = np.zeros(self.hidden_dim)
		c = np.zeros((T, self.hidden_dim)) # c(t) with array of arrays which has dimension (d x 1)
		c[-1] = np.random.uniform(-np.sqrt(1.0/word_dim), np.sqrt(1.0/word_dim))

		y = np.zeros((T, self.word_dim)) # y(t) with array of arrays which has dimension (d x 1)

		# Now, it's time to define those input layers
		f = np.zeros((T, self.hidden_dim)) # f(t)
		i = np.zeros((T, self.hidden_dim)) # i(t)
		a = np.zeros((T, self.hidden_dim)) # a(t)
		o = np.zeros((T, self.hidden_dim)) # o(t)

		for t in np.range(T):
			f[t] = self.sigma(self.Uf.dot(x[t]) + self.Wf.dot(h[t-1]) + self.Bf)
			i[t] = self.sigma(self.Ui.dot(x[t]) + self.Wi.dot(h[t-1]) + self.Bi)
			a[t] = self.tanh(self.Ua.dot(x[t]) + self.Wa.dot(h[t-1]) + self.Ba)
			o[t] = self.sigma(self.Uo.dot(x[t]) + self.Wo.dot(h[t-1]) + self.Bo)
			c[t] = np.multiply(i[t], a[t]) + np.multiply(f[i], s[t-1])
			h[t] = np.multiply(o[t], self.tanh(c[t]))
			y[t] = self.softmax(self.V.dot(h[t]))

		return [f, i, a, o, c, h, y]

	'''
	Predict the required variable based on argmax
	'''
	def predict(self, x):
		f, i, a, o, c, h, y = self.forward_propogation(x)
		return np.argmax(x, axis=0)


	'''
	Cross Entropy Loss Function
	'''
	def calculate_loss_value(self, x, y):
		L = 0
		for i in np.arange(len(y)):
			o, s = self.forward_propogation(x[i])
			correct_word_predictions = o[np.arange(len(y[i])), y[i]]
			L += -1 * np.sum(np.log(correct_word_predictions))
		return L

	'''
	Backpropogation Algorithm
	'''
	def backpropogation(self, x, y):
		f, i, o, o, c, h, y = self.forward_propogation(x)

		dLdV = np.zeros(self.V.shape)

		dLdUf = np.zeros(self.Uf.shape)
		dLdWf = np.zeros(self.Wf.shape)
		dLdBf = np.zeros(self.Bf.shape)
		
		dLdUi = np.zeros(self.Ui.shape)
		dLdWi = np.zeros(self.Wi.shape)
		dLdBi = np.zeros(self.Bi.shape)

		dLdUa = np.zeros(self.Ua.shape)
		dLdWa = np.zeros(self.Wa.shape)
		dLdBa = np.zeros(self.Ba.shape)

		dLdUo = np.zeros(self.Uo.shape)
		dLdWo = np.zeros(self.Wo.shape)
		dLdBo = np.zeros(self.Bo.shape)

		# Find the loss given y
		delta_o = o
		delta_o[np.arange(len(y), y)] -= 1

		for t in np.range(T)[::-1]:
			# Output Layer
			dLdV += np.outer(delta_o[t], h[t].T)

			# Backward Pass: Output
			delta_ht = np.outer((y[t] - o[t]), self.V.T)
			delta_ot = np.multiply(delta_ht, self.tanh(c[t]))
			delta_ct = np.multiply(delta_ht, o[t], (1 - (self.tanh(c[t]) * self.tanh(c[t]))))

			# Backward Pass: LSTM Memory cell update
			delta_it = np.multiply(delta_ct, i[t])
			delta_at = np.multiply(delta_ct, a[t])
			delta_ft = np.multiply(delta_ct, f[t])
			delta_ct_1 = np.multiply(delta_ct, c[t-1])

			# Backward Pass: Input and Gate Computation 
			delta_acapt = np.multiply(delta_at, (1 - (self.acapt(x) * self.acapt(x))))
			delta_icapt = np.multiply(delta_it, i[t], (1 - i[t]))
			delta_fcapt = np.multiply(delta_ft, f[t], (1 - f[t]))
			delta_ocapt = np.multiply(delta_ot, o[t], (1 - o[t]))

			# Backward Pass: Independent Variables
			dLdUf += np.outer(delta_fcapt, x[t].T)
			dLdWf += np.outer(delta_fcapt, h[t-1].T)
			dLdBf += delta_fcapt
			dLdUi += np.outer(delta_icapt, x[t].T)
			dLdWi += np.outer(delta_icapt, h[t-1].T)
			dLdBi += delta_icapt
			dLdUo += np.outer(delta_ocapt, x[t].T)
			dLdWo += np.outer(delta_ocapt, h[t-1].T)
			dLdBo += delta_ocapt
			dLdUa += np.outer(delta_acapt, x[t].T)
			dLdWa += np.outer(delta_acapt, h[t-1].T)
			dLdBa += delta_acapt

		return [dLdWf, dLdUf, dLdBf, dLdWi, dLdUi, dLdBi, dLdWo, dLdUo, dLdBo, dLdWa, dLdUa, dLdBa]

