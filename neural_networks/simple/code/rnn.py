'''
A simple recurrent neural network implemented in python.
We can use this to train but it will take ages to complete.
It's mostly for demonstration purposes and what's it supposedly can do

@Author: Rohith Uppala
'''

import itertools
import nltk
import sys
import csv
import pdb
import numpy as np
from datetime import datetime

class RNNNumpy:
	def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
		# Assign instance variables
		self.word_dim = word_dim
		self.hidden_dim = hidden_dim
		self.bptt_truncate = bptt_truncate
		# Randomly initialize the network parameters
		self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
		self.V = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (word_dim, hidden_dim))
		self.W = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))
		self.vocabulary_size = word_dim
		self.unknown_token = "UNKNOWN_TOKEN"
		self.sentence_start_token = "SENTENCE_START"
		self.sentence_end_token = "SENTENCE_END"

	def softmax(self, x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()

	def forward_propogation(self, x):
		T = len(x)
		s = np.zeros((T+1, self.hidden_dim))
		s[-1] = np.zeros(self.hidden_dim)
		o = np.zeros((T, self.word_dim))
		for t in np.arange(T):
			s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
			o[t] = self.softmax(self.V.dot(s[t]))
		return [o, s]

	def predict(self, x):
		o, s = self.forward_propogation(x)
		return np.argmax(o, axis=1)

	def calculate_loss_value(self, x, y):
		L = 0
		for i in np.arange(len(y)):
			o, s = self.forward_propogation(x[i])
			correct_word_predictions = o[np.arange(len(y[i])), y[i]]
			L += -1 * np.sum(np.log(correct_word_predictions))
		return L

	def calculate_loss(self, x, y):
		N = np.sum((len(y_i) for y_i in y))
		return self.calculate_loss_value(x, y) / N


        # This is the complicated piece in the whole neural network process
        # For reference, my notebook and this articl https://github.com/go2carter/nn-learn/blob/master/grad-deriv-tex/rnn-grad-deriv.pdf
        def bptt(self, x, y):
            T = len(y)
            o, s = self.forward_propogation(x)
            dLdU = np.zeros(self.U.shape)
            dLdV = np.zeros(self.V.shape)
            dLdW = np.zeros(self.W.shape)
            delta_o = o
            delta_o[np.arange(len(y)), y] -= 1
            for t in np.arange(T)[::-1]:
                dLdV += np.outer(delta_o[t], s[t].T)
                delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
                for bptt_step in np.arange(max(0, t - self.bptt_truncate), t+1)[::-1]:
                    dLdW += np.outer(delta_t, s[bptt_step - 1])
                    dLdU[:, x[bptt_step]] += delta_t
                    delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[bptt_step-1] ** 2))
            return [dLdU, dLdV, dLdW]


        def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
            bptt_gradients = self.bptt(x, y)
            model_parameters = ['U', 'V', 'W']
            for pidx, pname in enumerate(model_parameters):
                parameter = operator.attrgetter(pname)(self)
                print "Performing gradient check for parameter %s with size %d." %(pname, np.prod(parameter.shape))
                it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    ix = it.multi_index
                    original_value = parameter[ix]
                    parameter[ix] = original_value + h
                    gradplus = self.calculate_loss_value([x], [y])
                    parameter[ix] = original_value - h
                    gradminus = self.calculate_loss_value([x], [y])
                    estimated_gradient = (gradplus - gradminus) / (2 * h)
                    parameter[ix] = original_value
                    backprop_gradient = bptt_gradients[pidx][ix]
                    relative_error = np.abs(backprop_gradient - estimated_gradient) / (np.abs(backprop_gradient) + np.abs(estimated_gradient))
                    if relative_error >= error_threshold:
                        print "Gradient check Error: parameter=%s ix=%s" %(pname, ix)
                        print "+h loss: %f" %gradplus
                        print "-h loss: %f" %gradminus
                        print "Estimated gradient: %f" %estimated_gradient
                        print "Backpropogation gradient: %f" %backprop_gradient
                        print "Relative error: %f" %relative_error
                        return
                    it = it.iternext
                print "gradient check for parameter %s passed." %(pname)


        def numpy_sgd_step(self, x, y, learning_rate):
            dLdU, dLdV, dLdW = self.bptt(x, y)
            self.U -= learning_rate * dLdU
            self.V -= learning_rate * dLdV
            self.W -= learning_rate * dLdW


        def train_with_sgd(self, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
            losses = []
            num_examples_seen = 0
            for epoch in range(nepoch):
                if epoch % evaluate_loss_after == 0:
                    loss = model.calculate_loss(X_train, y_train)
                    losses.append((num_examples_seen, loss))
                    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print "%s: Loss after num_examples_seen=%d epoch=%d: %f" %(time, num_examples_seen, epoch, loss)
                    if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                        learning_rate = learning_rate * 0.5
                        print "Setting learning rate to %f" %(learning_rate)
                    sys.stdout.flush()
                for i in range(len(y_train)):
                    self.numpy_sgd_step(X_train[i], y_train[i], learning_rate)
                    num_examples_seen += 1

	def prepare_training_data(self):
		print "Reading CSV file..."
		with open('../data/reddit-comments.csv', 'rb') as f:
			reader = csv.reader(f, skipinitialspace=True)
			reader.next()
			sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
			sentences = ["%s %s %s" % (self.sentence_start_token, x, self.sentence_end_token) for x in sentences]
		print "Parsed %d sentences." % (len(sentences))
		tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
		word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
		print "Found %d unique words tokens." % len(word_freq.items())
		vocab = word_freq.most_common(self.vocabulary_size-1)
		index_to_word = [x[0] for x in vocab]
		index_to_word.append(self.unknown_token)
		word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
		print "Using vocabulary size %d." % self.vocabulary_size
		print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
		for i, sent in enumerate(tokenized_sentences):
			tokenized_sentences[i] = [w if w in word_to_index else self.unknown_token for w in sent]
		print "\nExample sentence: '%s'" % sentences[0]
		print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
		X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
		Y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
		return X_train, Y_train

if __name__ == "__main__":
	model = RNNNumpy(8000)
        X_train, y_train = model.prepare_training_data()
        losses = model.train_with_sgd(X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)
