# coding: utf-8
from rnnmath import *
import numpy as np
from sys import stdout
import time

class RNN(object):
	'''
	This class implements Recurrent Neural Networks.

	You should implement code in the following functions:
		predict				->	predict an output sequence for a given input sequence
		acc_deltas			->	accumulate update weights for the RNNs weight matrices, standard Back Propagation
		acc_deltas_bptt		->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time
		compute_loss 		->	compute the (cross entropy) loss between the desired output and predicted output for a given input sequence
		compute_mean_loss	->	compute the average loss over all sequences in a corpus
		generate_sequence	->	use the RNN to generate a new (unseen) sequnce

	Do NOT modify any other methods!
	Do NOT change any method signatures!
	'''

	def __init__(self, vocab_size, hidden_dims):
		'''
		initialize the RNN with random weight matrices.

		DO NOT CHANGE THIS

		vocab_size		size of vocabulary that is being used
		hidden_dims		number of hidden units
		'''
		self.vocab_size = vocab_size
		self.hidden_dims = hidden_dims

		# matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
		self.U = np.random.randn(self.hidden_dims, self.hidden_dims)*np.sqrt(0.1)
		self.V = np.random.randn(self.hidden_dims, self.vocab_size)*np.sqrt(0.1)
		self.W = np.random.randn(self.vocab_size, self.hidden_dims)*np.sqrt(0.1)

		# matrices to accumulate weight updates
		self.deltaU = np.zeros((self.hidden_dims, self.hidden_dims))
		self.deltaV = np.zeros((self.hidden_dims, self.vocab_size))
		self.deltaW = np.zeros((self.vocab_size, self.hidden_dims))
	def one_hot_enco(self,num,size):
		one_hot=np.zeros(size)
		one_hot[num]=1
		return one_hot

	def apply_deltas(self, learning_rate):
		'''
		update the RNN's weight matrices with corrections accumulated over some training instances

		DO NOT CHANGE THIS

		learning_rate	scaling factor for update weights
		'''
		# apply updates to U, V, W
		self.U += learning_rate*self.deltaU
		self.W += learning_rate*self.deltaW
		self.V += learning_rate*self.deltaV

		# reset matrices
		self.deltaU.fill(0.0)
		self.deltaV.fill(0.0)
		self.deltaW.fill(0.0)

	def predict(self, x):
		'''
		predict an output sequence y for a given input sequence x

		x	list of words, as indices, e.g.: [0, 4, 2]

		returns	y,s
		y	matrix of probability vectors for each input word
		s	matrix of hidden layers for each input word

		'''

		# matrix s for hidden states, y for output states, given input x.
		#rows correspond to times t, i.e., input words
		# s has one more row, since we need to look back even at time 0 (s(t=0-1) will just be [0. 0. ....] )
		s = np.zeros((len(x)+1, self.hidden_dims))
		y = np.zeros((len(x), self.vocab_size))

		for t in range(len(x)):
			##########################
			# --- your code here --- #
			##########################
			s[t] = sigmoid(self.V.dot(make_onehot(x[t],3).T)+self.U.dot(s[t-1].T))
			y[t] = softmax(self.W.dot(s[t].T))
		return y,s

	def compute_loss(self, x, d):
		'''
		compute the loss between predictions y for x, and desired output d.

		first predicts the output for x using the RNN, then computes the loss w.r.t. d

		x		list of words, as indices, e.g.: [0, 4, 2]
		d		list of words, as indices, e.g.: [4, 2, 3]

		return loss		the combined loss for all words
		'''

		loss = 0.
		softm,activa=self.predict(x)
		for i in range(len(softm)):
			loss+=(-np.array(make_onehot(d[i], self.vocab_size)).dot(np.log(softm[i]).T))
		##########################
		# --- your code here --- #
		##########################
		return loss

	def compute_mean_loss(self, X, D):
		'''
		compute the mean loss between predictions for corpus X and desired outputs in corpus D.

		X		corpus of sentences x1, x2, x3, [...], each a list of words as indices.
		D		corpus of desired outputs d1, d2, d3 [...], each a list of words as indices.

		return mean_loss		average loss over all words in D
		'''

		mean_loss = 0.
		totlen = 0.
		##########################
		# --- your code here --- #
		##########################
		for i in range(len(X)):
			mean_loss+=self.compute_loss(X[i],D[i])
			totlen+=len(X[i])
		mean_loss/=totlen
		return mean_loss
	def acc_deltas(self, x, d, y, s):
		'''
		accumulate updates for V, W, U
		standard back propagation

		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time

		x	list of words, as indices, e.g.: [0, 4, 2]
		d	list of words, as indices, e.g.: [4, 2, 3]
		y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
			should be part of the return value of predict(x)
		s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
			should be part of the return value of predict(x)

		no return values
		'''
		V_sum=np.zeros((self.hidden_dims,self.vocab_size))
		U_sum=np.zeros((self.hidden_dims,self.hidden_dims))
		W_sum=np.zeros((self.vocab_size,self.hidden_dims))
		for t in reversed(range(len(x))):
			##########################
			# --- your code here --- #
			##########################
			W1_loss=((make_onehot(d[t], self.vocab_size))-y[t])
			s_net_new = np.multiply(s[t],((s[t]*-1)+1.))
			WTL= self.W.T.dot(W1_loss)
			inputt=make_onehot(x[t], self.vocab_size)
			V_sum +=np.outer(WTL*s_net_new,inputt)
			U_sum += np.outer(WTL*s_net_new,s[t-1])
			W_sum += np.outer(W1_loss,s[t])
		self.deltaV=  V_sum
		self.deltaW = W_sum
		self.deltaU = U_sum

	def acc_deltas_bpt1t(self, x, d, y, s, steps):
		'''
		accumulate updates for V, W, U
		back propagation through time (BPTT)

		this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time

		x		list of words, as indices, e.g.: [0, 4, 2]
		d		list of words, as indices, e.g.: [4, 2, 3]
		y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
				should be part of the return value of predict(x)
		s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
				should be part of the return value of predict(x)
		steps	number of time steps to go back in BPTT

		no return values
		'''
		U_sum=np.zeros((self.hidden_dims,self.hidden_dims))
		V_sum = np.zeros((self.hidden_dims, self.vocab_size))
		W_sum = np.zeros((self.vocab_size, self.hidden_dims))
		U_tem=np.zeros((self.hidden_dims,self.hidden_dims))
		V_tem = np.zeros((self.hidden_dims, self.vocab_size))
		W_tem = np.zeros((self.vocab_size, self.hidden_dims))
		# s_net_recurr = np.ones(s[0].shape)
		for t in reversed(range(len(x))):
			s_net_recurr = np.ones(s[0].shape)
			##########################
			# --- your code here --- #
			##########################
			W1_loss = ((make_onehot(d[t], self.vocab_size)) - y[t])
			W_sum += np.outer(W1_loss, s[t])
			s_net_new = np.multiply(s[t], ((s[t] * -1) + 1.))
			WTL= self.W.T.dot(W1_loss)
			inputt=make_onehot(x[t], self.vocab_size)
			V_tem=np.outer(WTL*s_net_new,inputt)
			U_tem= np.outer(WTL*s_net_new,s[t-1])
			V_sum +=V_tem
			U_sum +=U_tem
			for step in range(1,steps+1):
				inputt_minus_r = make_onehot(x[t-step], self.vocab_size)
				# W_t_r_1_loss = ((make_onehot(d[t-step+1], self.vocab_size)) - y[t-step+1])
				# WTL_t_r_1 = self.W.T.dot(W_t_r_1_loss)
				# s_net_new = np.multiply(s[t-step+1], ((s[t-step+1] * -1) + 1.))
				s_net_new_minus_steps = np.multiply(s[t-step], ((s[t-step] * -1) + 1.))
				# cancha_in_p = (self.U.T.dot(WTL_t_r_1*s_net_new)) * s_net_new_minus_steps
				# V_sum +=np.outer(cancha_in_p,inputt_minus_r)
				# U_sum += np.outer(cancha_in_p,s[t-step-1])
				first_recu = WTL*s_net_new
				new_U = self.U**step
				# print 'aaa',s_net_recurr
				for i in range(1,step+1):
					s_net_recurr *= np.multiply(s[t-i], ((s[t-i] * -1) + 1.))
				# for i in range(1,steps+1):
				# 	for j in range(i,steps+1):
				# 		first_recu
				V_sum += np.outer((new_U.T.dot(first_recu)) * s_net_recurr,inputt_minus_r)
				U_sum += np.outer((new_U.T.dot(first_recu) * s_net_recurr),np.multiply(s[t-step], ((s[t-step] * -1) + 1.)))
				new______ = first_recu
				for i in range(0,step):
					new______=self.U.T.dot(new______)* np.multiply(s[t-i], ((s[t-i] * -1) + 1.))
			print V_sum



			# W_t_r_1_loss = ((make_onehot(d[t], self.vocab_size)) - y[t])
			# WTL_t_r_1 = self.W.T.dot(W_t_r_1_loss)
			# first_delta = WTL_t_r_1*np.multiply(s[t], ((s[t] * -1) + 1.))
			# X_input = make_onehot(x[t-steps],self.vocab_size)
			# S_t_steps_1 = s[t-steps-1]
			# new_U = self.U**steps
			# for i in range(1,steps+1):
			# 	s_net_recurr *= np.multiply(s[t-steps], ((s[t-steps] * -1) + 1.))
			# # print s_net_recurr
			# print new_U
			# for i in range(1,steps+1):
			# 	for j in range(1,i+1):
			# 		# deltaa_t_tau *=
			# 		print 'a'



		self.deltaV = V_sum
		self.deltaU = U_sum
		self.deltaW = W_sum

	def acc_deltas_bptt(self, x, d, y, s, steps):
		net_out_grad = np.ones(len(x))
		net_in_grad = np.array([s_t * (np.ones(len(s_t)) - s_t) for s_t in s])
		sum_deltaW = np.zeros((self.vocab_size, self.hidden_dims))
		sum_deltaV = np.zeros((self.hidden_dims, self.vocab_size))
		sum_deltaU = np.zeros((self.hidden_dims, self.hidden_dims))
		for t in reversed(range(len(x))):
			d_t_vector = make_onehot(d[t],self.vocab_size)
			delta_out_t = (d_t_vector - y[t]) * net_out_grad[t]
			sum_deltaW += np.outer(delta_out_t, s[t])
			delta_in = np.zeros((len(x), self.hidden_dims))
			for tau in range(0, 1 + steps):
				if t - tau < 0: break
				if tau == 0:
					delta_in[t - tau] = np.dot(self.W.T, delta_out_t) * net_in_grad[t]
				else:
					delta_in[t - tau] = np.dot(self.U.T, delta_in[t - tau + 1]) * net_in_grad[t - tau]
				sum_deltaV += np.outer(delta_in[t - tau], make_onehot(x[t - tau],self.vocab_size))
				sum_deltaU += np.outer(delta_in[t - tau], s[t - tau-1])

		self.deltaW = sum_deltaW
		self.deltaV = sum_deltaV
		self.deltaU = sum_deltaU
