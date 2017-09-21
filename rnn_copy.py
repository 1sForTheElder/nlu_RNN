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
			s[t] = sigmoid(self.V.dot(make_onehot(x[t],self.vocab_size).T)+self.U.dot(s[t-1].T))
			y[t] = softmax(self.W.dot(s[t].T))
		return y,s

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
		V_sum = np.zeros((self.hidden_dims, self.vocab_size))
		U_sum = np.zeros((self.hidden_dims, self.hidden_dims))
		W_sum = np.zeros((self.vocab_size, self.hidden_dims))
		for t in reversed(range(len(x))):
			##########################
			# --- your code here --- #
			##########################
			W1_loss = ((make_onehot(d[t], self.vocab_size)) - y[t])
			s_net_new = s[t]*((-s[t])+1.)
			WTL = self.W.T.dot(W1_loss)
			in_p = WTL * s_net_new
			V_sum += np.outer(in_p, make_onehot(x[t], self.vocab_size))
			U_sum += np.outer(in_p, s[t - 1])
			W_sum += np.outer(W1_loss, s[t])
		self.deltaV = V_sum
		self.deltaW = W_sum
		self.deltaU = U_sum


	def acc_deltas_bptt(self, x, d, y, s, steps):
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
		net_out_grad = np.ones(len(x))
		net_in_grad = np.array([s_t * (np.ones(len(s_t)) - s_t) for s_t in s])
		sum_deltaW = np.zeros((self.vocab_size, self.hidden_dims))
		sum_deltaV = np.zeros((self.hidden_dims, self.vocab_size))
		sum_deltaU = np.zeros((self.hidden_dims, self.hidden_dims))

		for t in reversed(range(len(x))):
			##########################
			# --- your code here --- #
			##########################
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
		for i in range(0,len(softm)):
			loss -= np.log(softm[i][d[i]])
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
		for i,p in zip(X,D):
			mean_loss+=self.compute_loss(i,p)
			totlen+=len(i)
		mean_loss/=totlen
		return mean_loss

	def generate_sequence(self, start, end, maxLength):
		'''
		generate a new sequence, using the RNN

		starting from the word-index for a start symbol, generate some output until the word-index of an end symbol is generated, or the sequence
		exceed maxLength

		HINT: make use of the "multinomial_sample" method in rnnmath.py !!!

		start		word index of start symbol (the symbol <s> in a vocabulary)
		end			word index of end symbol (the symbol </s> in a vocabulary)
		maxLength	maximum length of the generated sequence

		return sequence, loss

		sequence	the generated sequence as a list of word indices, e.g., [4, 2, 3, 5]
		loss		the loss of the generated sequence
		'''
		sequence = [start]
		loss = 0.
		x = [start]

		##########################
		# --- your code here --- #
		##########################
		sequence = [start]
		loss = 0.
		x = [start]
		while True:
			# predict next word from current sequene x
			y, s = self.predict(x)

			# generate next word by sampling the word  according to the last element of y
			word_index = multinomial_sample(y[-1])

			x.append(word_index)
			sequence.append(word_index)
			pointwise_loss = -np.log(y[-1][word_index])
			loss += pointwise_loss

			if word_index == end or len(sequence) > maxLength: break

		return sequence, loss

	def train(self, X, D, X_dev, D_dev, epochs=10, learning_rate=0.5, anneal=5, back_steps=0, batch_size=100, min_change=0.0001, log=True):
		'''
		train the RNN on some training set X, D while optimizing the loss on a dev set X_dev, D_dev

		DO NOT CHANGE THIS

		training stops after the first of the following is true:
			* number of epochs reached
			* insignificant change observed for more than 2 consecutive epochs

		X				a list of input vectors, e.g., 		[[0, 4, 2], [1, 3, 0]]
		D				a list of desired outputs, e.g., 	[[4, 2, 3], [3, 0, 3]]
		X_dev			a list of input vectors, e.g., 		[[0, 4, 2], [1, 3, 0]]
		D_dev			a list of desired outputs, e.g., 	[[4, 2, 3], [3, 0, 3]]
		epochs			maximum number of epochs (iterations) over the training set. default 10
		learning_rate	initial learning rate for training. default 0.5
		anneal			positive integer. if > 0, lowers the learning rate in a harmonically after each epoch.
						higher annealing rate means less change per epoch.
						anneal=0 will not change the learning rate over time.
						default 5
		back_steps		positive integer. number of timesteps for BPTT. if back_steps < 2, standard BP will be used. default 0
		batch_size		number of training instances to use before updating the RNN's weight matrices.
						if set to 1, weights will be updated after each instance. if set to len(X), weights are only updated after each epoch.
						default 100
		min_change		minimum change in loss between 2 epochs. if the change in loss is smaller than min_change, training stops regardless of
						number of epochs left.
						default 0.0001
		log				whether or not to print out log messages. (default log=True)
		'''
		if log:
			stdout.write("\nTraining model for {0} epochs\ntraining set: {1} sentences (batch size {2})".format(epochs, len(X), batch_size))
			stdout.write("\nOptimizing loss on {0} sentences".format(len(X_dev)))
			stdout.write("\nVocab size: {0}\nHidden units: {1}".format(self.vocab_size, self.hidden_dims))
			stdout.write("\nSteps for back propagation: {0}".format(back_steps))
			stdout.write("\nInitial learning rate set to {0}, annealing set to {1}".format(learning_rate, anneal))
			stdout.write("\n\ncalculating initial mean loss on dev set")
			stdout.flush()

		t_start = time.time()

		loss_sum = sum([len(d) for d in D_dev])
		initial_loss = sum([self.compute_loss(X_dev[i], D_dev[i]) for i in range(len(X_dev))])/loss_sum

		if log or not log:
			stdout.write(": {0}\n".format(initial_loss))
			stdout.flush()

		prev_loss = initial_loss
		loss_watch_count = -1
		min_change_count = -1

		a0 = learning_rate

		best_loss = initial_loss
		bestU, bestV, bestW = self.U, self.V, self.W
		best_epoch = 0

		for epoch in range(epochs):
			if anneal > 0:
				learning_rate = a0/((epoch+0.0+anneal)/anneal)
			else:
				learning_rate = a0

			if log:
				stdout.write("\nepoch %d, learning rate %.04f" % (epoch+1, learning_rate))
				stdout.flush()

			t0 = time.time()
			count = 0

			# use random sequence of instances in the training set (tries to avoid local maxima when training on batches)
			permutation = np.random.permutation(range(len(X)))
			if log:
				stdout.write("\tinstance 1")
			for i in range(len(X)):
				c = i+1
				if log:
					stdout.write("\b"*len(str(i)))
					stdout.write("{0}".format(c))
					stdout.flush()
				p = permutation[i]
				x_p = X[p]
				d_p = D[p]

				y_p, s_p = self.predict(x_p)
				if back_steps == 0:
					self.acc_deltas(x_p, d_p, y_p, s_p)
				else:
					self.acc_deltas_bptt(x_p, d_p, y_p, s_p, back_steps)

				if i % batch_size == 0:
					self.deltaU /= batch_size
					self.deltaV /= batch_size
					self.deltaW /= batch_size
					self.apply_deltas(learning_rate)

			if len(X) % batch_size > 0:
				mod = len(X) % batch_size
				self.deltaU /= mod
				self.deltaV /= mod
				self.deltaW /= mod
				self.apply_deltas(learning_rate)

			loss = sum([self.compute_loss(X_dev[i], D_dev[i]) for i in range(len(X_dev))])/loss_sum

			if log:
				stdout.write("\tepoch done in %.02f seconds" % (time.time() - t0))
				stdout.write("\tnew loss: {0}".format(loss))
				stdout.flush()

			if loss < best_loss:
				best_loss = loss
				bestU, bestV, bestW = self.U.copy(), self.V.copy(), self.W.copy()
				best_epoch = epoch

			# make sure we change the RNN enough
			if abs(prev_loss - loss) < min_change:
				min_change_count += 1
			else:
				min_change_count = 0
			if min_change_count > 2:
				print("\n\ntraining finished after {0} epochs due to minimal change in loss".format(epoch+1))
				break

			prev_loss = loss

		t = time.time() - t_start

		if min_change_count <= 2:
			print("\n\ntraining finished after reaching maximum of {0} epochs".format(epochs))
		print("best observed loss was {0}, at epoch {1}".format(best_loss, (best_epoch+1)))

		print("setting U, V, W to matrices from best epoch")
		self.U, self.V, self.W = bestU, bestV, bestW

		return best_loss

if __name__ == "__main__":
	import sys
	from utils import *
	mode = sys.argv[1].lower()

	if mode == "estimate":
		'''
		starter code for parameter estimation.
		'''

		''' do NOT change until "your code here" '''
		data_folder = sys.argv[2]
		vocab_size = 2000
		train_size = 1000
		dev_size = 1000

		# get the data set vocabulary
		vocab = pd.read_table(data_folder + "/vocab.ptb.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
		num_to_word = dict(enumerate(vocab.index[:vocab_size]))
		word_to_num = invert_dict(num_to_word)

		# calculate loss vocabulary words due to vocab_size
		fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
		print("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocab_size, len(vocab), 100*(1-fraction_lost)))

		docs = load_dataset(data_folder + '/ptb-train.txt')
		S_train = docs_to_indices(docs, word_to_num)
		X_train, D_train = seqs_to_lmXY(S_train)

		# Load the dev set (for tuning hyperparameters)
		docs = load_dataset(data_folder + '/ptb-dev.txt')
		S_dev = docs_to_indices(docs, word_to_num)
		X_dev, D_dev = seqs_to_lmXY(S_dev)

		X = X_train[:train_size]
		D = D_train[:train_size]
		X_dev = X_dev[:dev_size]
		D_dev = D_dev[:dev_size]

		# q = best unigram frequency from omitted vocab
		# this is the best expected loss out of that set
		q = vocab.freq[vocab_size] / sum(vocab.freq[vocab_size:])

		##########################
		# --- your code here --- #
		##########################
		best_loss = -100
		best_hdim = 0
		best_lookback = 0
		best_lr = 0
		best_params = []
		dictt = {}
		R = RNN(2000,25)
		R1 = RNN(2000,50)
		epochh = 10
		hidmm = [25,50]
		learningratee = [0.5,0.1,0.05]
		backsteps = [0,2,5]
		for hh in hidmm:
			for ll in learningratee:
				for bb in backsteps:
					r = RNN(2000,hh)
					dict_name = str(hh)+' '+str(bb)+' '+str(ll)
					dictt[dict_name] = r.train(X=X,D=D,X_dev=X_dev,D_dev=D_dev,epochs=epochh,learning_rate=ll,back_steps=bb)
		# dictt['RNN_25_LB_0_LR_0.5'] = R.train(X=X,D=D,X_dev=X_dev,D_dev=D_dev,epochs=epochh,learning_rate=0.5,back_steps=0)
		# dictt['RNN_25_LB_0_LR_0.1'] = R.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.1,
		# 									  back_steps=0)
		# dictt['RNN_25_LB_0_LR_0.05'] = R.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.05,
		# 									  back_steps=0)
		# dictt['RNN_25_LB_2_LR_0.5'] = R.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.5,
		# 									  back_steps=2)
		# dictt['RNN_25_LB_2_LR_0.1'] = R.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.1,
		# 									  back_steps=2)
		# dictt['RNN_25_LB_2_LR_0.05'] = R.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.05,
		# 									  back_steps=2)
		# dictt['RNN_25_LB_5_LR_0.5'] = R.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.5,
		# 									  back_steps=5)
		# dictt['RNN_25_LB_5_LR_0.1'] = R.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.1,
		# 									  back_steps=5)
		# dictt['RNN_25_LB_5_LR_0.05'] = R.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.05,
		# 									  back_steps=5)
		# dictt['RNN_50_LB_0_LR_0.5'] = R1.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.5,
		# 									  back_steps=0)
		# dictt['RNN_50_LB_0_LR_0.1'] = R1.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.1,
		# 									  back_steps=0)
		# dictt['RNN_50_LB_0_LR_0.05'] = R1.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.05,
		# 									  back_steps=0)
		# dictt['RNN_50_LB_2_LR_0.5'] = R1.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.5,
		# 									  back_steps=2)
		# dictt['RNN_50_LB_2_LR_0.1'] = R1.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.1,
		# 									  back_steps=2)
		# dictt['RNN_50_LB_2_LR_0.05'] = R1.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.05,
		# 									  back_steps=2)
		# dictt['RNN_50_LB_5_LR_0.5'] = R1.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.5,
		# 									  back_steps=5)
		# dictt['RNN_50_LB_5_LR_0.1'] = R1.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.1,
		# 									  back_steps=5)
		# dictt['RNN_50_LB_5_LR_0.05'] = R1.train(X=X, D=D, X_dev=X_dev, D_dev=D_dev, epochs=epochh, learning_rate=0.05,
		# 									  back_steps=5)
		for dd in dictt.keys():
			print dd,'   ',dictt[dd]
			print '/n'
		loss_list = []
		best = 0.
		information = []
		for i in dictt.keys():
			if dictt[i]<best:
				# print i
				best = dictt[i]
				p1 = re.split(' ', i)
				best_hdim = p1[0]
				best_lookback = p1[1]
				best_lr = p1[2]
				best_params = [p1[0],p1[1],p1[2]]

		print 'best_param',best_params
		pen = open('new_result.txt','a')
		for g in dictt.keys():
			pen.write(g)
			pen.write('\t')
			pen.write(str(dictt[g]))
			pen.write('\n')



		print("\nestimation finished.\n\tbest loss {0}:\tbest hidden/lookback/learn rate: {1}/{2}/{3}".format(best_loss, best_params[0], best_params[1], best_params[2]))



##########################################################################################
		best_params[]
		train_size = 2000
		dev_size = 1000
		vocab_size = 2000

		hdim = int(best_params[0])
		lookback = int(best_params[1])
		lr = float(best_params[2])

		# get the data set vocabulary
		vocab = pd.read_table(data_folder + "/vocab.ptb.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
		num_to_word = dict(enumerate(vocab.index[:vocab_size]))
		word_to_num = invert_dict(num_to_word)

		# calculate loss vocabulary words due to vocab_size
		fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
		print("Retained %d words from %d (%.02f%% of all tokens)\n" % (vocab_size, len(vocab), 100*(1-fraction_lost)))

		docs = load_dataset(data_folder + '/ptb-train.txt')
		S_train = docs_to_indices(docs, word_to_num)
		X_train, D_train = seqs_to_lmXY(S_train)

		# Load the dev set (for tuning hyperparameters)
		docs = load_dataset(data_folder + '/ptb-dev.txt')
		S_dev = docs_to_indices(docs, word_to_num)
		X_dev, D_dev = seqs_to_lmXY(S_dev)

		X_train = X_train[:train_size]
		D_train = D_train[:train_size]
		X_dev = X_dev[:dev_size]
		D_dev = D_dev[:dev_size]

		# q = best unigram frequency from omitted vocab
		# this is the best expected loss out of that set
		q = vocab.freq[vocab_size] / sum(vocab.freq[vocab_size:])

		##########################
		# --- your code here --- #
		##########################
		R = RNN(vocab_size,hdim)


		run_loss = 0.
		run_loss = R.train(X=X_train, D=D_train, X_dev=X_dev, D_dev=D_dev, epochs=10, learning_rate=lr, back_steps=lookback)
		adjusted_loss = 0.
		fraqloss = fraq_loss(vocab, word_to_num, vocab_size)
		adjusted_loss = adjust_loss(run_loss,fracloss=fraqloss,q=0.5)
		np.save("rnn.Ufor2.npy", R.U)
		np.save("rnn.Vfor2.npy", R.V)
		np.save("rnn.Wfor2.npy", R.W)
		print "Unadjusted: %.03f" % np.exp(run_loss)
		print "Adjusted for missing vocab: %.03f" % np.exp(adjusted_loss)
        #
		# maxLength = 1000
        #
		# # get saved RNN matrices and setup RNN
		# U,V,W = np.load("rnn.Ufor2.npy"), np.load("rnn.Vfor2.npy"), np.load("rnn.Wfor2.npy")
		# vocab_size = 2000
		# hdim = 50
        #
		# r = RNN(vocab_size, hdim)
		# r.U = U
		# r.V = V
		# r.W = W
        #
		# # get vocabulary
		# vocab = pd.read_table(data_folder + "/vocab.ptb.txt", header=None, sep="\s+", index_col=0, names=['count', 'freq'], )
		# num_to_word = dict(enumerate(vocab.index[:vocab_size]))
		# word_to_num = invert_dict(num_to_word)
        #
		# ##########################
		# # --- your code here --- #
		# ##########################
        #
		# # predict something
		# end = [word_to_num['</S>']]
		# start = [word_to_num['<S>']]
		# sequence,loss = r.generate_sequence(start=start,end=end,maxLength=maxLength)
		# sequence2, loss2 = r.generate_sequence(start=start, end=end, maxLength=maxLength)
		# sequence3, loss3 = r.generate_sequence(start=start, end=end, maxLength=maxLength)
		# file1 = open("se1.txt",'a')
		# file2 = open('se2.txt','a')
		# file3 = open('se3.txt','a')
		# for qq in sequence:
		# 	file1.write(str(qq))
		# 	file1.write('\t')
		# file1.write(loss)
		# for qqq in sequence2:
		# 	file2.write(str(qqq))
		# 	file2.write('\t')
		# file2.write(loss2)
		# for qqqq in sequence3:
		# 	file3.write(str(qqqq))
		# 	file3.write('\t')
		# file3.write(loss3)
		# print sequence
		# print '\n'
		# print sequence2
		# print '\n'
		# print sequence3

