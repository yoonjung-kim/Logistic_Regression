import sys
import random
import pickle
import data_loader as dl
from math import exp, log


class LogisticRegression():
	def __init__(self, 
				data_file, 
				step_size=0.5, 
				tolerance=10**-5,
				max_iter=10):
		self.data_file = data_file
		self.int_minmax, self.category_dicts, self.input_size = dl.load_data_structure(data_file)
		
		self.trained = False
		self.step_size = step_size
		self.tolerance = tolerance
		self.max_iter = max_iter
		self.w = None
		self.w0 = None
		self.set_inital_weights()

	def set_inital_weights(self):
		# assign random numbers between 0 and 1 as weights
		self.w = [random.uniform(0,1) for i in range(self.input_size)]
		# assign random numbers between 0 and 1 as intercept
		self.w0 = random.uniform(0,1)

	def compute_loss(self, w0, w, data_type):
		"""Compute mean log loss with given weights"""
		loss = 0
		num_data = 0
		for x_dict, y in dl.data_gen(self.data_file, 
									self.int_minmax, 
									self.category_dicts, 
									data_type):
			dot_prod = self.compute_w_x_dot(w0, w, x_dict)
			loss += -y*dot_prod+self.log_sum_exp(dot_prod)
			num_data += 1
		return loss/num_data

	def log_sum_exp(self, x):
		"""
		Compute log(1+exp(x))=log(exp(0)+exp(x))
		Reference: https://lingpipe-blog.com/2009/06/25/log-sum-of-exponentials/
		"""
		max_num = max(0, x)
		return max_num+log(exp(0-max_num)+exp(x-max_num))

	def compute_grad(self, w0, w, data_type):
		"""Compute Gradient with given weights"""
		dw0 = 0
		dw = [0]*len(w)
		num_data = 0
		for x_dict, y in dl.data_gen(self.data_file, 
									self.int_minmax, 
									self.category_dicts, 
									data_type):
			prob = self.compute_prob(w0, w, x_dict)
			dw0 += -y + prob
			for loc, x_val in x_dict.items():
				dw[loc] += x_val * (-y + prob)
			num_data += 1
		return dw0/num_data, [dwj/num_data for dwj in dw]

	def compute_w_x_dot(self, w0, w, x_dict):
		"""
		Compute inner product of weights and feature vector
		w0 + w1x1 + w2x2 + ...
		"""
		dot_prod = w0
		for loc, x_val in x_dict.items():
			dot_prod += w[loc]*x_val
		return dot_prod

	def compute_prob(self, w0, w, x_dict):
		"""Compute conversion probability with given weights and feature vector"""
		dot_prod = self.compute_w_x_dot(w0, w, x_dict)
		log_prob = dot_prod - self.log_sum_exp(dot_prod)
		return exp(log_prob)

	def train(self):
		"""Train the model with training data"""
		print("Training...")
		prev_loss = self.compute_loss(self.w0, self.w, 'train')
		for i in range(self.max_iter):
			# gradient descent
			dw0, dw = self.compute_grad(self.w0, self.w, 'train')
			self.w0 -= self.step_size * dw0
			self.w = [wj-self.step_size*dwj for wj, dwj in zip(self.w, dw)]
			curr_loss = self.compute_loss(self.w0, self.w, 'train')
			if i%(self.max_iter/10)==0:
				print('iteration: {}, loss: {}'.format(i, curr_loss))
			if abs(curr_loss-prev_loss) < self.tolerance:
				print('# of iterations:',i)
				break
		self.trained = True
		print('Mean log loss of TRAIN data:', curr_loss)

	def test(self, verbose=False):
		"""Test the model with test data"""
		if not self.trained: self.train()
		loss = self.compute_loss(self.w0, self.w, 'test')
		print('Mean log loss of TEST data:', loss)

	def test_single(self, single_input):
		"""Test the model with a single line input (for server usage)"""
		input_vec = dl.load_test_data_from_line(single_input, self.int_minmax, self.category_dicts)
		if not input_vec: return 'Wrong input'
		prob = self.compute_prob(self.w0, self.w, input_vec)
		return prob

	def save_model(self, filename):
		"""Save the Logistic Regression model"""
		pickle.dump(self, open(filename, 'wb'))
		print('Model saved in',filename)

def main(data_file,model_file):
	lr = LogisticRegression(data_file)
	lr.train()
	lr.test()
	lr.save_model(model_file)

def load_and_test(model_file, test_line):
	"""For test purposes"""
	model = pickle.load(open(model_file, 'rb'))
	print('Model loaded')
	print(model.test_single(test_line))
		
if __name__=='__main__':
	"""
	Should run this program passing two arguments:
	python logistic_regression.py <data file path> <model file path(where the model should be saved)>
	ex) python logistic_regression.py ./data/data.txt ./model/model.dat
	"""
	if len(sys.argv) != 3:
		print('Invalid number of arguments')
		exit()
	data_file = sys.argv[1] # './data/data.txt'
	model_file = sys.argv[2] # './model/model.dat'
	main(data_file, model_file)
	pass
