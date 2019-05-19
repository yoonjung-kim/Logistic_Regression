import warnings

def load_data_structure(filename):
	"""
	Read the file and build the category dictrionary to make dummy variables
	(8 integer features and 9 categorical features)
	"""
	print("Loading data structure...")
	int_minmax = [[float('inf'),-float('inf')] for j in range(8)]
	category_sets = [set() for j in range(9)]
	with open(filename) as f:
		for i, line in enumerate(f):
			if i%10 in [8,9,0]: # only for train data
				continue
			arr = line.rstrip('\n').split('\t')
			for j in range(8):
				# get min and max values for integer features
				if arr[j+2]!='':
					(minv, maxv) = int_minmax[j]
					if minv > int(arr[j+2]): 
						int_minmax[j][0] = int(arr[j+2])
					elif maxv < int(arr[j+2]):
						int_minmax[j][1] = int(arr[j+2])
			for j in range(9):
				# get categorical values
				if arr[j+10]!='':
					category_sets[j].add(arr[j+10])
	# categorical values to dictionary
	category_dicts = [dict(zip(s,range(len(s)))) for s in category_sets]
	input_length = 8 + sum([len(c) for c in category_dicts])
	print('done')
	return int_minmax, category_dicts, input_length

def raw_to_input(raw_vec, int_minmax, category_dicts):
	"""
	Refine a raw data line into model input
	by parsing the line and imputing missing values
	"""
	if len(raw_vec)!=17: return
	input_vec = {}
	# integer features: impute missing value with min value, normalize values (0-1)
	for i in range(8):
		if raw_vec[i]=='':
			val = int_minmax[i][0]
		else:
			val = int(raw_vec[i])
		input_vec[i] = (val-int_minmax[i][0]) / (int_minmax[i][1]-int_minmax[i][0])
	# categorical features: make dummy variables, create 'missing' category
	pivot = 8
	for i in range(9):
		if raw_vec[i+8] in category_dicts[i]:
			loc = pivot + category_dicts[i][raw_vec[i+8]]
			input_vec[loc] = 1
		pivot = pivot + len(category_dicts[i])
	return input_vec

def load_test_data_from_line(line, int_minmax, category_dicts):
	"""A single line to model input (for server usage)"""
	arr = line.rstrip('\n').split('\t')
	if len(arr)!=19: return False
	input_vec = raw_to_input(arr[-17:], int_minmax, category_dicts)
	return input_vec

def data_gen(filename, int_minmax, category_dicts, data_type):
	"""A generator that reads a line from the data file and produces a model input"""
	if data_type=='test':
		remainder = [8,9,0]
	else:
		remainder = [1,2,3,4,5,6,7]

	with open(filename) as f:
		for i, line in enumerate(f):
			if i%10 not in remainder: continue
			arr = line.rstrip('\n').split('\t')
			label = 0 if arr[1]=='' else 1
			input_vec = raw_to_input(arr[-17:], int_minmax, category_dicts)
			yield input_vec, label

if __name__ == '__main__':
	filename = './data/data5.txt'
	int_minmax, category_dicts, input_length = load_data_structure(filename)
	pass
