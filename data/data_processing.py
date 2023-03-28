import pandas as pd

dataset = pd.read_table('dataset_test_10.trc', header=None)

dataset = dataset.astype(int)
for i in range(len(dataset[0])):
	dataset[0][i] = dataset[0][i] >> 22

seq_len = 50
batch = 100

for i in range(1, seq_len+1):
	dataset[i] = 'a'
	for j in range(len(dataset[0])-seq_len):
		dataset[i][j] = dataset[0][i+j]

dataset.drop(dataset.index[len(dataset[0])-seq_len : ], axis = 0, inplace = True)

dataset.to_pickle('dataset_test_proc.trc')
