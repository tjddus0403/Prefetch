import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_pickle('data/dataset_train_proc.trc')

split_idx = int(len(dataset)*0.8)
trainset = dataset[:split_idx]
valset = dataset[split_idx:]

scaler_x = MinMaxScaler()
scaler_x.fit(trainset)
scaled_x_trainset = scaler_x.transform(trainset)
scaled_x_valset = scaler_x.transform(valset)

train_inp = np.array(scaled_x_trainset[:, :-1].astype(float))
train_label = np.array(scaled_x_trainset[:,-1].astype(float))

val_inp = np.array(scaled_x_valset[:, :-1].astype(float))
val_label = np.array(scaled_x_valset[:, -1].astype(float))

train_data = TensorDataset(torch.from_numpy(train_inp).unsqueeze(2), torch.from_numpy(train_label))
val_data = TensorDataset(torch.from_numpy(val_inp).unsqueeze(2), torch.from_numpy(val_label))

batch_size = 10

train_loader = DataLoader(train_data, shuffle=False, batch_size = batch_size)
val_loader = DataLoader(val_data, shuffle=False, batch_size = batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim, seq_len, n_layers):
		super(Model, self).__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.seq_len = seq_len
		self.n_layers = n_layers

		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = n_layers, batch_first = True)
		self.output_layer = nn.Linear(hidden_dim, output_dim)

	def forward(self, x, hidden):
		batch_size = x.size(0)
		x = x.type_as(hidden[0])
		out, hidden = self.lstm(x, hidden)
		out = self.output_layer(out)

		out = out.view(batch_size, -1)
		out = out[:, -1]
		out = out + 1e-8
		return out, hidden

	def init_hidden(self, batch_size):
		weight = next(self.parameters()).data
		hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device), weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
		return hidden

input_dim = 1
hidden_dim = 4
output_dim = 1
seq_len = 50
n_layers = 3

model = Model(input_dim, hidden_dim, output_dim, seq_len, n_layers)
model.to(device)

lr = 0.001
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 10
counter = 0
print_term = 100
model.train()
clip = 5
valid_loss_min = np.Inf

for i in range(epochs):
	h = model.init_hidden(batch_size)
	for inputs, labels in train_loader:
		counter += 1
		h = tuple([e.data for e in h])
		inputs, labels = inputs.float().to(device), labels.to(device)
		model.zero_grad()
		
		#print(inputs)
		#print(inputs.size())
		output, h = model(inputs, h)
		loss = criterion(output.squeeze(), labels.float())
		#print(output.size())
		#print(labels.size())
		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), clip)
		optimizer.step()

		if counter % print_term == 0:
			val_h = model.init_hidden(batch_size)
			val_losses=[]
			model.eval()
			for inp, lab in val_loader:
				val_h = tuple([each.data for each in val_h])
				inp, lab = inp.to(device), lab.to(device)
				out, val_h = model(inp, val_h)
				val_loss = criterion(out.squeeze(), lab.float())
				val_losses.append(val_loss.item())

			model.train()
			print("Epoch: {}/{}...".format(i+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
			if np.mean(val_losses) <= valid_loss_min:
				torch.save(model.state_dict(), './state_dict.pt')
				print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
				valid_loss_min = np.mean(val_losses)
