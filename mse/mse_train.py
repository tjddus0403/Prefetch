import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_pickle('../data/dataset_train_proc.trc')

split_idx = int(len(dataset)*0.8)
trainset = dataset[:split_idx]
valset = dataset[split_idx:]
# 정규화
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
# DataLoader 만들기
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
		x = x.float()
		hidden = (hidden[0].to(x.dtype), hidden[1].to(x.dtype))
		out, hidden = self.lstm(x, hidden)
		out = self.output_layer(out)

		out = out.view(batch_size, -1)
		out = out[:, -1]

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
# counter ; 현재까지 학습한 데이터 수
counter = 0
print_term = 100
model.train()
# clip ; gradient exploding 문제 방지하기 위해 gradient norm이 clip을 넘지 않도록 gradient clippiing
clip = 5
# 검증 손실의 최소값을 무한대로 설정 (검증 손실이 감소할 때마다 모델 파라미터를 저장하기 위해 사용)
valid_loss_min = np.Inf

# 전체 데이터 epoch 수만큼 반복 학습 시작
for i in range(epochs):
	# 모델의 hidden state 초기화
	h = model.init_hidden(batch_size)
	# 학습데이터를 mini-batch 단위로 불러옴
	for inputs, labels in train_loader:
		# 학습 데이터 수 증가
		counter += 1
		# hidden state를 detach하여 메모리 사용량 줄이기
		h = tuple([e.data for e in h])
		# input data와 label을 GPU로 옮기고, 데이터 자료형을 float로 변경
		inputs, labels = inputs.float().to(device), labels.to(device)
		# 모델 gradient 초기화
		model.zero_grad()

		output, h = model(inputs, h)
		# 출력과 레이블 비교하여 손실 계산
		loss = criterion(output.squeeze(), labels.float())
		# 손실을 역전파하여 gradient 계산
		loss.backward()
		
		# gradient clipping
		nn.utils.clip_grad_norm_(model.parameters(), clip)
		# optimizer 이용해 파라미터 업데이트
		optimizer.step()

		if counter % print_term == 0:
			# 검증 데이터에 대하여 모델의 hidden state 초기화
			val_h = model.init_hidden(batch_size)
			val_losses=[]
			# 모델 평가모드 전환
			model.eval()
			# 검증 데이터를 mini-batch 단위로 불러옴
			for inp, lab in val_loader:
				val_h = tuple([each.data for each in val_h])
				inp, lab = inp.to(device), lab.to(device)
				out, val_h = model(inp, val_h)
				val_loss = criterion(out.squeeze(), lab.float())
				# val_losses 리스트에 현재 batch의 손실 값 추가
				val_losses.append(val_loss.item())
			# 모델 훈련모드 전환
			model.train()
			print("Epoch: {}/{}...".format(i+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
			# 이전 검증 손실 값보다 현재 검증 손실 값이 낮은 경우, 모델의 상태 딕셔너리를 파일로 저장 (모델 저장)
			if np.mean(val_losses) <= valid_loss_min:
				torch.save(model.state_dict(), './state_dict.pt')
				print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
				valid_loss_min = np.mean(val_losses)
