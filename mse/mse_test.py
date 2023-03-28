import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# 테스트 데이터셋 세팅
testset = pd.read_pickle('dataset_test_proc.trc')

scaler_x = MinMaxScaler()
scaler_x.fit(testset)
scaled_x_testset = scaler_x.transform(testset)

test_inp = np.array(scaled_x_testset[:, :-1].astype(float))
test_label = np.array(scaled_x_testset[:,-1].astype(float))

test_data = TensorDataset(torch.from_numpy(test_inp).unsqueeze(2), torch.from_numpy(test_label))

batch_size = 10

test_loader = DataLoader(test_data, shuffle = False, batch_size = batch_size)

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

# 모델 가져오기
model = Model(1, 4, 1, 50, 3)
model.load_state_dict(torch.load('./state_dict.pt'))
model.to(device)

# 테스트 손실 값을 저장하기 위한 리스트
test_losses = []
# 예측이 맞은 갯수를 저장하기 위한 변수 초기화
num_correct = 0
# hidden state와 cell state 초기화
h = model.init_hidden(batch_size)

# 모델 평가 모드 전환
model.eval()
# 손실 함수로 MSE 사용
criterion = nn.MSELoss()
for inputs, labels in test_loader :
	h = tuple([each.data for each in h])
	inputs, labels = inputs.to(device), labels.to(device)
	output, h =model(inputs, h)
	test_loss = criterion(output.squeeze(), labels.float())
	test_losses.append(test_loss.item())
	# 모델의 출력값을 반올림하여 이진 분류 결과 생성
	pred = torch.round(output.squeeze())
	# 예측 결과와 정답 데이터를 비교하여 True/False 반환
	correct_tensor = pred.eq(labels.float().view_as(pred))
	# 예측이 맞은 갯수 누적 저장됨 (num_correct에)
	correct = np.squeeze(correct_tensor.cpu().numpy())
	num_correct += np.sum(correct)

# 테스트 손실 값의 평균 출력
print("Test loss: {:.3f}".format(np.mean(test_losses)))
# 테스트 데이터셋에 대한 정확도 계산
test_acc = num_correct/len(test_loader.dataset)
# 테스트 데이터셋에 대한 정확도를 백분율로 출력
print("Test accuracy: {:.3f}%".format(test_acc*100))

# 그래프 그리기
import matplotlib.pyplot as plt

plt.plot(range(1, len(test_losses)+1), test_losses)
plt.title("test result")
# x축 : batch num, y축 : 해당 batch의 loss 값
plt.xlabel("Num")
plt.ylabel("Losses")
plt.legend()
plt.show()
plt.savefig('result.png')
