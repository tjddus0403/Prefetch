import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# 테스트 데이터셋 세팅
testset = pd.read_pickle('../data/dataset_test_proc.trc')

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
        out = out + 1e-8
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device), weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

# 모델 가져오기
model = Model(1, 4, 1, 50, 3)
model.load_state_dict(torch.load('state_dict.pt'))
model.to(device)
test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)

model.eval()
criterion = nn.BCEWithLogitsLoss()
for inputs, labels in test_loader :
    h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    output, h =model(inputs, h)
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))

import matplotlib.pyplot as plt

plt.plot(range(1, len(test_losses)+1), test_losses)
plt.title("test result")

plt.xlabel("Num")
plt.ylabel("Losses")
plt.legend()
plt.show()
plt.savefig('result.png')
