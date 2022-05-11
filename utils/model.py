import torch
from torch import nn


class fukumori2021RNN():
    def __init__(self, input_size, ):
        self.LSTM_1 = nn.LSTM(input_size=input_size, hidden_size=8, num_layers=1)
        self.tanh = nn.Tanh()
        self.avgPool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.selfattention = nn.MultiheadAttention(num_heads=1, embed_dim=8)
        self.LSTM_2 = nn.LSTM(input_size=input_size/4, hidden_size=8, num_layers=1)
        self.classifier = nn.Linear(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.LSTM_1(x)
        x = self.tanh(x)
        x = self.avgPool(x)
        x, attention_weights = self.selfattention(x) 
        x = self.LSTM_2(x)
        x = self.tanh(x)
        x = self.avgPool(x)
        x = self.classifier(x.flatten())
         
        return self.sigmoid(x), attention_weights