from torch import nn
import torch


class fukumori2021RNN(nn.Module):
    def __init__(self, input_size, ):
        super().__init__()
        self.input_size = input_size
        self.LSTM_1 = nn.LSTM(input_size=input_size, hidden_size=8, num_layers=1)
        self.tanh = nn.Tanh()
        self.avgPool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.selfattention = nn.MultiheadAttention(num_heads=1, embed_dim=8)
        self.LSTM_2 = nn.LSTM(input_size=8, hidden_size=8, num_layers=1)
        self.classifier = nn.Linear(96, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.randn(1, self.input_size, 8)
        c0 = torch.randn(1, self.input_size, 8)

        # First LSTM
        x, (_, _) = self.LSTM_1(x)
        x = self.tanh(x)
        x = self.avgPool(x.transpose(1, 2))
        x = x.transpose(1, 2)
        # Self-attention Layer
        x_attention, attention_weights = self.selfattention(x, x, x)

        x = x + x_attention

        # Second LSTM
        x, (_, _) = self.LSTM_2(x)
        x = self.tanh(x)
        x = self.avgPool(x.transpose(1, 2))
        x = x.transpose(1, 2)

        # Classifier
        x = self.classifier(x.flatten(1))

        return self.sigmoid(x), attention_weights
