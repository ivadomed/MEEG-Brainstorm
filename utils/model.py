from torch import nn
import torch


class fukumori2021RNN(nn.Module):
    def __init__(self, input_size, ):
        super().__init__()
        self.input_size = input_size
        self.LSTM_1 = nn.LSTM(input_size=input_size,
                              hidden_size=8,
                              num_layers=1)
        self.tanh = nn.Tanh()
        self.avgPool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.selfattention = nn.MultiheadAttention(num_heads=1, embed_dim=8)
        self.LSTM_2 = nn.LSTM(input_size=8, hidden_size=8, num_layers=1)
        self.classifier = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.randn(1, 512, 8)
        c0 = torch.randn(1, 512, 8)
        # First LSTM
        x, (_, _) = self.LSTM_1(x) #, (h0, c0))
        x = self.tanh(x)
        x = self.avgPool(x.transpose(1, 2))
        x = x.transpose(1, 2)

        # Self-attention Layer
        x_attention, attention_weights = self.selfattention(x, x, x)
        x = x + x_attention

        h0 = torch.randn(1, 128, 8)
        c0 = torch.randn(1, 128, 8)
        # Second LSTM
        x, (_, _) = self.LSTM_2(x) #, (h0, c0))
        x = self.tanh(x)
        x = self.avgPool(x.transpose(1, 2))
        x = x.transpose(1, 2)

        # Classifier
        x = self.classifier(x.flatten(1))
        x = self.sigmoid(x)

        return x, attention_weights
