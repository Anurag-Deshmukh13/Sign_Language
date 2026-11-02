
import torch
import torch.nn as nn


class SignLanguageBiLSTM(nn.Module):
    def __init__(
        self,
        input_size=225,
        hidden_size=64,
        num_layers=1,
        num_classes=9,
        dropout=0.4,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)

        forward_hidden = h_n[-2]
        backward_hidden = h_n[-1]
        combined = torch.cat(
            (forward_hidden, backward_hidden), dim=1
        )

        combined = self.dropout(combined)
        return self.fc(combined)