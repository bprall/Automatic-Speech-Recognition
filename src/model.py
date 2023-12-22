import torch.nn as nn
import torch.nn.functional as F

class CNNLayerNorm(nn.Module):
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        x = x.transpose(2, 3).contiguous()
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous()


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, batch_first):
        super(BidirectionalLSTM, self).__init__()

        self.BiLSTM = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiLSTM(x)
        x = self.dropout(x)
        return x

class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn_layers, n_lstm_layers, lstm_dim, n_class, n_feats, dropout):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2

        self.cnn = nn.Conv2d(1, 32, 3, stride=2, padding=3//2)

        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])

        self.fully_connected = nn.Linear(n_feats * 32, lstm_dim)
        
        self.bilstm_layers = nn.Sequential(*[
            BidirectionalLSTM(input_size=lstm_dim if i==0 else lstm_dim*2,
                               hidden_size=lstm_dim, num_layers=1, dropout=dropout, batch_first=i==0)
            for i in range(n_lstm_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(lstm_dim*2, lstm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2)
        x = self.fully_connected(x)
        x = self.bilstm_layers(x)
        x = self.classifier(x)
        return x
