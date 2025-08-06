# src/models/gru.py
# PyTorch GRU network for volatility forecasting
# Learns patterns from historical volatility and returns data
# RELEVANT FILES: hybrid.py, preprocessor.py, main.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class GRUNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_sizes=[512, 256, 128], output_size=1, dropout=0.3):
        super().__init__()
        
        self.gru1 = nn.GRU(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.gru2 = nn.GRU(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.gru3 = nn.GRU(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_sizes[2], output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out, _ = self.gru2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        out, _ = self.gru3(out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        out = self.fc(out[:, -1, :])
        return out

class GRUModel:
    def __init__(self, sequence_length=6):
        self.sequence_length = sequence_length
        self.model = GRUNetwork()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0009)
        self.criterion = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def prepare_sequences(self, returns, volatility, garch_vol):
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(returns)):
            seq = np.column_stack([
                returns[i-self.sequence_length:i],
                volatility[i-self.sequence_length:i],
                garch_vol[i-self.sequence_length:i]
            ])
            sequences.append(seq)
            targets.append(volatility[i])
            
        return np.array(sequences), np.array(targets)
    
    def train(self, train_sequences, train_targets, val_sequences, val_targets, epochs=50, batch_size=100):
        train_data = TensorDataset(
            torch.FloatTensor(train_sequences),
            torch.FloatTensor(train_targets)
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        val_data = TensorDataset(
            torch.FloatTensor(val_sequences),
            torch.FloatTensor(val_targets)
        )
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    val_loss += self.criterion(outputs.squeeze(), batch_y).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_gru_model.pth')
    
    def predict(self, sequences):
        self.model.eval()
        with torch.no_grad():
            sequences = torch.FloatTensor(sequences).to(self.device)
            predictions = self.model(sequences)
        return predictions.cpu().numpy()
    
    def load_best_model(self):
        self.model.load_state_dict(torch.load('best_gru_model.pth', weights_only=True))
        return self