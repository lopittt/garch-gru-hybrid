# src/models/gru.py
# PyTorch GRU network for volatility forecasting
# Learns patterns from historical volatility and returns data
# RELEVANT FILES: hybrid.py, preprocessor.py, main.py

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class GRUNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[512, 256, 128], output_size=1, dropout=0.3):
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
        
    def prepare_sequences(self, returns, garch_forecasts, targets):
        """Prepare sequences using GARCH forecasts as input (not historical volatility)
        
        Key change: Input is returns + GARCH forecasts, target is realized volatility
        """
        sequences = []
        sequence_targets = []
        
        # Ensure all inputs have the same length
        min_len = min(len(returns), len(garch_forecasts), len(targets))
        
        for i in range(self.sequence_length, min_len):
            seq = np.column_stack([
                returns[i-self.sequence_length:i],
                garch_forecasts[i-self.sequence_length:i]  # Use GARCH forecasts, not historical volatility
            ])
            sequences.append(seq)
            sequence_targets.append(targets[i])
            
        return np.array(sequences), np.array(sequence_targets)
    
    def train(self, train_sequences, train_targets, val_sequences, val_targets, epochs=50, batch_size=32, max_train_size=500):
        # Limit training data to 500 data points as specified in paper
        if len(train_sequences) > max_train_size:
            indices = np.random.choice(len(train_sequences), max_train_size, replace=False)
            train_sequences = train_sequences[indices]
            train_targets = train_targets[indices]
        
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
        patience = 10
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        print(f"Training with {len(train_sequences)} samples, {len(val_sequences)} validation samples")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    val_loss += self.criterion(outputs.squeeze(), batch_y).item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Model checkpointing and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, 'best_gru_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        return {'train_losses': train_losses, 'val_losses': val_losses}
    
    def predict(self, sequences):
        self.model.eval()
        with torch.no_grad():
            sequences = torch.FloatTensor(sequences).to(self.device)
            predictions = self.model(sequences)
        return predictions.cpu().numpy()
    
    def load_best_model(self):
        checkpoint = torch.load('best_gru_model.pth', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['val_loss']:.6f}")
        return self