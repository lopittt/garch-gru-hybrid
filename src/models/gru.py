# src/models/gru.py
# PyTorch GRU network for volatility forecasting
# Learns patterns from historical volatility and returns data
# RELEVANT FILES: hybrid.py, preprocessor.py, main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class GRUNetwork(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[64, 32], output_size=1, dropout=0.1, sequence_length=6):
        super().__init__()
        
        # Simplified architecture for limited training data
        # Reduced from [512, 256, 128] to [64, 32] neurons
        
        self.sequence_length = sequence_length
        
        # Input normalization to prevent mode collapse
        self.input_norm = nn.BatchNorm1d(self.sequence_length, track_running_stats=False)
        
        # Single GRU layer with smaller hidden size
        self.gru1 = nn.GRU(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second smaller GRU layer
        self.gru2 = nn.GRU(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        # Simplified output layer
        self.fc1 = nn.Linear(hidden_sizes[-1], output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # Initialize weights to prevent mode collapse
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights to promote diversity"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # Small positive bias
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.01)
        
    def forward(self, x):
        # Normalize input if batch size > 1
        if x.size(0) > 1:
            x = self.input_norm(x)
        
        # First GRU layer
        out, _ = self.gru1(x)
        out = self.leaky_relu(out)
        out = self.dropout1(out)
        
        # Second GRU layer
        out, _ = self.gru2(out)
        out = self.leaky_relu(out)
        out = self.dropout2(out)
        
        # Take only the last timestep
        out = out[:, -1, :]
        
        # Output layer
        out = self.fc1(out)
        
        # Ensure positive output with ReLU + small offset
        # This prevents zero outputs while maintaining gradient flow
        out = F.relu(out) + 1e-6
        
        return out

class GRUModel:
    def __init__(self, sequence_length=6):
        self.sequence_length = sequence_length
        # Pass sequence length to network for batch norm
        self.model = GRUNetwork(sequence_length=sequence_length)
        
        # Adjusted optimizer for small dataset
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=0.005,  # Higher learning rate for faster convergence
            weight_decay=0.0001,  # Lower weight decay
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Use MSE loss which is standard for volatility prediction
        self.criterion = nn.MSELoss()
        
        # More aggressive scheduler for small dataset
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=False,
            min_lr=1e-6
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def prepare_sequences(self, returns, garch_conditional_vol, targets):
        """Prepare sequences using historical GARCH conditional volatility
        
        Args:
            returns: Historical returns (absolute values)
            garch_conditional_vol: GARCH conditional volatility (from fitted model)
            targets: GKYZ realized volatility (what we're predicting)
        """
        sequences = []
        sequence_targets = []
        
        min_len = min(len(returns), len(garch_conditional_vol), len(targets))
        
        for i in range(self.sequence_length, min_len):
            # Use HISTORICAL conditional volatility, not forecasts
            seq = np.column_stack([
                returns[i-self.sequence_length:i],
                garch_conditional_vol[i-self.sequence_length:i]  # Historical values only
            ])
            sequences.append(seq)
            # Use iloc for position-based indexing to avoid warning
            sequence_targets.append(targets.iloc[i] if hasattr(targets, 'iloc') else targets[i])
            
        return np.array(sequences), np.array(sequence_targets)
    
    def train(self, train_sequences, train_targets, val_sequences, val_targets, epochs=150, batch_size=500, max_train_size=500):
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
        patience = 25  # More patience for small dataset
        patience_counter = 0
        min_delta = 1e-7  # Minimum improvement threshold
        train_losses = []
        val_losses = []
        
        print(f"Training with {len(train_sequences)} samples, {len(val_sequences)} validation samples")
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            batch_count = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs.squeeze(), batch_y)
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                train_loss += loss.item()
                batch_count += 1
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_predictions = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    val_loss += self.criterion(outputs.squeeze(), batch_y).item()
                    val_predictions.extend(outputs.squeeze().cpu().numpy())
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Check prediction diversity
            val_pred_std = np.std(val_predictions) if val_predictions else 0
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Model checkpointing with minimum improvement threshold
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_pred_std': val_pred_std
                }, 'best_gru_model.pth')
            else:
                patience_counter += 1
            
            # Check for mode collapse
            if val_pred_std < 1e-6 and epoch > 10:
                print(f"Warning: Possible mode collapse detected (std={val_pred_std:.8f})")
                # Try to recover by reducing learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.5
            
            if epoch % 5 == 0:  # More frequent reporting
                print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Pred Std: {val_pred_std:.6f}")
            
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